"""Reusable Transformers vision embedders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ModelClass = Literal["auto", "vision_submodel"]
ClsStrategy = Literal["token", "pooler", "mean_tokens", "same_as_patch"]
PatchStart = int | Literal["after_register_tokens"]


@dataclass(frozen=True)
class VisionModelSpec:
    key: str
    model_id: str
    processor_id: str | None = None
    model_class: ModelClass = "auto"
    model_kwargs: dict[str, object] | None = None
    processor_kwargs: dict[str, object] | None = None
    preprocess_kwargs: dict[str, object] | None = None
    forward_kwargs: dict[str, object] | None = None
    use_fast_processor: bool = True
    trust_remote_code: bool = False
    cls_strategy: ClsStrategy = "token"
    cls_token_index: int = 0
    patch_start_index: PatchStart = 1


class TransformersVisionEmbedder:
    """Common image embedder for ViT-like Transformers vision models."""

    def __init__(self, spec: VisionModelSpec, device: str = "cuda", cache_dir: str | None = None) -> None:
        self.spec = spec
        self.device = torch.device(device)
        processor_id = spec.processor_id or spec.model_id
        processor_kwargs = spec.processor_kwargs or {}
        model_kwargs = spec.model_kwargs or {}
        self.preprocess_kwargs = spec.preprocess_kwargs or {}
        self.forward_kwargs = spec.forward_kwargs or {}

        self.processor = AutoImageProcessor.from_pretrained(
            processor_id,
            cache_dir=cache_dir,
            use_fast=spec.use_fast_processor,
            trust_remote_code=spec.trust_remote_code,
            **processor_kwargs,
        )

        model = AutoModel.from_pretrained(
            spec.model_id,
            cache_dir=cache_dir,
            trust_remote_code=spec.trust_remote_code,
            **model_kwargs,
        )
        if spec.model_class == "vision_submodel":
            if not hasattr(model, "vision_model"):
                raise ValueError(f"Model {spec.model_id} does not expose a vision_model submodule.")
            model = model.vision_model
        self.model = model

        self.model.to(self.device)
        self.model.eval()

    def _resolve_patch_start_index(self) -> int:
        if self.spec.patch_start_index == "after_register_tokens":
            num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            return 1 + num_register_tokens
        return int(self.spec.patch_start_index)

    def _pool_patch_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        start_index = self._resolve_patch_start_index()
        if start_index >= hidden.shape[1]:
            raise ValueError(
                f"Patch token start index {start_index} leaves no patch tokens for {self.spec.model_id}; "
                f"hidden state has {hidden.shape[1]} tokens."
            )
        return hidden[:, start_index:, :].mean(dim=1)

    def _pool_cls_embedding(self, outputs, hidden: torch.Tensor, patch_embedding: torch.Tensor) -> torch.Tensor:
        if self.spec.cls_strategy == "same_as_patch":
            return patch_embedding
        if self.spec.cls_strategy == "mean_tokens":
            return hidden.mean(dim=1)
        if self.spec.cls_strategy == "pooler":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is None:
                raise ValueError(f"Model {self.spec.model_id} did not return pooler_output for cls_strategy='pooler'.")
            return pooler_output
        if self.spec.cls_token_index >= hidden.shape[1]:
            raise ValueError(
                f"CLS token index {self.spec.cls_token_index} is out of range for {self.spec.model_id}; "
                f"hidden state has {hidden.shape[1]} tokens."
            )
        return hidden[:, self.spec.cls_token_index, :]

    @torch.inference_mode()
    def embed_batch(self, images: Sequence[Image.Image]) -> dict[str, np.ndarray]:
        if not images:
            raise ValueError("embed_batch requires at least one image.")

        inputs = self.processor(images=list(images), return_tensors="pt", **self.preprocess_kwargs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs, **self.forward_kwargs)

        if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
            raise ValueError(f"Model {self.spec.model_id} did not return last_hidden_state.")

        hidden = outputs.last_hidden_state
        if hidden.ndim == 4:
            pooled = hidden.mean(dim=(-2, -1))
            cls_embedding = pooled
            patch_embedding = pooled
        elif hidden.ndim == 3:
            patch_embedding = self._pool_patch_tokens(hidden)
            cls_embedding = self._pool_cls_embedding(outputs, hidden, patch_embedding)
        elif hidden.ndim == 2:
            cls_embedding = hidden
            patch_embedding = hidden
        else:
            raise ValueError(
                "Expected hidden states with shape [batch, tokens, dim], "
                f"[batch, channels, height, width], or [batch, dim], got {tuple(hidden.shape)}."
            )

        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        patch_embedding = F.normalize(patch_embedding, p=2, dim=-1)

        return {
            "cls": cls_embedding.detach().cpu().numpy().astype(np.float32),
            "patch": patch_embedding.detach().cpu().numpy().astype(np.float32),
        }
