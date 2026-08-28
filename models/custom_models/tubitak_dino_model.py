"""TUBITAK fine-tuned DINOv3 embedder."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_PATH = Path(
    "XXX"
)


class TubitakDinoEmbedder:
    """Local Hugging Face DINOv3 checkpoint with explicit CLS and patch pooling."""

    def __init__(self, device: str = "cuda", model_path: str | Path = MODEL_PATH) -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TUBITAK DINO model path does not exist: {self.model_path}")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.num_registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

    def _pool_tokens(self, last_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if last_hidden_state.ndim != 3:
            raise ValueError(
                "Expected DINO hidden states with shape [batch, tokens, dim], "
                f"got {tuple(last_hidden_state.shape)}."
            )

        cls_embedding = last_hidden_state[:, 0, :]
        patch_start_idx = 1 + self.num_registers
        if patch_start_idx >= last_hidden_state.shape[1]:
            raise ValueError(
                f"Patch start index {patch_start_idx} leaves no patch tokens; "
                f"hidden state has {last_hidden_state.shape[1]} tokens."
            )
        patch_embedding = last_hidden_state[:, patch_start_idx:, :].mean(dim=1)
        return cls_embedding, patch_embedding

    @torch.inference_mode()
    def embed_batch(self, images: Sequence[Image.Image]) -> dict[str, np.ndarray]:
        if not images:
            raise ValueError("embed_batch requires at least one image.")

        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        cls_embedding, patch_embedding = self._pool_tokens(outputs.last_hidden_state)
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        patch_embedding = F.normalize(patch_embedding, p=2, dim=-1)

        return {
            "cls": cls_embedding.detach().cpu().numpy().astype(np.float32),
            "patch": patch_embedding.detach().cpu().numpy().astype(np.float32),
        }


def create_embedder(
    device: str = "cuda",
    cache_dir: str | None = None,
    model_key: str | None = None,
) -> TubitakDinoEmbedder:
    return TubitakDinoEmbedder(device=device)
