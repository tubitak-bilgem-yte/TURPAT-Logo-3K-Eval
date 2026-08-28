"""Common model interface and embedding validation."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np
from PIL import Image


EmbeddingBatch = dict[str, np.ndarray]


class ImageEmbedder(Protocol):
    """Protocol every benchmark model must implement."""

    def embed_batch(self, images: Sequence[Image.Image]) -> Mapping[str, object]:
        """Return batched `cls` and `patch` embeddings for RGB PIL images."""


def _to_numpy(value: object, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} embeddings must have shape [batch, dim], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} embeddings contain NaN or infinite values.")
    return array


def normalize_rows(array: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} embeddings contain at least one zero vector.")
    return (array / norms).astype(np.float32)


def validate_embedding_batch(batch: Mapping[str, object], expected_count: int, model_name: str) -> EmbeddingBatch:
    """Validate and normalize an embedder's output contract."""

    missing = [name for name in ("cls", "patch") if name not in batch]
    if missing:
        raise ValueError(f"Model {model_name!r} did not return required embedding key(s): {missing}")

    cls_embeddings = normalize_rows(_to_numpy(batch["cls"], "cls"), "cls")
    patch_embeddings = normalize_rows(_to_numpy(batch["patch"], "patch"), "patch")

    for name, embeddings in (("cls", cls_embeddings), ("patch", patch_embeddings)):
        if embeddings.shape[0] != expected_count:
            raise ValueError(
                f"Model {model_name!r} returned {embeddings.shape[0]} {name} embeddings "
                f"for a batch of {expected_count} images."
            )

    return {"cls": cls_embeddings, "patch": patch_embeddings}
