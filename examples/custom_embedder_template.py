"""Template for benchmarking a custom local model.

Run with:

python benchmark.py \
  --custom-model my-trained=examples/custom_embedder_template.py:create_embedder \
  --models my-trained \
  --limit 8

Replace the toy projection with your own checkpoint loading, preprocessing, and
forward pass. The benchmark passes RGB PIL images into embed_batch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class MyCustomEmbedder:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        features = []
        for image in images:
            image = image.convert("RGB").resize((224, 224))
            array = np.asarray(image, dtype=np.float32) / 255.0
            channel_means = array.mean(axis=(0, 1))
            channel_stds = array.std(axis=(0, 1))
            features.append(np.concatenate([channel_means, channel_stds]))

        tensor = torch.tensor(np.stack(features), dtype=torch.float32, device=self.device)
        tensor = F.normalize(tensor, p=2, dim=-1)

        # If your model only produces one embedding, return it for both keys.
        embeddings = tensor.detach().cpu().numpy().astype(np.float32)
        return {"cls": embeddings, "patch": embeddings}


def create_embedder(device: str = "cuda", cache_dir: str | None = None, model_key: str | None = None) -> MyCustomEmbedder:
    return MyCustomEmbedder(device=device)
