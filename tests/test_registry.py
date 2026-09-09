from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.custom_models.tubitak_dino_model import TubitakDinoEmbedder
from models.registry import BUILTIN_CUSTOM_MODELS, DEFAULT_MODELS, MODEL_REGISTRY, available_models
from models.vision import TransformersVisionEmbedder


REQUESTED_MODELS = [
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
    "facebook/ijepa_vith16_1k",
    "facebook/ijepa_vitg16_22k",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "google/siglip2-so400m-patch16-512",
    "google/siglip2-large-patch16-512",
    "google/siglip2-giant-opt-patch16-384",
]

DEFAULT_MODEL_KEYS = [
    "facebook/dinov2-large",
    "facebook/ijepa_vith16_1k",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "google/siglip2-so400m-patch16-512",
    "google/siglip2-large-patch16-512",
]


class RegistryTest(unittest.TestCase):
    def test_registry_contains_only_requested_models(self) -> None:
        self.assertEqual(list(MODEL_REGISTRY), REQUESTED_MODELS)
        self.assertEqual(DEFAULT_MODELS, DEFAULT_MODEL_KEYS)
        for spec in MODEL_REGISTRY.values():
            resolution = 384 if spec.key == "google/siglip2-giant-opt-patch16-384" else 512
            self.assertEqual(spec.preprocess_kwargs["size"], {"height": resolution, "width": resolution})
            self.assertEqual(spec.preprocess_kwargs["crop_size"], {"height": resolution, "width": resolution})

    def test_builtin_custom_models_include_tubitak_dino(self) -> None:
        self.assertIn("tubitak_dino_model", BUILTIN_CUSTOM_MODELS)
        self.assertIn("tubitak_dino_model", available_models())
        self.assertEqual(
            BUILTIN_CUSTOM_MODELS["tubitak_dino_model"].factory_ref,
            "models.custom_models.tubitak_dino_model:create_embedder",
        )

    def test_dinov3_patch_pooling_skips_register_tokens(self) -> None:
        spec = MODEL_REGISTRY["facebook/dinov3-vith16plus-pretrain-lvd1689m"]
        embedder = object.__new__(TransformersVisionEmbedder)
        embedder.spec = spec
        embedder.model = SimpleNamespace(config=SimpleNamespace(num_register_tokens=4))

        hidden = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
        pooled = embedder._pool_patch_tokens(hidden)

        self.assertTrue(torch.equal(pooled, hidden[:, 5:, :].mean(dim=1)))

    def test_siglip_uses_pooler_for_cls_and_all_tokens_for_patch(self) -> None:
        spec = MODEL_REGISTRY["google/siglip2-large-patch16-512"]
        embedder = object.__new__(TransformersVisionEmbedder)
        embedder.spec = spec
        embedder.model = SimpleNamespace(config=SimpleNamespace())

        hidden = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
        patch = embedder._pool_patch_tokens(hidden)
        cls = embedder._pool_cls_embedding(SimpleNamespace(pooler_output=torch.ones(1, 2)), hidden, patch)

        self.assertTrue(torch.equal(patch, hidden.mean(dim=1)))
        self.assertTrue(torch.equal(cls, torch.ones(1, 2)))

    def test_tubitak_dino_patch_pooling_skips_register_tokens(self) -> None:
        embedder = object.__new__(TubitakDinoEmbedder)
        embedder.num_registers = 4

        hidden = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
        cls, patch = embedder._pool_tokens(hidden)

        self.assertTrue(torch.equal(cls, hidden[:, 0, :]))
        self.assertTrue(torch.equal(patch, hidden[:, 5:, :].mean(dim=1)))


if __name__ == "__main__":
    unittest.main()
