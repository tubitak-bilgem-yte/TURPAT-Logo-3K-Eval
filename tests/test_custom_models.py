from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.base import validate_embedding_batch
from models.custom import parse_custom_model_specs
from models.registry import available_models, create_embedder


class CustomModelTest(unittest.TestCase):
    def test_custom_model_factory_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "my_model.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    import numpy as np

                    class Embedder:
                        def embed_batch(self, images):
                            batch = len(images)
                            cls = np.ones((batch, 3), dtype=np.float32)
                            patch = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (batch, 1))
                            return {"cls": cls, "patch": patch}

                    def create_embedder(device="cpu", cache_dir=None, model_key=None):
                        return Embedder()
                    """
                ),
                encoding="utf-8",
            )

            custom_models = parse_custom_model_specs([f"local-test={module_path}:create_embedder"])
            self.assertIn("local-test", available_models(custom_models))

            embedder = create_embedder("local-test", device="cpu", custom_models=custom_models)
            images = [Image.new("RGB", (8, 8), color="white"), Image.new("RGB", (8, 8), color="black")]
            batch = validate_embedding_batch(embedder.embed_batch(images), expected_count=2, model_name="local-test")

            self.assertEqual(batch["cls"].shape, (2, 3))
            self.assertTrue(np.allclose(np.linalg.norm(batch["patch"], axis=1), 1.0))

    def test_invalid_custom_spec_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "KEY=MODULE_OR_FILE"):
            parse_custom_model_specs(["missing_separator"])


if __name__ == "__main__":
    unittest.main()
