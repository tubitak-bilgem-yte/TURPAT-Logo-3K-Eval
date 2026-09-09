from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader import TripletRecord
from metrics import (
    compute_triplet_metrics,
    cosine_similarity,
    temperature_scaled_triplet_confidence,
    triplet_silhouette_score,
)


class MetricsTest(unittest.TestCase):
    def test_cosine_similarity_normalizes_inputs(self) -> None:
        self.assertAlmostEqual(cosine_similarity(np.array([2.0, 0.0]), np.array([10.0, 0.0])), 1.0)
        self.assertAlmostEqual(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 3.0])), 0.0)

    def test_tstc_uses_temperature_scaled_sigmoid(self) -> None:
        self.assertAlmostEqual(temperature_scaled_triplet_confidence(0.8, 0.3, temperature=0.5), 0.7310585786)
        self.assertAlmostEqual(temperature_scaled_triplet_confidence(0.2, 0.2, temperature=0.5), 0.5)

    def test_tss_uses_margin_over_max_triplet_distance(self) -> None:
        self.assertAlmostEqual(triplet_silhouette_score(0.8, 0.3), 0.5 / 0.7)
        self.assertAlmostEqual(triplet_silhouette_score(0.3, 0.8), -0.5 / 0.7)
        self.assertAlmostEqual(triplet_silhouette_score(1.0, 1.0), 0.0)

    def test_triplet_accuracy_psr_tstc_and_tss(self) -> None:
        embeddings = {
            "anchor": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "positive": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "negative": {"cls": np.array([0.0, 1.0]), "patch": np.array([0.5, 0.5])},
        }
        triplets = [TripletRecord(row_id=0, anchor_id="anchor", positive_id="positive", negative_id="negative")]

        df, summary = compute_triplet_metrics(embeddings, triplets, model_name="synthetic")

        self.assertEqual(int(df.loc[0, "cls_correct"]), 1)
        self.assertEqual(int(df.loc[0, "patch_correct"]), 1)
        self.assertAlmostEqual(summary["cls_accuracy"], 1.0)
        self.assertAlmostEqual(summary["patch_accuracy"], 1.0)
        self.assertGreater(summary["patch_psr_mean"], 1.0)
        self.assertGreater(summary["patch_global_psr"], 1.0)
        self.assertGreater(summary["cls_tstc_mean"], 0.5)
        self.assertGreater(summary["patch_tstc_mean"], 0.5)
        self.assertGreater(summary["cls_tss_mean"], 0.0)
        self.assertGreater(summary["patch_tss_mean"], 0.0)

    def test_writes_csv(self) -> None:
        embeddings = {
            "a": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "p": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "n": {"cls": np.array([0.0, 1.0]), "patch": np.array([0.0, 1.0])},
        }
        triplets = [TripletRecord(row_id=7, anchor_id="a", positive_id="p", negative_id="n")]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv = Path(temp_dir) / "triplets.csv"
            compute_triplet_metrics(embeddings, triplets, output_csv=output_csv)
            self.assertTrue(output_csv.exists())
            output_text = output_csv.read_text(encoding="utf-8")
            self.assertIn("cls_sim_pos", output_text)
            self.assertIn("cls_psr", output_text)
            self.assertIn("cls_tstc", output_text)
            self.assertIn("cls_tss", output_text)

    def test_missing_embedding_raises_clear_error(self) -> None:
        embeddings = {
            "a": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "p": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
        }
        triplets = [TripletRecord(row_id=0, anchor_id="a", positive_id="p", negative_id="missing")]
        with self.assertRaisesRegex(KeyError, "Missing embedding"):
            compute_triplet_metrics(embeddings, triplets)

    def test_invalid_embedding_raises_clear_error(self) -> None:
        embeddings = {
            "a": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "p": {"cls": np.array([1.0, 0.0]), "patch": np.array([1.0, 0.0])},
            "n": {"cls": np.array([0.0, 1.0])},
        }
        triplets = [TripletRecord(row_id=0, anchor_id="a", positive_id="p", negative_id="n")]
        with self.assertRaisesRegex(ValueError, "missing 'patch'"):
            compute_triplet_metrics(embeddings, triplets)


if __name__ == "__main__":
    unittest.main()
