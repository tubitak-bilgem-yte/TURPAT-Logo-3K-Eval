"""Triplet retrieval metrics for normalized image embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-6
DEFAULT_TSTC_TEMPERATURE = 1.0


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Embedding contains a zero or non-finite vector.")
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def per_triplet_similarity_ratio(sim_pos: float, sim_neg: float) -> float:
    return float(sim_pos / (sim_neg + EPSILON))


def temperature_scaled_triplet_confidence(
    sim_pos: float,
    sim_neg: float,
    temperature: float = DEFAULT_TSTC_TEMPERATURE,
) -> float:
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("TSTC temperature must be a positive finite value.")

    logit = (sim_pos - sim_neg) / temperature
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))

    exp_logit = float(np.exp(logit))
    return float(exp_logit / (1.0 + exp_logit))


def triplet_silhouette_score(sim_pos: float, sim_neg: float) -> float:
    denominator = 1.0 - min(sim_pos, sim_neg)
    if denominator <= EPSILON:
        return 0.0
    return float((sim_pos - sim_neg) / denominator)


def _require_embedding(embeddings: Mapping[str, Mapping[str, np.ndarray]], image_id: str, kind: str) -> np.ndarray:
    if image_id not in embeddings:
        raise KeyError(f"Missing embedding for image id: {image_id}")
    if kind not in embeddings[image_id]:
        raise ValueError(f"Embedding for image id {image_id} is missing '{kind}' vector.")
    return np.asarray(embeddings[image_id][kind], dtype=np.float32)


def _safe_mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute {name}: no values were provided.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def compute_triplet_metrics(
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    triplets,
    output_csv: str | Path | None = None,
    model_name: str | None = None,
    tstc_temperature: float = DEFAULT_TSTC_TEMPERATURE,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-triplet similarities plus aggregate accuracy, PSR, TSTC, and TSS."""

    results = []

    for triplet in triplets:
        anchor_id = triplet.anchor_id
        positive_id = triplet.positive_id
        negative_id = triplet.negative_id

        row = {
            "row_id": triplet.row_id,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
        }

        for kind in ("cls", "patch"):
            anchor_embedding = _require_embedding(embeddings, anchor_id, kind)
            positive_embedding = _require_embedding(embeddings, positive_id, kind)
            negative_embedding = _require_embedding(embeddings, negative_id, kind)

            sim_pos = cosine_similarity(anchor_embedding, positive_embedding)
            sim_neg = cosine_similarity(anchor_embedding, negative_embedding)
            psr = per_triplet_similarity_ratio(sim_pos, sim_neg)
            tstc = temperature_scaled_triplet_confidence(
                sim_pos=sim_pos,
                sim_neg=sim_neg,
                temperature=tstc_temperature,
            )
            tss = triplet_silhouette_score(sim_pos=sim_pos, sim_neg=sim_neg)

            row[f"{kind}_sim_pos"] = sim_pos
            row[f"{kind}_sim_neg"] = sim_neg
            row[f"{kind}_psr"] = psr
            row[f"{kind}_tstc"] = tstc
            row[f"{kind}_tss"] = tss
            row[f"{kind}_correct"] = int(sim_pos > sim_neg)

        results.append(row)

    if not results:
        raise ValueError("No triplet metrics were computed.")

    df = pd.DataFrame(results)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    summary = {
        "model": model_name or "unknown",
        "num_triplets": int(len(df)),
        "tstc_temperature": float(tstc_temperature),
    }
    for kind in ("cls", "patch"):
        pos_values = df[f"{kind}_sim_pos"].astype(float).to_numpy()
        neg_values = df[f"{kind}_sim_neg"].astype(float).to_numpy()
        psr_values = df[f"{kind}_psr"].astype(float).to_numpy()
        tstc_values = df[f"{kind}_tstc"].astype(float).to_numpy()
        tss_values = df[f"{kind}_tss"].astype(float).to_numpy()
        correct = df[f"{kind}_correct"].astype(float).to_numpy()

        summary.update(
            {
                f"{kind}_accuracy": float(np.mean(correct)),
                f"{kind}_psr_mean": _safe_mean(psr_values.tolist(), f"{kind} PSR"),
                f"{kind}_global_psr": float(np.mean(pos_values) / (np.mean(neg_values) + EPSILON)),
                f"{kind}_tstc_mean": _safe_mean(tstc_values.tolist(), f"{kind} TSTC"),
                f"{kind}_tss_mean": _safe_mean(tss_values.tolist(), f"{kind} TSS"),
                f"{kind}_mean_sim_pos": float(np.mean(pos_values)),
                f"{kind}_mean_sim_neg": float(np.mean(neg_values)),
            }
        )

    return df, summary


def write_summary_files(summaries: Sequence[dict], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(list(summaries))
    summary_df.to_csv(output_path / "summary.csv", index=False)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(list(summaries), handle, indent=2)
