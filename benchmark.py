"""CLI entrypoint for the trademark triplet retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from dataloader import load_triplet_dataset
from metrics import compute_triplet_metrics, write_summary_files
from models.base import validate_embedding_batch
from models.custom import parse_custom_model_specs
from models.registry import DEFAULT_MODELS, available_models, create_embedder


EmbeddingDict = Dict[str, dict[str, np.ndarray]]
MEAN_PADDING_COLOR = (124, 116, 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark image retrieval models on trademark triplets.")
    parser.add_argument("--dataset", default="BILGEM-AI/TURPAT-Logo-3K", help="Hugging Face dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model registry keys to benchmark.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, cuda:1, or cpu.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of triplets for a smoke run.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore existing embedding caches.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for caches and result files.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face dataset/model cache directory.")
    parser.add_argument(
        "--preprocess-inference",
        action="store_true",
        help="Crop foreground content and pad to a mean-color square before embedding images.",
    )
    parser.add_argument(
        "--tstc-temperature",
        type=float,
        default=1.0,
        help="Positive temperature tau used for Temperature-Scaled Triplet Confidence.",
    )
    parser.add_argument("--list-models", action="store_true", help="List registered model keys and exit.")
    parser.add_argument(
        "--custom-model",
        action="append",
        default=[],
        metavar="KEY=MODULE_OR_FILE:FACTORY",
        help=(
            "Register a custom model factory. Example: "
            "--custom-model my-model=/abs/path/my_model.py:create_embedder --models my-model"
        ),
    )
    return parser.parse_args()


def safe_model_name(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_key).strip("_") or "model"


def run_name_for(model_key: str, preprocess_inference: bool = False) -> str:
    suffix = "_preprocessed" if preprocess_inference else ""
    return f"{safe_model_name(model_key)}{suffix}"


def cache_path_for(output_dir: Path, model_key: str, preprocess_inference: bool = False) -> Path:
    return output_dir / "embeddings" / f"{run_name_for(model_key, preprocess_inference)}.npz"


def result_path_for(output_dir: Path, model_key: str, preprocess_inference: bool = False) -> Path:
    return output_dir / "results" / f"{run_name_for(model_key, preprocess_inference)}.csv"


def find_foreground_bbox(image: Image.Image, padding: int = 10) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    gray_image = ImageOps.grayscale(rgb).filter(ImageFilter.GaussianBlur(radius=6.5))
    gray = np.asarray(gray_image, dtype=np.float32)

    local_mean = np.asarray(gray_image.filter(ImageFilter.BoxBlur(radius=5)), dtype=np.float32)
    adaptive_mask = gray < (local_mean - 2.0)

    rgb_array = np.asarray(rgb, dtype=np.float32)
    border_pixels = np.concatenate(
        [
            rgb_array[0, :, :],
            rgb_array[-1, :, :],
            rgb_array[:, 0, :],
            rgb_array[:, -1, :],
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    border_distances = np.linalg.norm(border_pixels - background_color, axis=1)
    distance_threshold = max(20.0, float(np.percentile(border_distances, 95) + 8.0))
    color_mask = np.linalg.norm(rgb_array - background_color, axis=2) > distance_threshold

    mask = adaptive_mask | color_mask
    if not np.any(mask):
        return None

    height, width = mask.shape
    row_threshold = max(1, int(width * 0.001))
    col_threshold = max(1, int(height * 0.001))
    rows = np.where(mask.sum(axis=1) >= row_threshold)[0]
    cols = np.where(mask.sum(axis=0) >= col_threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    left = max(0, int(cols[0]) - padding)
    upper = max(0, int(rows[0]) - padding)
    right = min(width, int(cols[-1]) + padding + 1)
    lower = min(height, int(rows[-1]) + padding + 1)
    if right <= left or lower <= upper:
        return None
    return left, upper, right, lower


def pad_to_square(image: Image.Image, fill_color: tuple[int, int, int] = MEAN_PADDING_COLOR) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width == height:
        return rgb.copy()

    side = max(width, height)
    result = Image.new("RGB", (side, side), fill_color)
    result.paste(rgb, ((side - width) // 2, (side - height) // 2))
    return result


def preprocess_image_for_inference(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    bbox = find_foreground_bbox(rgb)
    if bbox is not None:
        rgb = rgb.crop(bbox)
    return pad_to_square(rgb)


def load_embedding_cache(path: Path) -> EmbeddingDict:
    if not path.exists():
        return {}

    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str)
    cls_embeddings = data["cls"]
    patch_embeddings = data["patch"]

    if len(ids) != len(cls_embeddings) or len(ids) != len(patch_embeddings):
        raise ValueError(f"Invalid embedding cache shape in {path}")

    return {
        image_id: {
            "cls": np.asarray(cls_embeddings[index], dtype=np.float32),
            "patch": np.asarray(patch_embeddings[index], dtype=np.float32),
        }
        for index, image_id in enumerate(ids)
    }


def save_embedding_cache(path: Path, embeddings: EmbeddingDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(embeddings)
    cls_embeddings = np.stack([embeddings[image_id]["cls"] for image_id in ids]).astype(np.float32)
    patch_embeddings = np.stack([embeddings[image_id]["patch"] for image_id in ids]).astype(np.float32)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        cls=cls_embeddings,
        patch=patch_embeddings,
    )


def embed_missing_images(
    embedder,
    images: dict[str, Image.Image],
    existing_embeddings: EmbeddingDict,
    batch_size: int,
    model_key: str,
    preprocess_inference: bool,
) -> EmbeddingDict:
    image_ids = list(images.keys())
    missing_ids = [image_id for image_id in image_ids if image_id not in existing_embeddings]

    if not missing_ids:
        return existing_embeddings

    print(f"Embedding {len(missing_ids)} missing images...", flush=True)
    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start : start + batch_size]
        batch_images = [
            preprocess_image_for_inference(images[image_id]) if preprocess_inference else images[image_id]
            for image_id in batch_ids
        ]
        batch_embeddings = validate_embedding_batch(
            embedder.embed_batch(batch_images),
            expected_count=len(batch_images),
            model_name=model_key,
        )

        for image_id, cls_embedding, patch_embedding in zip(
            batch_ids,
            batch_embeddings["cls"],
            batch_embeddings["patch"],
        ):
            existing_embeddings[image_id] = {
                "cls": np.asarray(cls_embedding, dtype=np.float32),
                "patch": np.asarray(patch_embedding, dtype=np.float32),
            }

        done = min(start + batch_size, len(missing_ids))
        print(f"  embedded {done}/{len(missing_ids)}", flush=True)

    return existing_embeddings


def run_model(
    model_key: str,
    data,
    args: argparse.Namespace,
    output_dir: Path,
    custom_models,
) -> dict:
    cache_path = cache_path_for(output_dir, model_key, args.preprocess_inference)
    result_path = result_path_for(output_dir, model_key, args.preprocess_inference)

    print(f"\n== {model_key} ==", flush=True)
    embeddings = {} if args.force_recompute else load_embedding_cache(cache_path)
    if embeddings:
        print(f"Loaded {len(embeddings)} cached embeddings from {cache_path}", flush=True)

    required_ids = set(data.images)
    if required_ids - set(embeddings):
        embedder = create_embedder(
            model_key,
            device=args.device,
            cache_dir=args.cache_dir,
            custom_models=custom_models,
        )
        embeddings = embed_missing_images(
            embedder,
            data.images,
            embeddings,
            args.batch_size,
            model_key,
            preprocess_inference=args.preprocess_inference,
        )
        save_embedding_cache(cache_path, embeddings)
        print(f"Saved embedding cache to {cache_path}", flush=True)
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Embedding cache covers all images.", flush=True)

    _, summary = compute_triplet_metrics(
        embeddings=embeddings,
        triplets=data.triplets,
        output_csv=result_path,
        model_name=f"{model_key} [preprocessed]" if args.preprocess_inference else model_key,
        tstc_temperature=args.tstc_temperature,
    )
    print(f"Wrote per-triplet results to {result_path}", flush=True)
    return summary


def print_leaderboard(summaries: list[dict]) -> None:
    print("\nLeaderboard", flush=True)
    print("model                     cls_acc  patch_acc  cls_psr  patch_psr  cls_tstc  patch_tstc  cls_tss  patch_tss", flush=True)
    print("-" * 108, flush=True)
    for summary in sorted(summaries, key=lambda item: item["patch_accuracy"], reverse=True):
        print(
            f"{summary['model']:<25} "
            f"{summary['cls_accuracy']:.4f}   "
            f"{summary['patch_accuracy']:.4f}     "
            f"{summary['cls_psr_mean']:.4f}   "
            f"{summary['patch_psr_mean']:.4f}     "
            f"{summary['cls_tstc_mean']:.4f}     "
            f"{summary['patch_tstc_mean']:.4f}     "
            f"{summary['cls_tss_mean']:.4f}   "
            f"{summary['patch_tss_mean']:.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    custom_models = parse_custom_model_specs(args.custom_model)
    if args.list_models:
        for model_key in available_models(custom_models):
            print(model_key)
        return

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")
    if args.tstc_temperature <= 0.0 or not np.isfinite(args.tstc_temperature):
        raise ValueError("--tstc-temperature must be a positive finite value.")

    unknown_models = [model_key for model_key in args.models if model_key not in available_models(custom_models)]
    if unknown_models:
        raise ValueError(f"Unknown model(s): {unknown_models}. Available: {available_models(custom_models)}")

    output_dir = Path(args.output_dir)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "models": args.models,
                "batch_size": args.batch_size,
                "device": args.device,
                "limit": args.limit,
                "output_dir": str(output_dir),
                "custom_models": sorted(custom_models),
                "tstc_temperature": args.tstc_temperature,
                "preprocess_inference": args.preprocess_inference,
            },
            indent=2,
        ),
        flush=True,
    )

    data = load_triplet_dataset(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        cache_dir=args.cache_dir,
        token=True,
    )
    print(f"Loaded {len(data.triplets)} triplets with {len(data.images)} unique images.", flush=True)

    summaries = [run_model(model_key, data, args, output_dir, custom_models) for model_key in args.models]
    write_summary_files(summaries, output_dir / "results")
    print(f"\nWrote summary files to {output_dir / 'results'}", flush=True)
    print_leaderboard(summaries)


if __name__ == "__main__":
    main()
