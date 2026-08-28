"""Dataset loading helpers for trademark triplet benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, Optional

from datasets import load_dataset
from PIL import Image


IMAGE_COLUMNS = ("anchor_img", "positive_img", "negative_img")


@dataclass(frozen=True)
class TripletRecord:
    """A single benchmark triplet expressed as stable image ids."""

    row_id: int
    anchor_id: str
    positive_id: str
    negative_id: str


@dataclass
class BenchmarkData:
    """Loaded triplets and deduplicated images."""

    triplets: list[TripletRecord]
    images: Dict[str, Image.Image]


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return a detached RGB PIL image."""

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image)!r}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.copy()


def stable_image_id(image: Image.Image, row_id: int, role: str) -> str:
    """Create a stable id from image pixels, with a row-role fallback."""

    try:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image)!r}")
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        digest = sha256()
        digest.update(rgb.mode.encode("utf-8"))
        digest.update(str(rgb.size).encode("utf-8"))
        digest.update(rgb.tobytes())
        return f"sha256-{digest.hexdigest()}"
    except Exception:
        return f"row-{row_id:06d}-{role}"


def iter_triplet_rows(dataset: Iterable[dict], limit: Optional[int] = None) -> Iterable[tuple[int, dict]]:
    """Yield dataset rows with an optional smoke-test limit."""

    for row_id, row in enumerate(dataset):
        if limit is not None and row_id >= limit:
            break
        yield row_id, row


def load_triplet_dataset(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
    token: bool | str | None = True,
) -> BenchmarkData:
    """Load triplets from Hugging Face and deduplicate images by content hash."""

    split_expr = f"{split}[:{limit}]" if limit is not None else split
    dataset = load_dataset(
        dataset_name,
        split=split_expr,
        cache_dir=cache_dir,
        token=token,
        streaming=False,
    )

    triplets: list[TripletRecord] = []
    images: Dict[str, Image.Image] = {}

    for row_id, row in iter_triplet_rows(dataset, limit=None):
        missing_columns = [column for column in IMAGE_COLUMNS if column not in row]
        if missing_columns:
            raise KeyError(f"Dataset row is missing expected image columns: {missing_columns}")

        row_ids: dict[str, str] = {}
        for role, column in zip(("anchor", "positive", "negative"), IMAGE_COLUMNS):
            image = ensure_rgb(row[column])
            image_id = stable_image_id(image, row_id=row_id, role=role)
            row_ids[role] = image_id
            if image_id not in images:
                images[image_id] = image

        triplets.append(
            TripletRecord(
                row_id=row_id,
                anchor_id=row_ids["anchor"],
                positive_id=row_ids["positive"],
                negative_id=row_ids["negative"],
            )
        )

    if not triplets:
        raise ValueError("No triplets were loaded. Check the dataset split or --limit value.")

    return BenchmarkData(triplets=triplets, images=images)
