"""Available embedding model registry."""

from __future__ import annotations

from .base import ImageEmbedder
from .custom import CustomModelSpec, create_custom_embedder
from .vision import TransformersVisionEmbedder, VisionModelSpec


def square_image_kwargs(resolution: int) -> dict[str, object]:
    size = {"height": resolution, "width": resolution}
    return {"size": size, "crop_size": size}


MODEL_REGISTRY = {
    "facebook/dinov2-large": VisionModelSpec(
        key="facebook/dinov2-large",
        model_id="facebook/dinov2-large",
        model_class="auto",
        preprocess_kwargs=square_image_kwargs(512),
        cls_strategy="token",
        cls_token_index=0,
        patch_start_index=1,
    ),
    "facebook/dinov2-giant": VisionModelSpec(
        key="facebook/dinov2-giant",
        model_id="facebook/dinov2-giant",
        model_class="auto",
        preprocess_kwargs=square_image_kwargs(512),
        cls_strategy="token",
        cls_token_index=0,
        patch_start_index=1,
    ),
    "facebook/ijepa_vith16_1k": VisionModelSpec(
        key="facebook/ijepa_vith16_1k",
        model_id="facebook/ijepa_vith16_1k",
        model_class="auto",
        preprocess_kwargs=square_image_kwargs(512),
        forward_kwargs={"interpolate_pos_encoding": True},
        cls_strategy="mean_tokens",
        patch_start_index=0,
    ),
    "facebook/ijepa_vitg16_22k": VisionModelSpec(
        key="facebook/ijepa_vitg16_22k",
        model_id="facebook/ijepa_vitg16_22k",
        model_class="auto",
        preprocess_kwargs=square_image_kwargs(512),
        forward_kwargs={"interpolate_pos_encoding": True},
        cls_strategy="mean_tokens",
        patch_start_index=0,
    ),
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": VisionModelSpec(
        key="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        model_class="auto",
        preprocess_kwargs=square_image_kwargs(512),
        cls_strategy="token",
        cls_token_index=0,
        patch_start_index="after_register_tokens",
    ),
    "google/siglip2-so400m-patch16-512": VisionModelSpec(
        key="google/siglip2-so400m-patch16-512",
        model_id="google/siglip2-so400m-patch16-512",
        model_class="vision_submodel",
        preprocess_kwargs=square_image_kwargs(512),
        cls_strategy="pooler",
        patch_start_index=0,
    ),
    "google/siglip2-large-patch16-512": VisionModelSpec(
        key="google/siglip2-large-patch16-512",
        model_id="google/siglip2-large-patch16-512",
        model_class="vision_submodel",
        preprocess_kwargs=square_image_kwargs(512),
        cls_strategy="pooler",
        patch_start_index=0,
    ),
    "google/siglip2-giant-opt-patch16-384": VisionModelSpec(
        key="google/siglip2-giant-opt-patch16-384",
        model_id="google/siglip2-giant-opt-patch16-384",
        model_class="vision_submodel",
        preprocess_kwargs=square_image_kwargs(384),
        cls_strategy="pooler",
        patch_start_index=0,
    ),
}

DEFAULT_MODELS = [
    "facebook/dinov2-large",
    "facebook/ijepa_vith16_1k",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "google/siglip2-so400m-patch16-512",
    "google/siglip2-large-patch16-512",
]

BUILTIN_CUSTOM_MODELS = {
    "tubitak_dino_model": CustomModelSpec(
        key="tubitak_dino_model",
        factory_ref="models.custom_models.tubitak_dino_model:create_embedder",
    ),
}


def available_models(custom_models: dict[str, CustomModelSpec] | None = None) -> list[str]:
    keys = set(MODEL_REGISTRY)
    keys.update(BUILTIN_CUSTOM_MODELS)
    keys.update(custom_models or {})
    return sorted(keys)


def create_embedder(
    model_key: str,
    device: str = "cuda",
    cache_dir: str | None = None,
    custom_models: dict[str, CustomModelSpec] | None = None,
) -> ImageEmbedder:
    merged_custom_models = {**BUILTIN_CUSTOM_MODELS, **(custom_models or {})}
    if model_key in merged_custom_models:
        return create_custom_embedder(merged_custom_models[model_key], device=device, cache_dir=cache_dir)
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}. Available models: {available_models(merged_custom_models)}")
    return TransformersVisionEmbedder(MODEL_REGISTRY[model_key], device=device, cache_dir=cache_dir)
