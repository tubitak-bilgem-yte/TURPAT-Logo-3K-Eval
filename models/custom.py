"""Runtime loading for user-defined benchmark models."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from .base import ImageEmbedder


@dataclass(frozen=True)
class CustomModelSpec:
    key: str
    factory_ref: str


def parse_custom_model_specs(raw_specs: list[str] | None) -> dict[str, CustomModelSpec]:
    specs: dict[str, CustomModelSpec] = {}
    for raw_spec in raw_specs or []:
        if "=" not in raw_spec:
            raise ValueError(
                "Custom model specs must use KEY=MODULE_OR_FILE:FACTORY, "
                f"got {raw_spec!r}."
            )
        key, factory_ref = raw_spec.split("=", 1)
        key = key.strip()
        factory_ref = factory_ref.strip()
        if not key:
            raise ValueError(f"Custom model key is empty in {raw_spec!r}.")
        if ":" not in factory_ref:
            raise ValueError(
                "Custom model factory must use MODULE_OR_FILE:FACTORY, "
                f"got {factory_ref!r}."
            )
        if key in specs:
            raise ValueError(f"Duplicate custom model key: {key}")
        specs[key] = CustomModelSpec(key=key, factory_ref=factory_ref)
    return specs


def _load_module(module_ref: str) -> ModuleType:
    module_path = Path(module_ref).expanduser()
    if module_ref.endswith(".py") or module_path.exists():
        resolved_path = module_path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Custom model module file does not exist: {resolved_path}")
        digest = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        module_name = f"_turpat_custom_model_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, resolved_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import custom model module from {resolved_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def _load_factory(factory_ref: str) -> Callable[..., ImageEmbedder]:
    module_ref, factory_name = factory_ref.rsplit(":", 1)
    module = _load_module(module_ref)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise AttributeError(f"Custom model factory {factory_name!r} was not found in {module_ref!r}.")
    if not callable(factory):
        raise TypeError(f"Custom model factory {factory_ref!r} is not callable.")
    return factory


def _call_factory(factory: Callable[..., ImageEmbedder], **candidate_kwargs: object) -> ImageEmbedder:
    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**candidate_kwargs)

    kwargs = {
        name: value
        for name, value in candidate_kwargs.items()
        if name in signature.parameters
    }
    return factory(**kwargs)


def create_custom_embedder(
    spec: CustomModelSpec,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> ImageEmbedder:
    factory = _load_factory(spec.factory_ref)
    embedder = _call_factory(
        factory,
        device=device,
        cache_dir=cache_dir,
        model_key=spec.key,
    )
    if not hasattr(embedder, "embed_batch") or not callable(embedder.embed_batch):
        raise TypeError(
            f"Custom model {spec.key!r} must return an object with an embed_batch(images) method."
        )
    return embedder
