"""Model and checkpoint series loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.loader.load import LoadedModel, load_custom_module, load_model
    from engine.loader.registry import (
        MAX_LOADED_MODELS,
        clear_registry,
        get_loaded_model,
        list_loaded_models,
        register_loaded_model,
        unload_model,
    )
    from engine.loader.series import get_series, list_series, load_checkpoint, register_series

__all__ = [
    "LoadedModel",
    "MAX_LOADED_MODELS",
    "clear_registry",
    "get_loaded_model",
    "get_series",
    "list_loaded_models",
    "list_series",
    "load_checkpoint",
    "load_custom_module",
    "load_model",
    "register_loaded_model",
    "register_series",
    "unload_model",
]


def __getattr__(name: str) -> Any:
    if name in {"LoadedModel", "load_custom_module", "load_model"}:
        from engine.loader import load

        return getattr(load, name)
    if name in {
        "MAX_LOADED_MODELS",
        "clear_registry",
        "get_loaded_model",
        "list_loaded_models",
        "register_loaded_model",
        "unload_model",
    }:
        from engine.loader import registry

        return getattr(registry, name)
    if name in {"get_series", "list_series", "load_checkpoint", "register_series"}:
        from engine.loader import series

        return getattr(series, name)
    raise AttributeError(f"module 'engine.loader' has no attribute {name!r}")
