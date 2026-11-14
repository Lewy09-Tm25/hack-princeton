"""In-memory LRU registry keeping at most two loaded models at a time."""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from engine.loader.load import LoadedModel

MAX_LOADED_MODELS = 2

_LOCK = threading.RLock()
_MODELS: OrderedDict[str, LoadedModel] = OrderedDict()


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def register_loaded_model(model: LoadedModel) -> LoadedModel:
    """Store a loaded model as most-recently-used, evicting oldest beyond limit 2."""
    evicted = False
    with _LOCK:
        if model.model_id in _MODELS:
            _MODELS.move_to_end(model.model_id)
        _MODELS[model.model_id] = model
        while len(_MODELS) > MAX_LOADED_MODELS:
            _MODELS.popitem(last=False)
            evicted = True
    if evicted:
        _release_memory()
    return model


def get_loaded_model(model_id: str) -> LoadedModel | None:
    """Retrieve a loaded model by ID and refresh its LRU position."""
    with _LOCK:
        model = _MODELS.get(model_id)
        if model is not None:
            _MODELS.move_to_end(model_id)
        return model


def unload_model(model_id: str) -> None:
    """Unload a model from the in-memory registry and release accelerator caches."""
    removed = False
    with _LOCK:
        if model_id in _MODELS:
            del _MODELS[model_id]
            removed = True
    if removed:
        _release_memory()


def list_loaded_models() -> list[LoadedModel]:
    """Return currently loaded models in least-to-most-recently-used order."""
    with _LOCK:
        return list(_MODELS.values())


def clear_registry() -> None:
    """Unload all models currently held in memory."""
    with _LOCK:
        had_models = bool(_MODELS)
        _MODELS.clear()
    if had_models:
        _release_memory()
