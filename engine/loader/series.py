"""Ordered pretraining checkpoint and post-training stage series management."""

from __future__ import annotations

import re
import threading
import uuid
from typing import Any

from engine.contracts import Series, SeriesItem, SeriesKind
from engine.loader.load import LoadedModel, load_model
from engine.loader.registry import unload_model

_SERIES_LOCK = threading.RLock()
_SERIES_REGISTRY: dict[str, Series] = {}
_ACTIVE_SERIES_MODEL: dict[str, str] = {}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned or "model"


def register_series(
    kind: SeriesKind | str,
    items: list[SeriesItem] | list[dict[str, Any]],
    model_family: str,
    series_id: str | None = None,
) -> Series:
    """Validate, sort, and register an ordered checkpoint or post-training series."""
    if kind not in ("pretraining", "post_training"):
        raise ValueError(
            f"Invalid series kind {kind!r}; expected 'pretraining' or 'post_training'."
        )
    if not items:
        raise ValueError("Series must contain at least one checkpoint or stage item.")

    parsed_items = [
        item if isinstance(item, SeriesItem) else SeriesItem.model_validate(item)
        for item in items
    ]
    sorted_items = sorted(parsed_items, key=lambda it: it.order)

    seen_ids: set[str] = set()
    for item in sorted_items:
        if item.checkpoint_id in seen_ids:
            raise ValueError(f"Duplicate checkpoint_id in series: {item.checkpoint_id!r}")
        seen_ids.add(item.checkpoint_id)

    assigned_id = series_id or f"series_{_slugify(model_family)}_{uuid.uuid4().hex[:8]}"
    series = Series(
        series_id=assigned_id,
        kind=kind,  # type: ignore[arg-type]
        model_family=model_family,
        items=sorted_items,
    )
    with _SERIES_LOCK:
        _SERIES_REGISTRY[series.series_id] = series
    return series


def get_series(series_id: str) -> Series | None:
    """Return a previously registered series by ID."""
    with _SERIES_LOCK:
        return _SERIES_REGISTRY.get(series_id)


def list_series() -> list[Series]:
    """Return all registered checkpoint/stage series."""
    with _SERIES_LOCK:
        return list(_SERIES_REGISTRY.values())


def load_checkpoint(
    series: Series,
    checkpoint_id: str,
    allow_remote_code: bool = False,
    allow_unsafe_weights: bool = False,
    is_placeholder: bool | None = None,
) -> LoadedModel:
    """Load one checkpoint from a series after unloading any previously active checkpoint."""
    matching = [item for item in series.items if item.checkpoint_id == checkpoint_id]
    if not matching:
        raise KeyError(
            f"Checkpoint {checkpoint_id!r} not found in series {series.series_id!r}."
        )
    target = matching[0]

    with _SERIES_LOCK:
        prev_model_id = _ACTIVE_SERIES_MODEL.pop(series.series_id, None)
    if prev_model_id is not None:
        unload_model(prev_model_id)

    source_spec = target.source
    revision: str | None = None
    adapter: str | None = None

    if source_spec.startswith("revision:"):
        source = series.model_family
        revision = source_spec.split("revision:", 1)[1]
    elif source_spec.startswith("adapter:"):
        source = series.model_family
        adapter = source_spec.split("adapter:", 1)[1]
    else:
        source = source_spec

    loaded = load_model(
        source=source,
        revision=revision,
        allow_remote_code=allow_remote_code,
        adapter=adapter,
        allow_unsafe_weights=allow_unsafe_weights,
        is_placeholder=is_placeholder,
    )
    if loaded.revision is None:
        loaded.revision = checkpoint_id
    with _SERIES_LOCK:
        _ACTIVE_SERIES_MODEL[series.series_id] = loaded.model_id
    return loaded
