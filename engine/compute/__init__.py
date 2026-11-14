"""Device detection, memory budgeting, and runtime estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.compute.budget import (
        MAX_BATCH_SIZE,
        MEMORY_HEADROOM_FACTOR,
        MIN_BATCH_SIZE,
        LoadPlan,
        batch_size_for,
        estimate_seconds,
        plan_for_model,
    )
    from engine.compute.device import ComputeProfile, detect, precision_name

__all__ = [
    "ComputeProfile",
    "LoadPlan",
    "MAX_BATCH_SIZE",
    "MEMORY_HEADROOM_FACTOR",
    "MIN_BATCH_SIZE",
    "batch_size_for",
    "detect",
    "estimate_seconds",
    "plan_for_model",
    "precision_name",
]


def __getattr__(name: str) -> Any:
    if name in {"ComputeProfile", "detect", "precision_name"}:
        from engine.compute import device

        return getattr(device, name)
    if name in {
        "LoadPlan",
        "MAX_BATCH_SIZE",
        "MEMORY_HEADROOM_FACTOR",
        "MIN_BATCH_SIZE",
        "batch_size_for",
        "estimate_seconds",
        "plan_for_model",
    }:
        from engine.compute import budget

        return getattr(budget, name)
    raise AttributeError(f"module 'engine.compute' has no attribute {name!r}")
