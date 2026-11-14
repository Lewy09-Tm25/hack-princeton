"""Architecture inspection and component mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.modelmap.build import apply_confirmation, build_model_map, hook_path

__all__ = [
    "apply_confirmation",
    "build_model_map",
    "hook_path",
]


def __getattr__(name: str) -> Any:
    if name in {"apply_confirmation", "build_model_map", "hook_path"}:
        from engine.modelmap import build

        return getattr(build, name)
    raise AttributeError(f"module 'engine.modelmap' has no attribute {name!r}")
