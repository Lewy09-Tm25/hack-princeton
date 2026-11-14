"""Training data attribution interface."""

from __future__ import annotations

from engine.attribution.interface import (
    AttributionMethod,
    DocumentScore,
    attribute,
    search_exact_spans,
)

__all__ = [
    "AttributionMethod",
    "DocumentScore",
    "attribute",
    "search_exact_spans",
]
