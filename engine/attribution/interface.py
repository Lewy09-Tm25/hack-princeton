"""Described interface for training data attribution (component E7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AttributionMethod = Literal["influence", "span_match"]


@dataclass(frozen=True)
class DocumentScore:
    """Candidate training document linked to an observed model behavior."""

    document_id: str
    score: float
    method: AttributionMethod = "influence"
    excerpt_redacted: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def attribute(
    behavior_examples: list[dict[str, Any]],
    corpus_ref: str,
) -> list[DocumentScore]:
    """Rank training documents in `corpus_ref` related to `behavior_examples`.

    This component is documented as an interface only in this version; see
    `engine/attribution/README.md` for method details and reliability limits.
    """
    raise NotImplementedError(
        "Training data attribution is described only in this version. "
        "See engine/attribution/README.md."
    )


def search_exact_spans(
    spans: list[str],
    corpus_ref: str,
) -> list[DocumentScore]:
    """Search `corpus_ref` for exact trigger or phrase spans (described interface only)."""
    raise NotImplementedError(
        "Exact training-corpus span search is described only in this version. "
        "See engine/attribution/README.md."
    )
