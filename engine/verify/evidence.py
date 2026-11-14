"""Evidence bundle assembly for reviewer inspection (`design.md` section 12.2)."""

from __future__ import annotations

from typing import Any

from engine.contracts import Finding, ResultRecord


def build_evidence_bundle(
    finding: Finding,
    record: Any | None = None,
    max_examples: int = 3,
) -> dict[str, Any]:
    """Assemble the complete, redacted evidence bundle inspected by the reviewer subagent."""
    rows: list[ResultRecord] = []
    if record is not None and hasattr(record, "query_results"):
        raw_rows = record.query_results(run_id=finding.run_id)
        for item in raw_rows or []:
            if isinstance(item, ResultRecord):
                rows.append(item)
            elif isinstance(item, dict):
                rows.append(ResultRecord.model_validate(item))

    examples: list[Any] = []
    if record is not None and hasattr(record, "get_examples"):
        try:
            examples = list(record.get_examples(finding.run_id, n=max_examples) or [])
        except TypeError:
            examples = list(record.get_examples(finding.run_id) or [])[:max_examples]

    devices = sorted({r.device for r in rows})
    precisions = sorted({r.precision for r in rows})
    seeds = sorted({r.seed for r in rows})
    datasets = sorted({r.dataset_id for r in rows})
    any_quantized = any(r.quantized for r in rows)
    any_placeholder = bool(finding.is_placeholder or any(r.is_placeholder for r in rows))

    return {
        "finding_id": finding.finding_id,
        "run_id": finding.run_id,
        "check": finding.check,
        "claim": finding.claim,
        "strength": finding.strength,
        "maturity": finding.maturity,
        "verified": finding.verified,
        "is_placeholder": any_placeholder,
        "coverage": finding.coverage,
        "not_shown": finding.not_shown,
        "evidence": list(finding.evidence),
        "controls": [c.model_dump() for c in finding.controls],
        "shift_tests": [s.model_dump() for s in finding.shift_tests],
        "results": [r.model_dump() for r in rows],
        "settings": {
            "devices": devices,
            "precisions": precisions,
            "quantized": any_quantized,
            "seeds": seeds,
            "datasets": datasets,
            "is_placeholder": any_placeholder,
        },
        "examples": examples,
    }
