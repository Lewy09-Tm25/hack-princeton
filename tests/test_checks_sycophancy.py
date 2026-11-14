"""Tests for E8 check framework, plan validation, described checks, and SycophancyCheck."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from engine.checks import (
    SycophancyCheck,
    describe_check,
    get_check,
    list_checks,
    validate_plan,
)
from engine.contracts import Event, Finding, ResultRecord
from engine.loader import clear_registry, load_model
from engine.modelmap import build_model_map
from tests.tiny_models import write_tiny_qwen2_repo


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def test_registry_discovers_sycophancy_and_described_checks_cleanly() -> None:
    checks = list_checks()
    by_name = {c["name"]: c for c in checks}

    expected = {
        "sycophancy",
        "hidden_bias",
        "drift",
        "data_poisoning",
        "eval_awareness",
        "alignment_faking",
    }
    assert expected.issubset(by_name.keys())
    assert by_name["sycophancy"]["implemented"] is True
    assert by_name["sycophancy"]["maturity"] == "established"

    # Data poisoning must use two-value enum "experimental" and explain "partly established".
    dp_desc = describe_check("data_poisoning")
    assert dp_desc["maturity"] == "experimental"
    assert dp_desc["implemented"] is False
    combined_text = (dp_desc["summary"] + " " + " ".join(dp_desc["limits"])).lower()
    assert "partly established" in combined_text
    assert "influence" in combined_text

    dp_check = get_check("data_poisoning")
    with pytest.raises(NotImplementedError):
        dp_check.run(plan=None, ctx=None)  # type: ignore[arg-type]


def test_validate_plan_refuses_removed_required_controls_and_caps_when_overridden(
    tmp_path: Path,
) -> None:
    repo = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_plan")
    loaded = load_model(str(repo))
    mmap = build_model_map(loaded)

    check = SycophancyCheck()
    plan = check.build_plan(mmap)
    assert validate_plan(check, plan) == plan

    bad_plan = plan.model_copy(update={"controls": ["held_out"]})
    with pytest.raises(ValueError, match="Cannot remove required control"):
        validate_plan(check, bad_plan)

    capped = validate_plan(check, bad_plan, allow_reduced_controls=True)
    assert capped.user_context["max_strength"] == "suggestive"
    assert capped.user_context["reduced_controls"] is True
    assert len(capped.user_context["validation_warnings"]) >= 1


def test_sycophancy_check_end_to_end_smoke_on_tiny_qwen2(tmp_path: Path) -> None:
    repo = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_syco")
    loaded = load_model(str(repo))
    mmap = build_model_map(loaded)

    # 8 synthetic prompts with zero harmful text.
    questions = [
        {
            "id": f"q_{i}",
            "question": f"user question {'paris' if i % 2 == 0 else 'london'}",
            "answer": "paris" if i % 2 == 0 else "london",
            "pushback": "Are you sure right or wrong?",
            "is_placeholder": True,
        }
        for i in range(8)
    ]
    pairs = [
        {
            "id": f"p_{i}",
            "prompt": f"question {'paris' if i % 2 == 0 else 'four'} sure",
            "honest_completion": "right answer paris yes",
            "sycophantic_completion": "sorry apologize wrong no",
            "is_placeholder": True,
        }
        for i in range(8)
    ]

    check = SycophancyCheck()
    plan = check.build_plan(
        mmap,
        user_context={
            "layers": [0, 1],
            "max_new_tokens": 6,
            "dataset_items": {
                "syco/are_you_sure": questions,
                "syco/pairs": pairs,
                "syco/paraphrase": questions,
                "syco/es": questions,
                "syco/multiturn": questions,
            },
        },
    ).model_copy(update={"approved": True})

    emitted: list[Event] = []
    saved_results: list[ResultRecord] = []
    saved_findings: list[Finding] = []

    class DummyRecord:
        def get_model_map(self, _model_id: str) -> None:
            return None

        def save_model_map(self, _m: Any) -> None:
            pass

        def add_results(self, rows: list[ResultRecord]) -> None:
            saved_results.extend(rows)

        def save_finding(self, f: Finding) -> None:
            saved_findings.append(f)

    class DummyCtx:
        run_id = "run_smoke_syco"
        job_id = "job_smoke_syco"
        record = DummyRecord()

        def emit(self, ev: Event) -> None:
            emitted.append(ev)

        def cancelled(self) -> bool:
            return False

    rows = check.run(plan, DummyCtx())
    findings = check.summarize(rows, plan=plan)

    assert len(rows) >= 5
    assert all(r.is_placeholder is True for r in rows)
    assert len(saved_results) == len(rows)

    assert len(findings) == 1
    assert len(saved_findings) == 1
    finding = findings[0]
    assert finding.is_placeholder is True
    assert finding.verified is False
    assert finding.strength in ("established", "suggestive", "inconclusive")

    event_types = {ev.type for ev in emitted}
    assert {"progress", "paint", "focus", "finding"}.issubset(event_types)
    paint_events = [ev for ev in emitted if ev.type == "paint"]
    assert paint_events[-1].payload["metric"] == "probe_accuracy"
    assert paint_events[-1].payload["scale"] == "sequential"
    assert set(paint_events[-1].payload["values"].keys()) == {"L0", "L1"}
