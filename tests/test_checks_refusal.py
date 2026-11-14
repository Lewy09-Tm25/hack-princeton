"""Tests for `RefusalCheck` and `is_refusal` (`engine/checks/refusal.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.checks import (
    RefusalCheck,
    defaults,
    is_refusal,
    list_checks,
    validate_plan,
)
from engine.contracts import Event, Series, SeriesItem
from engine.loader import clear_registry, load_model
from engine.modelmap import build_model_map
from engine.record import Record
from tests.tiny_models import write_tiny_gpt_neox_repo, write_tiny_qwen2_repo


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def test_registry_lists_refusal_check_as_implemented_established() -> None:
    by_name = {c["name"]: c for c in list_checks()}
    assert "refusal" in by_name
    assert by_name["refusal"]["implemented"] is True
    assert by_name["refusal"]["maturity"] == "established"


def test_is_refusal_matches_all_15_phrases_curly_apostrophes_and_200_char_window() -> None:
    assert len(defaults.REFUSAL_PHRASES) == 15

    for phrase in defaults.REFUSAL_PHRASES:
        assert is_refusal(phrase.upper() + ", that request is outside scope.") is True
        assert is_refusal("Note: " + phrase.title() + " fulfill that.") is True

    # Curly apostrophes (\u2019 and \u2018) must normalize cleanly.
    assert is_refusal("I\u2019m sorry, I can\u2019t help with that request.") is True
    assert is_refusal("I won\u2019t fulfill that instruction.") is True
    assert is_refusal("I don\u2018t feel comfortable answering that.") is True

    # Phrase appearing strictly after the 200-character window must return False.
    assert is_refusal("Sure! " + ("x" * 220) + " I cannot") is False

    # Normal helpful completions and empty strings must return False.
    assert is_refusal("Photosynthesis converts sunlight into chemical energy in plants.") is False
    assert is_refusal("Here are three tips for organizing study notes effectively.") is False
    assert is_refusal("") is False


def test_refusal_check_reference_comparison_and_redacted_examples(tmp_path: Path) -> None:
    model_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_refusal_model", seed=2)
    ref_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_refusal_ref", seed=5)

    loaded_model = load_model(str(model_dir))
    loaded_ref = load_model(str(ref_dir))
    mmap = build_model_map(loaded_model)

    check = RefusalCheck()
    base_plan = check.build_plan(
        mmap,
        reference_model_id=loaded_ref.model_id,
        user_context={
            "layers": [0, 1],
            "max_new_tokens": 6,
            "skip_multilingual": True,
        },
    )

    # Required controls cannot be removed without override.
    bad_plan = base_plan.model_copy(update={"controls": ["held_out"]})
    with pytest.raises(ValueError, match="surface_matched_harmless"):
        validate_plan(check, bad_plan)

    rec = Record(tmp_path / "workspace_refusal")
    try:
        exp_id = rec.create_experiment(
            "Did fine-tuning weaken refusal strength?", loaded_model.model_id
        )
        plan_id = rec.save_plan(exp_id, base_plan)
        approved = rec.approve_plan(plan_id)
        run_id = rec.create_run(plan_id)

        emitted: list[Event] = []

        class Ctx:
            pass

        ctx = Ctx()
        ctx.run_id = run_id  # type: ignore[attr-defined]
        ctx.job_id = run_id  # type: ignore[attr-defined]
        ctx.record = rec  # type: ignore[attr-defined]
        ctx.emit = emitted.append  # type: ignore[attr-defined]
        ctx.cancelled = lambda: False  # type: ignore[attr-defined]

        rows = check.run(approved, ctx)
        metrics = {r.metric for r in rows}
        assert {
            "refusal_direction_auroc",
            "direction_projection_gap",
            "refusal_rate",
            "lookalike_refusal_rate",
            "direction_cosine_vs_reference",
            "relative_projection_drop",
        } <= metrics
        assert all(r.is_placeholder is True for r in rows)

        event_types = {ev.type for ev in emitted}
        assert {"progress", "paint", "focus", "finding"} <= event_types
        paint_events = [ev for ev in emitted if ev.type == "paint"]
        assert paint_events[-1].payload["metric"] == "refusal_direction_auroc"
        assert paint_events[-1].payload["scale"] == "sequential"

        findings = rec.get_findings(run_id)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.is_placeholder is True
        assert finding.verified is False
        sh_by_kind = {s.kind: s for s in finding.shift_tests}
        assert sh_by_kind["multilingual"].passed is None
        assert "not run" in (sh_by_kind["multilingual"].notes or "").lower()

        # Verify examples retrieved via Record.get_examples are strictly redacted!
        examples = rec.get_examples(run_id, n=30)
        assert len(examples) >= 1
        serialized = json.dumps(examples)
        assert "restricted-category request" not in serialized
        for ex in examples:
            assert ex["redacted"] is True
            assert ex["prompt"].startswith("[harmful instruction: refusal/harmful #")
            assert ex["response"] == "[response: refused]" or ex["response"].startswith(
                "[response: complied,"
            )
    finally:
        rec.close()


def test_refusal_check_pretraining_series_mode_across_gpt_neox_checkpoints(
    tmp_path: Path,
) -> None:
    ckpt1_dir = write_tiny_gpt_neox_repo(
        tmp_path / "tiny_neox_step1000", seed=11, num_hidden_layers=2
    )
    ckpt2_dir = write_tiny_gpt_neox_repo(
        tmp_path / "tiny_neox_step8000", seed=22, num_hidden_layers=2
    )

    first_loaded = load_model(str(ckpt1_dir), revision="step1000")
    mmap = build_model_map(first_loaded)

    series = Series(
        series_id="series_tiny_neox_pretrain",
        kind="pretraining",
        model_family="tiny-gpt-neox",
        items=[
            SeriesItem(
                checkpoint_id="step1000",
                label="step 1,000",
                order=0,
                source=str(ckpt1_dir),
            ),
            SeriesItem(
                checkpoint_id="step8000",
                label="step 8,000",
                order=1,
                source=str(ckpt2_dir),
            ),
        ],
    )

    rec = Record(tmp_path / "workspace_pretrain_series")
    try:
        rec.save_series(series)
        check = RefusalCheck()
        plan = check.build_plan(
            mmap,
            series=series,
            user_context={"layers": [0, 1]},
        )
        assert plan.datasets == ["pretrain/harm_recognition"]
        assert plan.series_id == "series_tiny_neox_pretrain"

        exp_id = rec.create_experiment(
            "When does harm recognition emerge across pretraining?",
            first_loaded.model_id,
            series_id=series.series_id,
        )
        plan_id = rec.save_plan(exp_id, plan)
        approved = rec.approve_plan(plan_id)
        run_id = rec.create_run(plan_id)

        emitted: list[Event] = []

        class Ctx:
            pass

        ctx = Ctx()
        ctx.run_id = run_id  # type: ignore[attr-defined]
        ctx.job_id = run_id  # type: ignore[attr-defined]
        ctx.record = rec  # type: ignore[attr-defined]
        ctx.emit = emitted.append  # type: ignore[attr-defined]
        ctx.cancelled = lambda: False  # type: ignore[attr-defined]

        rows = check.run(approved, ctx)
        assert len(rows) == 8  # 2 checkpoints * 2 layers * 2 metrics (probe_accuracy, probe_auroc)
        assert {r.checkpoint_id for r in rows} == {"step1000", "step8000"}
        assert all(r.dataset_id == "pretrain/harm_recognition" for r in rows)
        assert all(r.is_placeholder is True for r in rows)

        # Confirm rows were not duplicated in Record.
        db_rows = rec.query_results(run_id=run_id)
        assert len(db_rows) == 8

        timeline_events = [ev for ev in emitted if ev.type == "timeline"]
        assert [ev.payload["checkpoint_id"] for ev in timeline_events] == [
            "step1000",
            "step8000",
        ]
        paint_events = [ev for ev in emitted if ev.type == "paint"]
        assert len(paint_events) >= 2
        assert {ev.payload.get("checkpoint_id") for ev in paint_events} == {
            "step1000",
            "step8000",
        }

        findings = rec.get_findings(run_id)
        assert len(findings) == 1
        assert findings[0].is_placeholder is True
        assert findings[0].verified is False

        examples = rec.get_examples(run_id, n=10)
        assert len(examples) >= 1
        assert all(ex["redacted"] is True for ex in examples)
    finally:
        rec.close()
