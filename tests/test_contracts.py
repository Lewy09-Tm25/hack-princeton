"""Contract validation tests for shared Python data formats and Phase 0 fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.contracts import (
    Component,
    ControlResult,
    Event,
    Finding,
    ModelMap,
    ResultRecord,
    Series,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize(
    ("filename", "expected_layers", "expected_heads", "expected_arch"),
    [
        ("modelmap_pythia70m.json", 6, 8, "gpt_neox"),
        ("modelmap_gemma3_270m.json", 18, 4, "gemma3"),
        ("modelmap_qwen25_05b.json", 24, 14, "qwen2"),
    ],
)
def test_modelmap_fixtures_validate_and_are_internally_consistent(
    filename: str,
    expected_layers: int,
    expected_heads: int,
    expected_arch: str,
) -> None:
    raw = (FIXTURES_DIR / filename).read_text(encoding="utf-8")
    mmap = ModelMap.model_validate_json(raw)

    assert mmap.n_layers == expected_layers
    assert mmap.arch_family == expected_arch
    assert mmap.confirmed_by_user is True
    assert mmap.needs_confirmation == []

    by_id = {comp.id: comp for comp in mmap.components}
    assert len(by_id) == len(mmap.components), "Duplicate component IDs found"
    assert "embed" in by_id and by_id["embed"].kind == "embedding"
    assert "unembed" in by_id and by_id["unembed"].kind == "unembedding"

    layers = [comp for comp in mmap.components if comp.kind == "layer"]
    assert len(layers) == expected_layers

    for comp in mmap.components:
        for child_id in comp.children:
            assert child_id in by_id, f"Missing child component {child_id} referenced by {comp.id}"

    attn_blocks = [comp for comp in mmap.components if comp.kind == "attention"]
    assert len(attn_blocks) == expected_layers
    for attn in attn_blocks:
        assert attn.n_heads == expected_heads
        assert len(attn.children) == expected_heads
        for head_idx, head_id in enumerate(attn.children):
            head_comp = by_id[head_id]
            assert head_comp.kind == "attention_head"
            assert head_comp.head == head_idx


def test_series_pythia70m_fixture_validates() -> None:
    raw = (FIXTURES_DIR / "series_pythia70m.json").read_text(encoding="utf-8")
    series = Series.model_validate_json(raw)

    assert series.kind == "pretraining"
    assert series.model_family == "EleutherAI/pythia-70m"
    assert len(series.items) == 6
    assert [item.order for item in series.items] == list(range(6))
    assert series.items[0].checkpoint_id == "step1000"
    assert series.items[-1].checkpoint_id == "step143000"
    assert all(item.source.startswith("revision:step") for item in series.items)


def test_events_sycophancy_fixture_validates() -> None:
    lines = (
        (FIXTURES_DIR / "events_sycophancy_run.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert 50 <= len(lines) <= 80

    events = [Event.model_validate_json(line) for line in lines]
    seen_types = {ev.type for ev in events}
    assert {"job_status", "progress", "paint", "focus", "finding"}.issubset(seen_types)
    assert events[0].type == "job_status"
    assert events[0].payload["status"] == "queued"
    assert events[-1].type == "job_status"
    assert events[-1].payload["status"] == "completed"


def test_result_record_and_finding_round_trip() -> None:
    record = ResultRecord(
        run_id="run_fixture_001",
        check="sycophancy",
        component_id="L16",
        checkpoint_id=None,
        metric="probe_accuracy",
        value=0.89,
        control=ControlResult(kind="shuffled_labels", value=0.52, passed=True),
        dataset_id="syco/pairs",
        split="held_out",
        seed=0,
        device="cpu",
        precision="fp32",
        quantized=False,
    )
    restored_record = ResultRecord.model_validate_json(record.model_dump_json())
    assert restored_record == record

    finding = Finding(
        finding_id="finding_fixture_001",
        run_id="run_fixture_001",
        check="sycophancy",
        claim="Sycophancy probe accuracy increased over the base model in layers 14 to 18.",
        strength="suggestive",
        maturity="established",
        evidence=["run_fixture_001:probe_accuracy"],
        controls=[
            ControlResult(kind="shuffled_labels", value=0.52, passed=True),
            ControlResult(kind="random_direction", value=0.03, passed=True),
        ],
        shift_tests=[
            ControlResult(kind="paraphrase", value=0.81, passed=True),
            ControlResult(kind="multilingual", value=None, passed=None, notes="not run"),
        ],
        coverage="Layers 0 to 23 on 200 English contrast pairs, single seed.",
        not_shown=(
            "Does not show that the model uses this direction causally when generating replies."
        ),
        reviewer_verdict={
            "agrees": True,
            "notes": "Claim matches suggestive strength and coverage.",
        },
    )
    restored_finding = Finding.model_validate_json(finding.model_dump_json())
    assert restored_finding == finding


def test_invalid_contracts_rejected() -> None:
    with pytest.raises(ValidationError):
        Component(id="bad", kind="unknown_kind")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Event(type="focus", payload={"reason": "missing component_ids"})

    with pytest.raises(ValidationError):
        Event(type="paint", payload={"metric": "acc", "values": {"L0": 0.5}, "scale": "rainbow"})

    with pytest.raises(ValidationError):
        Event(type="job_status", payload={"job_id": "j1", "status": "unknown"})

    with pytest.raises(ValidationError):
        Finding(
            finding_id="f1",
            run_id="r1",
            check="sycophancy",
            claim="Test",
            strength="established",
            maturity="established",
            evidence=[],
            controls=[],
            shift_tests=[],
            coverage="Layers 0 to 5.",
            not_shown="Single seed.",
            reviewer_verdict={"notes": "missing required 'agrees' boolean"},
        )
