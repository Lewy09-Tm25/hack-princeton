"""Tests for E6 checkpoint timeline execution and local tiny GPT-NeoX checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from engine.attribution import DocumentScore, attribute
from engine.contracts import ControlResult, Event, ResultRecord, Series, SeriesItem
from engine.dynamics import (
    HARM_PROBE_CHECK_NAME,
    PYTHIA_DEFAULT_CHECKPOINTS,
    build_pythia_series,
    estimate_local_learning_coefficient,
    run_across_checkpoints,
    run_pretraining_harm_recognition,
    summarize_harm_recognition_timeline,
)
from engine.loader.registry import clear_registry, list_loaded_models, unload_model
from engine.loader.series import load_checkpoint, register_series
from engine.record.db import Record
from tests.tiny_models import write_tiny_gpt_neox_repo


class DummyRecord:
    def __init__(self) -> None:
        self.batches: list[list[ResultRecord]] = []
        self.rows: list[ResultRecord] = []

    def add_results(self, rows: list[ResultRecord]) -> None:
        self.batches.append(list(rows))
        self.rows.extend(rows)


class DummyContext:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.record = DummyRecord()

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _create_three_gpt_neox_checkpoints(base_dir: Path) -> Series:
    steps = [
        ("step1000", "step 1,000", 0, 10),
        ("step8000", "step 8,000", 1, 20),
        ("step16000", "step 16,000", 2, 30),
    ]
    items: list[SeriesItem] = []
    for ckpt_id, label, order, seed in steps:
        ckpt_dir = write_tiny_gpt_neox_repo(base_dir / ckpt_id, seed=seed, num_hidden_layers=2)
        items.append(
            SeriesItem(
                checkpoint_id=ckpt_id,
                label=label,
                order=order,
                source=str(ckpt_dir),
            )
        )
    return Series(
        series_id="series_tiny_neox",
        kind="pretraining",
        model_family="tiny-gpt-neox",
        items=items,
    )


def test_run_across_checkpoints_loads_one_at_a_time_and_stamps_results(tmp_path: Path) -> None:
    series = _create_three_gpt_neox_checkpoints(tmp_path)
    ctx = DummyContext()

    active_ids: list[str] = []
    max_concurrent = 0
    lifecycle_log: list[tuple[str, str]] = []

    from engine.dynamics.timeline import _default_load_checkpoint, _default_unload_checkpoint

    def tracked_load(s: Series, ckpt_id: str) -> Any:
        nonlocal max_concurrent
        handle = _default_load_checkpoint(s, ckpt_id)
        active_ids.append(ckpt_id)
        max_concurrent = max(max_concurrent, len(active_ids))
        lifecycle_log.append(("load", ckpt_id))
        return handle

    def tracked_unload(handle: Any) -> None:
        ckpt_id = str(handle.revision)
        assert active_ids == [ckpt_id], "Previous checkpoint must be unloaded before next load"
        active_ids.remove(ckpt_id)
        lifecycle_log.append(("unload", ckpt_id))
        _default_unload_checkpoint(handle)

    def measure_fn(loaded: Any, item: SeriesItem) -> list[ResultRecord]:
        assert active_ids == [item.checkpoint_id]
        tokens = loaded.tokenizer("question answer safe test", return_tensors="pt")
        with torch.inference_mode():
            outputs = loaded.model(**tokens, output_hidden_states=True)
        rows: list[ResultRecord] = []
        for layer_idx in range(loaded.config.num_hidden_layers):
            h = outputs.hidden_states[layer_idx + 1]
            score = float(torch.sigmoid(h.mean()).item())
            rows.append(
                ResultRecord(
                    run_id="run_timeline_01",
                    check="pretrain_harm_recognition",
                    component_id=f"L{layer_idx}",
                    checkpoint_id=None,
                    metric="probe_accuracy",
                    value=score,
                    control=ControlResult(kind="shuffled_labels", value=0.5, passed=True),
                    dataset_id="pretrain/harm_recognition",
                    split="held_out",
                    seed=0,
                    device="cpu",
                    precision="fp32",
                    quantized=False,
                    is_placeholder=False,
                )
            )
        return rows

    results = run_across_checkpoints(
        series=series,
        checkpoint_ids=None,
        measure_fn=measure_fn,
        ctx=ctx,
        load_checkpoint_fn=tracked_load,
        unload_fn=tracked_unload,
    )

    assert max_concurrent == 1
    assert active_ids == []
    assert lifecycle_log == [
        ("load", "step1000"),
        ("unload", "step1000"),
        ("load", "step8000"),
        ("unload", "step8000"),
        ("load", "step16000"),
        ("unload", "step16000"),
    ]

    assert len(results) == 6
    assert [r.checkpoint_id for r in results] == [
        "step1000",
        "step1000",
        "step8000",
        "step8000",
        "step16000",
        "step16000",
    ]
    assert all(r.is_placeholder is True for r in results)

    assert len(ctx.record.batches) == 3
    assert len(ctx.record.rows) == 6

    assert [e.type for e in ctx.events] == [
        "timeline",
        "paint",
        "timeline",
        "paint",
        "timeline",
        "paint",
    ]
    assert ctx.events[0].payload["checkpoint_id"] == "step1000"
    assert ctx.events[1].payload["checkpoint_id"] == "step1000"
    assert set(ctx.events[1].payload["values"].keys()) == {"L0", "L1"}
    assert ctx.events[4].payload["checkpoint_id"] == "step16000"


def test_subset_filtering_ordering_and_unload_on_error(tmp_path: Path) -> None:
    series = _create_three_gpt_neox_checkpoints(tmp_path)
    # Reverse item list to verify `order` field governs iteration order
    reversed_series = Series(
        series_id=series.series_id,
        kind=series.kind,
        model_family=series.model_family,
        items=list(reversed(series.items)),
    )
    ctx = DummyContext()

    results = run_across_checkpoints(
        series=reversed_series,
        checkpoint_ids=["step16000", "step1000"],
        measure_fn=lambda loaded, item: [
            ResultRecord(
                run_id="run_sub",
                check="sycophancy",
                component_id="L0",
                metric="probe_accuracy",
                value=0.6,
                dataset_id="syco/pairs",
                split="held_out",
                seed=0,
                device="cpu",
                precision="fp32",
                quantized=False,
            )
        ],
        ctx=ctx,
    )
    assert [r.checkpoint_id for r in results] == ["step1000", "step16000"]

    with pytest.raises(ValueError, match="Unknown checkpoint_id"):
        run_across_checkpoints(
            series=series,
            checkpoint_ids=["step999999"],
            measure_fn=lambda loaded, item: [],
            ctx=ctx,
        )

    unloaded: list[str] = []

    def boom_measure(loaded: Any, item: SeriesItem) -> list[ResultRecord]:
        raise RuntimeError("boom inside measure_fn")

    with pytest.raises(RuntimeError, match="boom inside measure_fn"):
        run_across_checkpoints(
            series=series,
            checkpoint_ids=["step1000"],
            measure_fn=boom_measure,
            ctx=ctx,
            load_checkpoint_fn=lambda s, cid: cid,
            unload_fn=lambda loaded: unloaded.append(str(loaded)),
        )
    assert unloaded == ["step1000"]


def test_local_learning_coefficient_stub_and_pythia_helper() -> None:
    pythia = build_pythia_series()
    assert [item.checkpoint_id for item in pythia.items] == list(PYTHIA_DEFAULT_CHECKPOINTS)

    with pytest.raises(
        NotImplementedError,
        match="Local learning coefficient via devinterp is described only in this version.",
    ):
        estimate_local_learning_coefficient(pythia, "pretrain/harm_recognition")

    doc = DocumentScore(document_id="doc_001", score=0.42, method="span_match")
    assert doc.document_id == "doc_001"
    with pytest.raises(NotImplementedError, match="described only in this version"):
        attribute([{"prompt": "p", "completion": "c"}], corpus_ref="local/corpus")


class WorkspaceTimelineContext:
    def __init__(self, record: Record, run_id: str, job_id: str = "job_timeline_e2e") -> None:
        self.record = record
        self.run_id = run_id
        self.job_id = job_id
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def test_end_to_end_pretraining_harm_recognition_three_checkpoints(tmp_path: Path) -> None:
    clear_registry()
    ckpt_root = tmp_path / "neox_checkpoints"
    steps = [
        ("step1000", "step 1,000", 0, 101),
        ("step8000", "step 8,000", 1, 202),
        ("step143000", "step 143,000", 2, 303),
    ]
    items: list[SeriesItem] = []
    for ckpt_id, label, order, seed in steps:
        ckpt_dir = write_tiny_gpt_neox_repo(ckpt_root / ckpt_id, seed=seed, num_hidden_layers=2)
        items.append(
            SeriesItem(
                checkpoint_id=ckpt_id,
                label=label,
                order=order,
                source=str(ckpt_dir),
            )
        )

    series = register_series(
        kind="pretraining",
        items=items,
        model_family="tiny-gpt-neox",
        series_id="series_neox_harm_e2e",
    )

    max_registry_models = 0
    load_unload_trace: list[tuple[str, str]] = []

    def tracked_load(s: Series, ckpt_id: str) -> Any:
        nonlocal max_registry_models
        assert list_loaded_models() == [], "Registry must be empty before loading next checkpoint"
        loaded = load_checkpoint(s, ckpt_id)
        current = list_loaded_models()
        max_registry_models = max(max_registry_models, len(current))
        assert len(current) == 1
        load_unload_trace.append(("load", ckpt_id))
        return loaded

    def tracked_unload(loaded: Any) -> None:
        ckpt_id = str(loaded.revision)
        unload_model(loaded.model_id)
        assert list_loaded_models() == [], "Checkpoint must be evicted immediately on unload"
        load_unload_trace.append(("unload", ckpt_id))

    record = Record(tmp_path / "workspace")
    try:
        run_id = "run_neox_harm_3ckpt"
        ctx = WorkspaceTimelineContext(record=record, run_id=run_id)

        results, finding = run_pretraining_harm_recognition(
            series=series,
            ctx=ctx,
            run_id=run_id,
            seed=0,
            shuffled_draws=20,
            load_checkpoint_fn=tracked_load,
            unload_fn=tracked_unload,
        )

        # 1. Single-checkpoint-in-memory invariant
        assert max_registry_models == 1
        assert list_loaded_models() == []
        assert load_unload_trace == [
            ("load", "step1000"),
            ("unload", "step1000"),
            ("load", "step8000"),
            ("unload", "step8000"),
            ("load", "step143000"),
            ("unload", "step143000"),
        ]

        # 2. Returned ResultRecords: 3 ckpts x 2 layers x 2 metrics + 1 seed-1 row = 13
        assert len(results) == 13
        assert all(r.check == HARM_PROBE_CHECK_NAME for r in results)
        assert all(r.dataset_id == "pretrain/harm_recognition" for r in results)
        assert all(r.split == "held_out" for r in results)
        assert all(r.is_placeholder is True for r in results)

        acc_rows = [r for r in results if r.metric == "probe_accuracy"]
        auroc_rows = [r for r in results if r.metric == "probe_auroc"]
        assert len(acc_rows) == 6
        assert len(auroc_rows) == 6
        for row in acc_rows:
            assert row.control is not None
            assert row.control.kind == "shuffled_labels"
            assert row.control.value is not None
            assert 0.0 <= row.control.value <= 1.0
            assert 0.0 <= row.value <= 1.0

        # 3. Record SQLite + Parquet persistence
        saved_series = record.get_series("series_neox_harm_e2e")
        assert saved_series is not None
        assert [it.checkpoint_id for it in saved_series.items] == [
            "step1000",
            "step8000",
            "step143000",
        ]

        mid_rows = record.query_results(run_id=run_id, checkpoint_id="step8000")
        assert len(mid_rows) == 4
        assert {r.component_id for r in mid_rows} == {"L0", "L1"}
        assert {r.metric for r in mid_rows} == {"probe_accuracy", "probe_auroc"}
        assert (record.run_dir(run_id) / "results.parquet").is_file()

        saved_findings = record.get_findings(run_id)
        assert len(saved_findings) == 1
        assert saved_findings[0].finding_id == finding.finding_id
        assert finding.is_placeholder is True
        assert finding.verified is False
        assert "not whether the model refuses requests" in finding.not_shown
        for ckpt_id in ("step1000", "step8000", "step143000"):
            assert ckpt_id in finding.coverage
            assert any(f"checkpoint={ckpt_id}:" in ref for ref in finding.evidence)
        assert len(finding.controls) == 1
        assert finding.controls[0].kind == "shuffled_labels"

        # 4. Emitted live events (progress, timeline, paint, finding)
        timeline_events = [e for e in ctx.events if e.type == "timeline"]
        paint_events = [e for e in ctx.events if e.type == "paint"]
        progress_events = [e for e in ctx.events if e.type == "progress"]
        finding_events = [e for e in ctx.events if e.type == "finding"]

        assert [e.payload["checkpoint_id"] for e in timeline_events] == [
            "step1000",
            "step8000",
            "step143000",
        ]
        assert [e.payload["checkpoint_id"] for e in paint_events] == [
            "step1000",
            "step8000",
            "step143000",
        ]
        for pe in paint_events:
            assert pe.payload["metric"] == "probe_accuracy"
            assert set(pe.payload["values"].keys()) == {"L0", "L1"}
        assert [e.payload["step"] for e in progress_events] == [1, 2, 3]
        assert all(e.payload["total"] == 3 for e in progress_events)
        assert len(finding_events) == 1
        assert finding_events[0].payload["finding_id"] == finding.finding_id
    finally:
        record.close()
        clear_registry()


def test_harm_probe_layer_subset_and_summary_thresholds(tmp_path: Path) -> None:
    series = _create_three_gpt_neox_checkpoints(tmp_path)
    results, finding = run_pretraining_harm_recognition(
        series=series,
        checkpoint_ids=["step1000", "step16000"],
        layers=[1],
        run_id="run_layer_subset",
        shuffled_draws=5,
        seed=0,
    )
    assert len(results) == 5  # 2 ckpts x 1 layer x 2 metrics + 1 seed-1 row
    assert {r.component_id for r in results} == {"L1"}
    assert {r.checkpoint_id for r in results} == {"step1000", "step16000"}
    assert finding.is_placeholder is True
    assert finding.verified is False

    # Verify summary logic when shuffled-label control & AUROC >= 0.70 pass on non-placeholder data
    non_ph_rows = [
        ResultRecord(
            run_id="run_synth_summary",
            check=HARM_PROBE_CHECK_NAME,
            component_id="L0",
            checkpoint_id="step1000",
            metric="probe_accuracy",
            value=0.58,
            control=ControlResult(kind="shuffled_labels", value=0.50, passed=False),
            dataset_id="pretrain/harm_recognition",
            split="held_out",
            seed=0,
            device="cpu",
            precision="fp32",
            quantized=False,
            is_placeholder=False,
        ),
        ResultRecord(
            run_id="run_synth_summary",
            check=HARM_PROBE_CHECK_NAME,
            component_id="L0",
            checkpoint_id="step1000",
            metric="probe_auroc",
            value=0.61,
            dataset_id="pretrain/harm_recognition",
            split="held_out",
            seed=0,
            device="cpu",
            precision="fp32",
            quantized=False,
            is_placeholder=False,
        ),
        ResultRecord(
            run_id="run_synth_summary",
            check=HARM_PROBE_CHECK_NAME,
            component_id="L0",
            checkpoint_id="step16000",
            metric="probe_accuracy",
            value=0.86,
            control=ControlResult(kind="shuffled_labels", value=0.51, passed=True),
            dataset_id="pretrain/harm_recognition",
            split="held_out",
            seed=0,
            device="cpu",
            precision="fp32",
            quantized=False,
            is_placeholder=False,
        ),
        ResultRecord(
            run_id="run_synth_summary",
            check=HARM_PROBE_CHECK_NAME,
            component_id="L0",
            checkpoint_id="step16000",
            metric="probe_auroc",
            value=0.89,
            dataset_id="pretrain/harm_recognition",
            split="held_out",
            seed=0,
            device="cpu",
            precision="fp32",
            quantized=False,
            is_placeholder=False,
        ),
    ]
    summary = summarize_harm_recognition_timeline(
        non_ph_rows,
        series,
        run_id="run_synth_summary",
    )
    assert summary.is_placeholder is False
    assert summary.verified is True
    assert summary.strength == "established"
    assert "+0.280" in summary.claim


