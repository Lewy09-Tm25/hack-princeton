"""Tests for E6 checkpoint timeline execution and local tiny GPT-NeoX checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from engine.attribution import DocumentScore, attribute
from engine.contracts import ControlResult, Event, ResultRecord, Series, SeriesItem
from engine.dynamics import (
    PYTHIA_DEFAULT_CHECKPOINTS,
    build_pythia_series,
    estimate_local_learning_coefficient,
    run_across_checkpoints,
)
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

