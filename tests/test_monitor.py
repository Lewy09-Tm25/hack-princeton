"""Unit tests for E15 training monitor (`SafetyMonitorCallback`, `StepMonitorHook`, alert rule)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import engine  # noqa: F401 - enforce Windows mcp-before-nnsight import order
from engine.contracts import Event
from engine.monitor import (
    SafetyMonitorCallback,
    StepMonitorHook,
    choose_default_monitor_layers,
    evaluate_alert_rule,
)
from engine.record import Record
from examples.train_with_monitor import main as example_main
from tests.tiny_models import build_tiny_qwen2


def test_evaluate_alert_rule_fewer_than_three_measurements() -> None:
    # First measurement establishes baseline without alerting
    d0 = evaluate_alert_rule([], 2.0)
    assert d0.triggered is False
    assert d0.delta_abs == 0.0
    assert d0.baseline == 2.0
    assert d0.std_first3 is None

    # 1 prior measurement (< 3): 10% shift stays quiet, >= 20% triggers
    quiet_10pct = evaluate_alert_rule([1.0], 1.10)
    assert quiet_10pct.triggered is False
    assert quiet_10pct.std_first3 is None
    assert quiet_10pct.threshold_abs == pytest.approx(0.20)

    alert_20pct = evaluate_alert_rule([1.0], 1.20)
    assert alert_20pct.triggered is True
    assert bool(alert_20pct) is True

    alert_drop = evaluate_alert_rule([1.0, 1.02], 0.78)
    assert alert_drop.triggered is True
    assert alert_drop.std_first3 is None

    # Zero baseline uses floor 1e-6
    zero_quiet = evaluate_alert_rule([0.0], 0.0)
    assert zero_quiet.triggered is False
    zero_alert = evaluate_alert_rule([0.0], 1e-5)
    assert zero_alert.triggered is True


def test_evaluate_alert_rule_three_or_more_low_noise_vs_high_noise() -> None:
    # Case 1: Low noise across first 3 measurements -> 0.20 * |v_0| dominates 3 * std_first3
    low_noise_prior = [1.00, 1.01, 0.99]
    expected_low_std = float(np.std(low_noise_prior, ddof=0))
    assert 3.0 * expected_low_std < 0.20

    quiet_15pct = evaluate_alert_rule(low_noise_prior, 1.15)
    assert quiet_15pct.triggered is False
    assert quiet_15pct.std_first3 == pytest.approx(expected_low_std)
    assert quiet_15pct.threshold_abs == pytest.approx(0.20)

    alert_25pct = evaluate_alert_rule(low_noise_prior, 1.25)
    assert alert_25pct.triggered is True
    assert alert_25pct.threshold_abs == pytest.approx(0.20)

    # Case 2: Higher noise -> 3 * std_first3 > 0.20 * |v_0| ("whichever is larger")
    high_noise_prior = [1.00, 1.20, 0.80]
    expected_high_std = float(np.std(high_noise_prior, ddof=0))
    assert 3.0 * expected_high_std > 0.20  # ~0.4899 > 0.20

    # +30% shift exceeds 20% relative, but stays below 3 * std_first3 (~0.4899) -> quiet
    below_3std = evaluate_alert_rule(high_noise_prior, 1.30)
    assert below_3std.triggered is False
    assert below_3std.std_first3 == pytest.approx(expected_high_std)
    assert below_3std.threshold_abs == pytest.approx(3.0 * expected_high_std)

    # +55% shift exceeds 3 * std_first3 -> triggers alert
    above_3std = evaluate_alert_rule(high_noise_prior, 1.55)
    assert above_3std.triggered is True
    assert above_3std.threshold_abs == pytest.approx(3.0 * expected_high_std)

    # Caller relative_fraction override
    override_alert = evaluate_alert_rule(low_noise_prior, 1.12, relative_fraction=0.10)
    assert override_alert.triggered is True
    assert override_alert.threshold_abs == pytest.approx(0.10)


def test_choose_default_monitor_layers() -> None:
    assert choose_default_monitor_layers(1) == ["L0"]
    assert choose_default_monitor_layers(2) == ["L0", "L1"]
    assert choose_default_monitor_layers(24) == ["L12", "L23"]


def test_safety_monitor_callback_end_to_end_on_tiny_qwen2(tmp_path: Path) -> None:
    model, tokenizer = build_tiny_qwen2(seed=7, num_hidden_layers=2)
    model.train()
    assert model.training is True

    ws_dir = tmp_path / "monitor_ws"
    sink_events: list[Event] = []
    callback = SafetyMonitorCallback(
        checks=["refusal", "sycophancy"],
        every_n_steps=2,
        workspace=ws_dir,
        tokenizer=tokenizer,
        n_prompts=16,
        event_sink=sink_events.append,
        is_placeholder=True,
    )

    args_stub = SimpleNamespace(output_dir=str(ws_dir))
    control_stub = SimpleNamespace(should_training_stop=False)

    # Steps 1..6: measurements run on steps 2, 4, 6; odd steps are skipped.
    for step_idx in range(1, 7):
        assert model.training is True
        ret = callback.on_step_end(
            args=args_stub,
            state=SimpleNamespace(global_step=step_idx),
            control=control_stub,
            model=model,
        )
        assert ret is control_stub
        assert model.training is True
        assert all(mod.training for mod in model.modules())

    # 3 monitored steps (2, 4, 6) x 2 checks x 2 layers = 12 rows, 0 alerts on stable weights
    assert len(callback.results) == 12
    assert callback.alerts == []

    # Step 7 is skipped
    callback.on_step_end(
        args=args_stub,
        state=SimpleNamespace(global_step=7),
        control=control_stub,
        model=model,
    )
    assert len(callback.results) == 12

    # Step 8: inject a forward hook flipping layer hidden-state signs to simulate sharp drift
    def _flip_layer_output(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output * -3.0
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (output[0] * -3.0, *output[1:])
        return output

    hook_handles = [
        layer_mod.register_forward_hook(_flip_layer_output)
        for layer_mod in model.model.layers
    ]
    try:
        assert model.training is True
        callback.on_step_end(
            args=args_stub,
            state=SimpleNamespace(global_step=8),
            control=control_stub,
            model=model,
        )
        assert model.training is True
    finally:
        for h in hook_handles:
            h.remove()

    # Step 8 recorded 4 more rows (total 16) and fired alerts
    assert len(callback.results) == 16
    assert len(callback.alerts) >= 2
    assert all(a["checkpoint_id"] == "step8" for a in callback.alerts)
    assert all(a["is_placeholder"] is True for a in callback.alerts)

    # Verify events delivered to sink
    assert len(sink_events) == len(callback.events)
    timeline_ckpts = [e.payload["checkpoint_id"] for e in sink_events if e.type == "timeline"]
    assert timeline_ckpts == ["step2", "step4", "step6", "step8"]
    alert_events = [
        e for e in sink_events if e.type == "job_status" and e.payload.get("alert") is True
    ]
    assert len(alert_events) == len(callback.alerts)

    # Verify SQLite + Series persistence in Record(workspace)
    with contextlib.closing(Record(ws_dir)) as rec:
        stored_rows = rec.query_results(run_id=callback.run_id)
        assert len(stored_rows) == 16
        assert {r.checkpoint_id for r in stored_rows} == {"step2", "step4", "step6", "step8"}
        assert all(r.is_placeholder is True for r in stored_rows)

        stored_series = rec.get_series(callback.series_id)
        assert stored_series is not None
        assert [it.checkpoint_id for it in stored_series.items] == [
            "step2",
            "step4",
            "step6",
            "step8",
        ]


def test_step_monitor_hook_plain_pytorch_loop(tmp_path: Path) -> None:
    model, tokenizer = build_tiny_qwen2(seed=11, num_hidden_layers=2)
    model.train()

    ws_dir = tmp_path / "hook_ws"
    hook = StepMonitorHook(
        checks=["refusal"],
        every_n_steps=2,
        workspace=ws_dir,
        thresholds={"refusal": 0.15},
        tokenizer=tokenizer,
        layer_ids=["L1"],
        n_prompts=12,
        is_placeholder=True,
    )

    for s in range(1, 7):
        rows = hook.step(s, model)
        if s % 2 == 0:
            assert len(rows) == 1
            assert rows[0].component_id == "L1"
            assert rows[0].checkpoint_id == f"step{s}"
        else:
            assert rows == []
        assert model.training is True

    assert hook.alerts == []

    # Negate layer 1 projection weights on step 8 to trigger alert
    with torch.no_grad():
        for param in model.model.layers[1].parameters():
            param.mul_(-2.5)

    step8_rows = hook.step(8, model)
    assert len(step8_rows) == 1
    assert model.training is True
    assert len(hook.alerts) == 1
    assert hook.alerts[0]["checkpoint_id"] == "step8"
    assert hook.alerts[0]["component_id"] == "L1"


def test_train_with_monitor_cli_guards_and_dry_run_tiny(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Refuses real training when --confirm-train is absent
    rc_no_flag = example_main([])
    captured = capsys.readouterr()
    assert rc_no_flag == 2
    assert "--confirm-train" in captured.err

    # Refuses --confirm-train when --base-dir is missing
    rc_no_base = example_main(["--confirm-train"])
    captured_no_base = capsys.readouterr()
    assert rc_no_base == 2
    assert "--base-dir" in captured_no_base.err

    # Runs offline --dry-run-tiny cleanly and writes placeholder rows to workspace
    dry_ws = tmp_path / "dry_run_ws"
    rc_dry = example_main(
        [
            "--dry-run-tiny",
            "--steps",
            "4",
            "--every-n-steps",
            "2",
            "--workspace",
            str(dry_ws),
        ]
    )
    out_dry = capsys.readouterr()
    assert rc_dry == 0
    assert "Completed tiny dry-run" in out_dry.out

    with contextlib.closing(Record(dry_ws)) as rec:
        all_rows = rec.query_results()
        assert len(all_rows) == 8  # steps 2 and 4 x 2 checks x 2 layers
        assert {r.checkpoint_id for r in all_rows} == {"step2", "step4"}
        assert all(r.is_placeholder is True for r in all_rows)
