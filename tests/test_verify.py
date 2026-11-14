"""Tests for E10 verification controls, shift rules, strength decisions, and evidence bundles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from engine.blocks.probes import train_probe
from engine.contracts import ControlResult, ResultRecord
from engine.verify import (
    build_evidence_bundle,
    check_seed_stability,
    compare_behavioral_rates,
    control_random_direction,
    control_reference_model,
    control_shuffled_labels,
    decide_strength,
    finalize_finding,
    shift_test,
)


def test_shuffled_labels_passes_planted_signal_and_fails_noise() -> None:
    rng = np.random.default_rng(12)
    n_tr, n_te, dim = 80, 40, 16
    x_tr = rng.normal(size=(n_tr, dim)).astype(np.float32)
    y_tr = np.array([1] * (n_tr // 2) + [0] * (n_tr // 2))
    x_tr[: n_tr // 2, 0] += 3.0
    x_tr[n_tr // 2 :, 0] -= 3.0

    x_te = rng.normal(size=(n_te, dim)).astype(np.float32)
    y_te = np.array([1] * (n_te // 2) + [0] * (n_te // 2))
    x_te[: n_te // 2, 0] += 3.0
    x_te[n_te // 2 :, 0] -= 3.0

    planted_ctrl = control_shuffled_labels(
        lambda x, y: train_probe(x, y, x_te, y_te),
        x_tr,
        y_tr,
        n_draws=20,
        seed=0,
    )
    assert planted_ctrl.kind == "shuffled_labels"
    assert planted_ctrl.passed is True

    noise_x_tr = rng.normal(size=(n_tr, dim)).astype(np.float32)
    noise_x_te = rng.normal(size=(n_te, dim)).astype(np.float32)
    noise_ctrl = control_shuffled_labels(
        lambda x, y: train_probe(x, y, noise_x_te, y_te),
        noise_x_tr,
        y_tr,
        n_draws=20,
        seed=0,
    )
    assert noise_ctrl.passed is False


def test_random_direction_control_uses_10_draws_and_thresholds() -> None:
    target = torch.zeros(32, dtype=torch.float32)
    target[0] = 2.0

    # Real direction aligns with axis 0 (effect = 2.0); random vectors have small axis-0 projection.
    real_ctrl = control_random_direction(
        effect_fn=lambda vec: float(abs(vec[0].item())),
        direction=target,
    )
    assert real_ctrl.kind == "random_direction"
    assert real_ctrl.passed is True
    assert "seeds=100..109" in (real_ctrl.notes or "")

    # Weak effect fails the 2x mean abs / max rule.
    weak_ctrl = control_random_direction(
        effect_fn=lambda _vec: 0.15,
        direction=target,
    )
    assert weak_ctrl.passed is False


def test_behavioral_rate_comparison_and_reference_control() -> None:
    # 45/100 vs 15/100 -> diff = 0.30 >= 0.10 and Fisher p << 0.05 -> passes.
    sig = compare_behavioral_rates(45, 100, 15, 100)
    assert sig["passed"] is True
    assert sig["p_value"] < 0.05

    # 54/100 vs 48/100 -> diff = 0.06 < 0.10 -> fails.
    small_diff = compare_behavioral_rates(54, 100, 48, 100)
    assert small_diff["passed"] is False

    ref_ctrl = control_reference_model(
        run_fn=lambda _ref: (12, 100),
        reference="base_model",
        model_counts=(40, 100),
    )
    assert ref_ctrl.kind == "reference_model"
    assert ref_ctrl.passed is True


def test_shift_test_rules_and_not_run_handling() -> None:
    # Same sign, >= 50% of 0.40 -> 0.25 passes.
    p_shift = shift_test(
        effect_fn=lambda _ds: (0.25, True),
        variant_dataset_id="syco/paraphrase",
        kind="paraphrase",
        original_effect=0.40,
    )
    assert p_shift.passed is True

    # Below 50% size (0.15 < 0.20) -> fails.
    small_shift = shift_test(
        effect_fn=lambda _ds: (0.15, True),
        variant_dataset_id="syco/paraphrase",
        kind="paraphrase",
        original_effect=0.40,
    )
    assert small_shift.passed is False

    # Opposite sign -> fails.
    opp_shift = shift_test(
        effect_fn=lambda _ds: (-0.30, True),
        variant_dataset_id="syco/es",
        kind="multilingual",
        original_effect=0.40,
    )
    assert opp_shift.passed is False

    # Variant control failed -> fails.
    ctrl_fail_shift = shift_test(
        effect_fn=lambda _ds: (0.35, False),
        variant_dataset_id="syco/multiturn",
        kind="multi_turn",
        original_effect=0.40,
    )
    assert ctrl_fail_shift.passed is False

    # Not run -> passed=None, notes="not run".
    not_run = shift_test(effect_fn=None, variant_dataset_id=None, kind="multilingual")
    assert not_run.passed is None
    assert not_run.notes == "not run"


def test_decide_strength_all_branches_and_missing_control_refusal() -> None:
    req_c = ["shuffled_labels", "random_direction"]
    req_s = ["paraphrase", "multilingual"]

    c_pass = [
        ControlResult(kind="shuffled_labels", value=0.50, passed=True),
        ControlResult(kind="random_direction", value=0.04, passed=True),
    ]
    s_pass = [
        ControlResult(kind="paraphrase", value=0.30, passed=True),
        ControlResult(kind="multilingual", value=0.28, passed=True),
    ]

    assert decide_strength(c_pass, s_pass, req_c, req_s, seed_stable=True) == "established"
    assert decide_strength(c_pass, s_pass, req_c, req_s, seed_stable=False) == "suggestive"

    s_not_run = [
        ControlResult(kind="paraphrase", value=0.30, passed=True),
        ControlResult(kind="multilingual", value=None, passed=None, notes="not run"),
    ]
    assert decide_strength(c_pass, s_not_run, req_c, req_s) == "suggestive"

    c_fail = [
        ControlResult(kind="shuffled_labels", value=0.50, passed=False),
        ControlResult(kind="random_direction", value=0.04, passed=True),
    ]
    assert decide_strength(c_fail, s_pass, req_c, req_s) == "inconclusive"

    # Missing required control must raise ValueError.
    with pytest.raises(ValueError, match="random_direction"):
        decide_strength([c_pass[0]], s_pass, req_c, req_s)

    assert check_seed_stability(0.82, 0.85) is True
    assert check_seed_stability(0.82, 0.90) is False


def test_finalize_finding_placeholder_enforcement_and_evidence_bundle() -> None:
    ctrls = [ControlResult(kind="shuffled_labels", value=0.51, passed=True)]
    shifts = [ControlResult(kind="paraphrase", value=0.31, passed=True)]

    placeholder_finding = finalize_finding(
        finding_id="f_placeholder",
        run_id="run_01",
        check="sycophancy",
        claim="Synthetic probe test.",
        maturity="established",
        evidence=["run_01:probe_accuracy"],
        controls=ctrls,
        shift_tests=shifts,
        required_controls=["shuffled_labels"],
        required_shifts=["paraphrase"],
        coverage="Layers 0 to 1 on 8 synthetic prompts.",
        not_shown="Does not establish causal use.",
        is_placeholder=True,
    )
    assert placeholder_finding.is_placeholder is True
    assert placeholder_finding.verified is False
    assert placeholder_finding.strength == "established"

    real_finding = finalize_finding(
        finding_id="f_real",
        run_id="run_02",
        check="sycophancy",
        claim="Probe accuracy rose in layer 14.",
        maturity="experimental",
        evidence=["run_02:probe_accuracy"],
        controls=ctrls,
        shift_tests=shifts,
        required_controls=["shuffled_labels"],
        required_shifts=["paraphrase"],
        coverage="Layers 0 to 23 on 200 English prompts.",
        not_shown="Single seed.",
        is_placeholder=False,
        seed_stable=False,
        quantized=True,
    )
    assert real_finding.verified is True
    assert real_finding.strength == "suggestive"
    assert "approximate (quantized)" in real_finding.not_shown
    assert "unstable across seeds" in real_finding.not_shown
    assert "not well established" in real_finding.not_shown

    class DummyRecord:
        def query_results(self, run_id: str | None = None) -> list[ResultRecord]:
            return [
                ResultRecord(
                    run_id=run_id or "run_02",
                    check="sycophancy",
                    component_id="L14",
                    metric="probe_accuracy",
                    value=0.88,
                    dataset_id="syco/pairs",
                    split="held_out",
                    seed=0,
                    device="cpu",
                    precision="fp32",
                    quantized=True,
                )
            ]

        def get_examples(self, run_id: str, n: int = 3) -> list[dict[str, Any]]:
            return [{"run_id": run_id, "prompt": "example question", "n": n}]

    bundle = build_evidence_bundle(real_finding, DummyRecord())
    assert bundle["finding_id"] == "f_real"
    assert bundle["settings"]["quantized"] is True
    assert len(bundle["results"]) == 1
    assert len(bundle["examples"]) == 1
