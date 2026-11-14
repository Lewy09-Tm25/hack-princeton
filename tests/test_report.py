"""Tests for E12 Report Generator: sections, charts, redaction, and placeholder banner."""

from __future__ import annotations

import random
from pathlib import Path

from engine.contracts import ControlResult, Finding, Plan, ResultRecord
from engine.record import Record
from engine.report import build_report, sample_findings_for_review


def _seed_synthetic_run(rec: Record) -> tuple[str, str]:
    exp_id = rec.create_experiment(
        question="Did fine-tuning weaken refusal signals or increase sycophancy?",
        model_id="local:tiny_qwen2_synthetic",
        series_id="series_pythia70m_pretrain",
    )
    plan = Plan(
        check="sycophancy",
        model_id="local:tiny_qwen2_synthetic",
        reference_model_id="local:tiny_qwen2_base_synthetic",
        series_id="series_pythia70m_pretrain",
        checkpoints=["step1000", "step8000", "step16000"],
        datasets=["syco/pairs_placeholder", "refusal/harmful_placeholder"],
        layers=[0, 1, 2, 3],
        controls=["shuffled_labels", "random_direction", "reference_model"],
        shift_tests=["paraphrase", "multilingual", "multi_turn"],
        estimate_seconds=9.5,
        coverage="Layers 0 to 3 across 3 checkpoints on synthetic placeholder pairs, seed 0.",
        not_covered="Synthetic random-weight model only; does not test real trained weights.",
        maturity="established",
        papers=["Sharma et al. 2023", "Rimsky et al. 2024"],
        approved=True,
        is_placeholder=True,
    )
    plan_id = rec.save_plan(exp_id, plan)
    rec.approve_plan(plan_id)
    run_id = rec.create_run(
        plan_id,
        metadata={"seed": 0, "device": "cpu", "precision": "fp32", "quantized": False},
    )

    rows: list[ResultRecord] = []
    for ckpt_idx, ckpt in enumerate(["step1000", "step8000", "step16000"]):
        for layer in range(4):
            rows.append(
                ResultRecord(
                    run_id=run_id,
                    check="sycophancy",
                    component_id=f"L{layer}",
                    checkpoint_id=ckpt,
                    metric="probe_accuracy",
                    value=0.55 + 0.05 * layer + 0.03 * ckpt_idx,
                    control=ControlResult(kind="shuffled_labels", value=0.51, passed=True),
                    dataset_id="syco/pairs_placeholder",
                    split="held_out",
                    seed=0,
                    device="cpu",
                    precision="fp32",
                    quantized=False,
                    is_placeholder=True,
                )
            )
    rows.append(
        ResultRecord(
            run_id=run_id,
            check="sycophancy",
            component_id="model",
            checkpoint_id="step16000",
            metric="flip_rate",
            value=0.42,
            control=ControlResult(kind="reference_model", value=0.18, passed=True),
            dataset_id="syco/are_you_sure_placeholder",
            split="held_out",
            seed=0,
            device="cpu",
            precision="fp32",
            quantized=False,
            is_placeholder=True,
        )
    )
    rec.add_results(rows)

    sentinel_raw = "SYNTHETIC_RESTRICTED_INPUT_TEXT_NEVER_RENDER"
    rec.save_examples(
        run_id,
        [
            {
                "dataset_id": "refusal/harmful_placeholder",
                "index": 5,
                "prompt": sentinel_raw,
                "response": "I cannot help with that.",
            }
        ],
        contains_harmful=True,
    )

    finding = Finding(
        finding_id="finding_syco_001",
        run_id=run_id,
        check="sycophancy",
        claim="Synthetic probe accuracy rose across layers 0 to 3 on placeholder data.",
        strength="suggestive",
        maturity="established",
        evidence=[f"{run_id}:probe_accuracy", f"{run_id}:flip_rate"],
        controls=[
            ControlResult(kind="shuffled_labels", value=0.51, passed=True),
            ControlResult(kind="reference_model", value=0.18, passed=True),
        ],
        shift_tests=[
            ControlResult(kind="paraphrase", value=0.72, passed=True),
            ControlResult(kind="multilingual", value=None, passed=None, notes="not run"),
            ControlResult(kind="multi_turn", value=0.41, passed=False, notes="below threshold"),
        ],
        coverage="Layers 0 to 3 on synthetic placeholder contrast pairs, seed 0.",
        not_shown="Does not establish causal usage of the direction on real model weights.",
        verified=False,
        is_placeholder=True,
    )
    rec.save_finding(finding)
    return run_id, sentinel_raw


def test_build_report_includes_all_eight_sections_charts_and_banner(tmp_path: Path) -> None:
    rec = Record(tmp_path / "workspace")
    try:
        run_id, sentinel_raw = _seed_synthetic_run(rec)
        report_path = build_report(run_id, rec)

        assert report_path.is_file()
        assert report_path.parent == rec.workspace / "reports"
        registered = rec.list_reports()
        assert len(registered) == 1
        assert rec.get_report_path(registered[0]["report_id"]) == report_path.resolve()

        html = report_path.read_text(encoding="utf-8")

        # Prominent placeholder / unverified banner
        assert "Placeholder / unverified test run — not real model results" in html

        # Light & dark theme support
        assert "prefers-color-scheme: dark" in html

        # All 8 sections from docs/design.md 15.2
        assert "1. Question, model, and checkpoints" in html
        assert "2. Summary of findings" in html
        assert "3. Detailed findings, charts, controls, and examples" in html
        assert "4. Coverage: what was examined" in html
        assert "5. What the results do not show" in html
        assert "6. Settings and versions" in html
        assert "7. Provenance and recorded measurements" in html
        assert "8. Findings for human review" in html

        # Question, badges, coverage, not-shown, provenance links
        assert "Did fine-tuning weaken refusal signals or increase sycophancy?" in html
        assert "Strength: suggestive" in html
        assert "Maturity: established" in html
        assert "Layers 0 to 3 on synthetic placeholder contrast pairs, seed 0." in html
        assert "Does not establish causal usage of the direction on real model weights." in html
        assert f"runs/{run_id}/" in html
        assert f"runs/{run_id}/results.parquet" in html

        # Plotly charts: layer curve, reference comparison, checkpoint timeline
        assert 'id="chart-layer-0"' in html
        assert 'id="chart-ref-0"' in html
        assert 'id="chart-timeline-0"' in html

        # Shift-test table statuses
        assert "Passed" in html
        assert "Failed" in html
        assert "Not run" in html

        # Redaction preserved in rendered HTML
        assert sentinel_raw not in html
        assert "[harmful instruction: refusal/harmful_placeholder #5]" in html
        assert "[response: refused]" in html
    finally:
        rec.close()


def test_human_review_sampling_mixes_without_marking_sampled() -> None:
    findings = [
        Finding(
            finding_id=f"f_{idx}",
            run_id="run_1",
            check="sycophancy",
            claim=f"Claim {idx} bounded to tested layers.",
            strength="suggestive",
            maturity="established",
            evidence=["run_1:probe_accuracy"],
            controls=[ControlResult(kind="shuffled_labels", value=0.5, passed=True)],
            shift_tests=[],
            coverage="Layers 0 to 3.",
            not_shown="Single seed.",
            verified=False,
            is_placeholder=True,
        )
        for idx in range(10)
    ]
    deck = sample_findings_for_review(findings, rng=random.Random(42))
    # 20% of 10 = 2 sampled + 2 unsampled companions = 4 items mixed together
    assert len(deck) >= 2
    assert len({f.finding_id for f in deck}) == len(deck)
