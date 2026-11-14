"""HTML report generator for experiment runs (component E12)."""

from __future__ import annotations

import hashlib
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from engine.contracts import Finding, ResultRecord
from engine.record import Record, safe_filename
from engine.report.charts import (
    build_checkpoint_timeline_html,
    build_layer_curve_html,
    build_reference_comparison_html,
    format_shift_rows,
)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def sample_findings_for_review(
    findings: list[Finding],
    *,
    rng: random.Random | None = None,
) -> list[Finding]:
    """Select the 20% (minimum 1) human-review sample mixed with unsampled findings.

    Per docs/defaults.md section 11 and docs/design.md section 12.3, at least 20% of
    findings (minimum 1) are randomly sampled and mixed with other findings without
    marking which items were sampled.
    """
    if not findings:
        return []

    generator = rng or random.Random()
    sample_count = max(1, math.ceil(0.20 * len(findings)))
    pool = list(findings)
    sampled = generator.sample(pool, k=min(sample_count, len(pool)))

    sampled_ids = {f.finding_id for f in sampled}
    remaining = [f for f in pool if f.finding_id not in sampled_ids]
    # Mix sampled items with unsampled companions so readers cannot tell which were sampled.
    companion_count = min(len(remaining), max(1, sample_count))
    companions = generator.sample(remaining, k=companion_count) if remaining else []

    combined = sampled + companions
    generator.shuffle(combined)
    return combined


def _collect_settings(
    run_meta: dict[str, Any],
    results: list[ResultRecord],
    datasets_from_plan: list[str],
) -> dict[str, Any]:
    devices = sorted({r.device for r in results}) or [str(run_meta.get("device", "cpu"))]
    precisions = sorted({r.precision for r in results}) or [str(run_meta.get("precision", "fp32"))]
    seeds = sorted({r.seed for r in results})
    if not seeds and "seed" in run_meta:
        seeds = [int(run_meta["seed"])]
    quantized = any(r.quantized for r in results) or bool(run_meta.get("quantized", False))
    datasets = datasets_from_plan or sorted({r.dataset_id for r in results})

    return {
        "device": ", ".join(devices),
        "precision": ", ".join(precisions),
        "quantized": quantized,
        "seeds": ", ".join(str(s) for s in seeds) if seeds else "0",
        "tool_version": str(run_meta.get("tool_version", "0.1.0")),
        "check_version": str(run_meta.get("check_version", "0.1.0")),
        "model_revision": str(run_meta.get("model_revision") or "unspecified"),
        "datasets": ", ".join(datasets) if datasets else "unspecified",
    }


def build_report(
    run_id: str,
    record: Record,
    *,
    report_id: str | None = None,
) -> Path:
    """Build a self-contained HTML report for `run_id`, write it to `reports/`, and register it."""
    run_info = record.get_run(run_id)
    if run_info is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    plan = record.get_plan(run_info["plan_id"], version=run_info.get("plan_version"))
    if plan is None:
        plan = record.get_plan(run_info["plan_id"])

    exp_id = run_info.get("experiment_id") or (plan.experiment_id if plan is not None else None)
    experiment = record.get_experiment(exp_id) if exp_id else None

    result_rows = record.query_results_with_ids(run_id)
    results = [rec for _, rec in result_rows]
    findings = record.get_findings(run_id)
    examples = record.get_examples(run_id, n=3)

    run_meta: dict[str, Any] = dict(run_info.get("metadata") or {})
    is_placeholder = (
        bool(run_meta.get("is_placeholder"))
        or (plan is not None and plan.is_placeholder)
        or any(r.is_placeholder for r in results)
        or any(f.is_placeholder for f in findings)
    )
    has_unverified = (not findings) or any(not f.verified for f in findings)
    show_placeholder_banner = is_placeholder or has_unverified

    question = (
        str(experiment["question"])
        if experiment is not None and experiment.get("question")
        else str(run_meta.get("question") or f"Safety check run ({run_id})")
    )
    model_id = (
        plan.model_id
        if plan is not None
        else str(run_meta.get("model_id") or "unspecified")
    )
    reference_model_id = (
        plan.reference_model_id
        if plan is not None
        else run_meta.get("reference_model_id")
    )
    check_name = (
        plan.check
        if plan is not None
        else str(run_meta.get("check") or (findings[0].check if findings else "unspecified"))
    )

    checkpoints_in_results = list(
        dict.fromkeys(r.checkpoint_id for r in results if r.checkpoint_id is not None)
    )
    if plan is not None and plan.checkpoints:
        checkpoints_label = ", ".join(plan.checkpoints)
    elif checkpoints_in_results:
        checkpoints_label = ", ".join(checkpoints_in_results)
    else:
        checkpoints_label = "single model snapshot"

    finding_blocks: list[dict[str, Any]] = []
    for idx, finding in enumerate(findings):
        check_results = [r for r in results if r.check == finding.check] or results
        finding_blocks.append(
            {
                "finding": finding,
                "layer_chart_html": build_layer_curve_html(
                    check_results,
                    div_id=f"chart-layer-{idx}",
                ),
                "ref_chart_html": build_reference_comparison_html(
                    check_results,
                    div_id=f"chart-ref-{idx}",
                ),
                "timeline_chart_html": build_checkpoint_timeline_html(
                    check_results,
                    div_id=f"chart-timeline-{idx}",
                ),
                "control_rows": format_shift_rows(finding.controls),
                "shift_rows": format_shift_rows(finding.shift_tests),
            }
        )

    standalone_layer_chart_html = (
        build_layer_curve_html(results, div_id="chart-layer-standalone")
        if not finding_blocks and results
        else None
    )

    coverage_statements: list[str] = []
    if plan is not None and plan.coverage:
        coverage_statements.append(plan.coverage)
    for f in findings:
        if f.coverage and f.coverage not in coverage_statements:
            coverage_statements.append(f.coverage)
    if not coverage_statements:
        coverage_statements.append("No explicit coverage statement was recorded for this run.")

    not_shown_statements: list[str] = []
    if plan is not None and plan.not_covered:
        not_shown_statements.append(plan.not_covered)
    for f in findings:
        if f.not_shown and f.not_shown not in not_shown_statements:
            not_shown_statements.append(f.not_shown)
    if not not_shown_statements:
        not_shown_statements.append(
            "Does not establish behavior outside the prompts, layers, and settings "
            "measured in this run."
        )

    settings = _collect_settings(
        run_meta,
        results,
        plan.datasets if plan is not None else [],
    )
    papers = plan.papers if plan is not None else []

    seed_int = int.from_bytes(hashlib.sha256(run_id.encode("utf-8")).digest()[:8], "big")
    review_deck = sample_findings_for_review(findings, rng=random.Random(seed_int))

    assigned_report_id = report_id or f"report_{safe_filename(run_id)}"
    output_path = record.layout.reports_dir / f"{safe_filename(assigned_report_id)}.html"

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")
    rendered = template.render(
        report_id=assigned_report_id,
        run_id=run_id,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        show_placeholder_banner=show_placeholder_banner,
        question=question,
        model_id=model_id,
        reference_model_id=reference_model_id,
        check_name=check_name,
        checkpoints_label=checkpoints_label,
        findings=findings,
        finding_blocks=finding_blocks,
        standalone_layer_chart_html=standalone_layer_chart_html,
        examples=examples,
        coverage_statements=coverage_statements,
        not_shown_statements=not_shown_statements,
        settings=settings,
        papers=papers,
        result_rows=result_rows,
        review_deck=review_deck,
    )

    output_path.write_text(rendered, encoding="utf-8")
    record.register_report(assigned_report_id, run_id, output_path)
    return output_path
