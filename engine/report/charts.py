"""Plotly chart builders for HTML experiment reports (component E12)."""

from __future__ import annotations

import re
from typing import Any

import plotly.graph_objects as go

from engine.contracts import ControlResult, ResultRecord

_LAYER_RE = re.compile(r"^L(\d+)$")

# Colorblind-safe research palette (blue / orange / slate, never red-green only).
COLOR_PRIMARY = "#2b5c8f"
COLOR_REFERENCE = "#d97724"
COLOR_CONTROL_LINE = "#64748b"
COLOR_CONTROL_FILL = "rgba(100, 116, 139, 0.18)"


def _layer_sort_key(component_id: str) -> tuple[int, int, str]:
    match = _LAYER_RE.match(component_id)
    if match:
        return (0, int(match.group(1)), component_id)
    if component_id == "embed":
        return (-1, -1, component_id)
    if component_id == "unembed":
        return (1, 999999, component_id)
    return (2, 0, component_id)


def _checkpoint_sort_key(checkpoint_id: str) -> tuple[int, int, str]:
    digits = re.findall(r"\d+", checkpoint_id)
    if digits:
        return (0, int(digits[-1]), checkpoint_id)
    return (1, 0, checkpoint_id)


def _base_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "font": {"size": 15}},
        "xaxis": {"title": xaxis_title, "gridcolor": "rgba(148, 163, 184, 0.25)"},
        "yaxis": {"title": yaxis_title, "gridcolor": "rgba(148, 163, 184, 0.25)"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 56, "r": 24, "t": 48, "b": 48},
        "height": 320,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    }


def build_layer_curve_html(
    results: list[ResultRecord],
    *,
    div_id: str,
    preferred_metric: str | None = None,
) -> str | None:
    """Build a per-layer measurement curve with paired control values when available."""
    layer_rows = [
        r for r in results if _LAYER_RE.match(r.component_id) or r.component_id.startswith("L")
    ]
    if not layer_rows:
        return None

    metrics_seen = list(dict.fromkeys(r.metric for r in layer_rows))
    metric = preferred_metric if preferred_metric in metrics_seen else metrics_seen[0]
    filtered = [r for r in layer_rows if r.metric == metric]

    # When multiple checkpoints exist, plot the latest checkpoint in the per-layer summary curve.
    checkpoints = sorted(
        {r.checkpoint_id for r in filtered if r.checkpoint_id is not None},
        key=_checkpoint_sort_key,
    )
    chosen_ckpt = checkpoints[-1] if checkpoints else None
    if chosen_ckpt is not None:
        filtered = [r for r in filtered if r.checkpoint_id == chosen_ckpt]

    if filtered:
        primary_seed = min(r.seed for r in filtered)
        filtered = [r for r in filtered if r.seed == primary_seed]

    filtered.sort(key=lambda r: _layer_sort_key(r.component_id))
    if not filtered:
        return None

    x_labels = [r.component_id for r in filtered]
    y_values = [r.value for r in filtered]

    fig = go.Figure()

    ctrl_values = [
        r.control.value
        for r in filtered
        if r.control is not None and r.control.value is not None
    ]
    if len(ctrl_values) == len(filtered):
        upper = [min(1.0, v + 0.05) for v in ctrl_values]
        lower = [max(0.0, v - 0.05) for v in ctrl_values]
        fig.add_trace(
            go.Scatter(
                x=x_labels + x_labels[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=COLOR_CONTROL_FILL,
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name="Control band (+/- 0.05)",
            )
        )
        ctrl_kind = filtered[0].control.kind if filtered[0].control is not None else "control"
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=ctrl_values,
                mode="lines+markers",
                name=f"Control ({ctrl_kind})",
                line={"color": COLOR_CONTROL_LINE, "dash": "dash", "width": 2},
                marker={"size": 6, "symbol": "diamond"},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_values,
            mode="lines+markers",
            name=f"Measured ({metric})",
            line={"color": COLOR_PRIMARY, "width": 2.5},
            marker={"size": 7, "symbol": "circle"},
        )
    )

    ckpt_note = f" ({chosen_ckpt})" if chosen_ckpt else ""
    fig.update_layout(
        **_base_layout(
            title=f"Per-layer {metric}{ckpt_note}",
            xaxis_title="Model component",
            yaxis_title=metric.replace("_", " "),
        )
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def build_reference_comparison_html(
    results: list[ResultRecord],
    *,
    div_id: str,
) -> str | None:
    """Build a bar or line comparison between target model measurements and reference/control."""
    # Look first for explicit model vs reference metrics, or paired reference_model controls.
    ref_controls = [
        r
        for r in results
        if r.control is not None
        and r.control.value is not None
        and ("reference" in r.control.kind or "clean" in r.control.kind or "base" in r.control.kind)
    ]
    if ref_controls:
        ref_controls.sort(key=lambda r: _layer_sort_key(r.component_id))
        labels = [f"{r.component_id} ({r.metric})" for r in ref_controls]
        target_vals = [r.value for r in ref_controls]
        ref_vals = [float(r.control.value) for r in ref_controls if r.control is not None]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=target_vals,
                name="Target model",
                marker_color=COLOR_PRIMARY,
            )
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=ref_vals,
                name="Reference / control model",
                marker_color=COLOR_REFERENCE,
            )
        )
        layout = _base_layout(
            title="Target model vs reference model",
            xaxis_title="Component and metric",
            yaxis_title="Value",
        )
        layout["barmode"] = "group"
        fig.update_layout(**layout)
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

    model_level = [r for r in results if r.component_id in ("model", "reference_model")]
    if model_level:
        labels = [f"{r.component_id}: {r.metric}" for r in model_level]
        values = [r.value for r in model_level]
        colors = [
            COLOR_REFERENCE if "reference" in r.component_id else COLOR_PRIMARY
            for r in model_level
        ]
        fig = go.Figure(
            data=[go.Bar(x=labels, y=values, marker_color=colors, name="Behavioral / aggregate")]
        )
        fig.update_layout(
            **_base_layout(
                title="Model-level comparison",
                xaxis_title="Measurement",
                yaxis_title="Value",
            )
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

    return None


def build_checkpoint_timeline_html(
    results: list[ResultRecord],
    *,
    div_id: str,
) -> str | None:
    """Build a checkpoint timeline chart when measurements span multiple checkpoints."""
    ckpt_rows = [r for r in results if r.checkpoint_id is not None]
    checkpoints = sorted(
        {r.checkpoint_id for r in ckpt_rows if r.checkpoint_id},
        key=_checkpoint_sort_key,
    )
    if len(checkpoints) < 2:
        return None

    metric = ckpt_rows[0].metric
    metric_rows = [r for r in ckpt_rows if r.metric == metric]

    # Plot up to 4 representative components across checkpoints plus mean across layers.
    components = sorted({r.component_id for r in metric_rows}, key=_layer_sort_key)
    if len(components) > 4:
        step = max(1, len(components) // 3)
        selected = components[::step][:3]
        if components[-1] not in selected:
            selected.append(components[-1])
    else:
        selected = components

    palette = ["#2b5c8f", "#d97724", "#0f766e", "#7c3aed"]
    fig = go.Figure()
    for idx, comp_id in enumerate(selected):
        by_ckpt = {
            r.checkpoint_id: r.value
            for r in metric_rows
            if r.component_id == comp_id and r.checkpoint_id is not None
        }
        xs = [c for c in checkpoints if c in by_ckpt]
        ys = [by_ckpt[c] for c in xs]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=comp_id,
                line={"color": palette[idx % len(palette)], "width": 2},
                marker={"size": 7},
            )
        )

    fig.update_layout(
        **_base_layout(
            title=f"Checkpoint timeline ({metric})",
            xaxis_title="Checkpoint / stage",
            yaxis_title=metric.replace("_", " "),
        )
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def format_shift_rows(shift_tests: list[ControlResult]) -> list[dict[str, str]]:
    """Format shift-test outcomes into plain table rows with explicit text labels."""
    rows: list[dict[str, str]] = []
    for item in shift_tests:
        if item.passed is True:
            status_label = "Passed"
            status_class = "status-passed"
        elif item.passed is False:
            status_label = "Failed"
            status_class = "status-failed"
        else:
            status_label = "Not run"
            status_class = "status-not-run"

        rows.append(
            {
                "kind": item.kind,
                "value": f"{item.value:.4f}" if item.value is not None else "n/a",
                "status_label": status_label,
                "status_class": status_class,
                "notes": item.notes or "",
            }
        )
    return rows
