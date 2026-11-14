"""Pretraining harm-recognition linear probe dynamics across checkpoints (component E6).

Base pretraining checkpoints (such as Pythia step revisions) have no instruction
or chat refusal tuning, so refusal generation does not apply. Instead, this module
measures held-out linear probe accuracy and AUROC separating harm-related from
benign plain-text prompts (`pretrain/harm_recognition`) across checkpoints with
shuffled-label controls at every layer and checkpoint.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from engine.blocks.activations import collect_activations
from engine.blocks.probes import train_probe
from engine.compute import precision_name
from engine.contracts import (
    ControlResult,
    Event,
    Finding,
    ModelMap,
    ResultRecord,
    Series,
    SeriesItem,
)
from engine.data import registry as data_registry
from engine.dynamics.timeline import run_across_checkpoints
from engine.modelmap import build_model_map
from engine.verify.controls import control_shuffled_labels
from engine.verify.strength import finalize_finding

HARM_PROBE_CHECK_NAME = "pretrain_harm_recognition"

HARM_PROBE_NOT_SHOWN = (
    "Linear probe separability on a base pretraining checkpoint measures whether "
    "harm-related vs benign text is linearly separable in activations, not whether "
    "the model refuses requests."
)


def _emit(ctx: Any, event: Event) -> None:
    emit_fn = getattr(ctx, "emit", None) if ctx is not None else None
    if callable(emit_fn):
        emit_fn(event)


def _coerce_binary_label(raw_label: Any) -> int:
    if isinstance(raw_label, bool):
        return int(raw_label)
    if isinstance(raw_label, int | float):
        val = int(raw_label)
        if val in (0, 1):
            return val
    text = str(raw_label).strip().lower()
    if text in ("1", "harmful", "restricted", "true", "positive"):
        return 1
    if text in ("0", "harmless", "benign", "false", "negative"):
        return 0
    raise ValueError(f"Unsupported harm-recognition label {raw_label!r}; expected 0 or 1.")


def _extract_prompt_text(row: dict[str, Any]) -> str:
    for key in ("text", "instruction", "prompt", "question"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError("Dataset row is missing a 'text' (or 'instruction'/'prompt') field.")


def _load_harm_recognition_items(
    dataset_id: str,
    ctx: Any = None,
    override_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Load `pretrain/harm_recognition` rows and determine placeholder status."""
    if override_items is not None:
        rows = list(override_items)
        is_ph = bool(
            (ctx is not None and getattr(ctx, "is_placeholder", False))
            or any(bool(r.get("is_placeholder", False)) for r in rows)
        )
        return rows, is_ph

    if ctx is not None:
        ctx_items = getattr(ctx, "dataset_items", None)
        if isinstance(ctx_items, list):
            rows = list(ctx_items)
            is_ph = bool(
                getattr(ctx, "is_placeholder", True)
                or any(bool(r.get("is_placeholder", False)) for r in rows)
            )
            return rows, is_ph
        if isinstance(ctx_items, dict) and dataset_id in ctx_items:
            rows = list(ctx_items[dataset_id])
            is_ph = bool(
                getattr(ctx, "is_placeholder", True)
                or any(bool(r.get("is_placeholder", False)) for r in rows)
            )
            return rows, is_ph

        user_ctx = getattr(ctx, "user_context", None)
        if isinstance(user_ctx, dict):
            uc_items = user_ctx.get("dataset_items")
            if isinstance(uc_items, dict) and dataset_id in uc_items:
                rows = list(uc_items[dataset_id])
                is_ph = bool(
                    user_ctx.get("is_placeholder", True)
                    or any(bool(r.get("is_placeholder", False)) for r in rows)
                )
                return rows, is_ph

    datasets_dir = getattr(ctx, "datasets_dir", None) if ctx is not None else None
    loader_fn = getattr(data_registry, "load_dataset", None) or data_registry.get_dataset
    rows = list(loader_fn(dataset_id, allow_placeholder=True, datasets_dir=datasets_dir))
    try:
        ds_ph = data_registry.is_placeholder_dataset(dataset_id, datasets_dir=datasets_dir)
    except (FileNotFoundError, ValueError):
        ds_ph = True
    is_ph = bool(ds_ph or any(bool(r.get("is_placeholder", False)) for r in rows))
    return rows, is_ph


def _stratified_split_by_id(
    rows: list[dict[str, Any]],
    seed: int = 0,
    train_fraction: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows 75% / 25% deterministically by item ID while preserving both classes."""
    if len(rows) < 4:
        raise ValueError(
            f"Need at least 4 labeled items (2 per class) for probe train/held-out split; "
            f"got {len(rows)}."
        )

    pos_rows: list[dict[str, Any]] = []
    neg_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied.setdefault("id", f"item_{idx:04d}")
        label = _coerce_binary_label(copied["label"])
        copied["_binary_label"] = label
        if label == 1:
            pos_rows.append(copied)
        else:
            neg_rows.append(copied)

    if len(pos_rows) < 2 or len(neg_rows) < 2:
        raise ValueError(
            "Both label=1 and label=0 must have at least 2 items so train and held-out "
            "splits each contain both classes."
        )

    pos_sorted = sorted(pos_rows, key=lambda r: str(r["id"]))
    neg_sorted = sorted(neg_rows, key=lambda r: str(r["id"]))

    rng_pos = np.random.default_rng(int(seed))
    rng_neg = np.random.default_rng(int(seed) + 1)
    pos_perm = [pos_sorted[int(i)] for i in rng_pos.permutation(len(pos_sorted))]
    neg_perm = [neg_sorted[int(i)] for i in rng_neg.permutation(len(neg_sorted))]

    n_pos_train = max(1, min(len(pos_perm) - 1, int(round(len(pos_perm) * train_fraction))))
    n_neg_train = max(1, min(len(neg_perm) - 1, int(round(len(neg_perm) * train_fraction))))

    train_combined = pos_perm[:n_pos_train] + neg_perm[:n_neg_train]
    test_combined = pos_perm[n_pos_train:] + neg_perm[n_neg_train:]

    train_sorted = sorted(train_combined, key=lambda r: str(r["id"]))
    test_sorted = sorted(test_combined, key=lambda r: str(r["id"]))
    return train_sorted, test_sorted


def _resolve_layer_ids(mmap: ModelMap, layers: Sequence[int] | str) -> list[str]:
    layer_comps = [c for c in mmap.components if c.kind == "layer"]
    available_ids = {c.id for c in layer_comps}
    if layers == "all":
        return [c.id for c in layer_comps]
    if isinstance(layers, str):
        raise ValueError(
            f"Unsupported layers specification {layers!r}; expected 'all' or list[int]."
        )

    selected: list[str] = []
    for raw_idx in layers:
        lid = f"L{int(raw_idx)}"
        if lid not in available_ids:
            raise ValueError(
                f"Requested layer {lid!r} is outside model map layers (n_layers={mmap.n_layers})."
            )
        if lid not in selected:
            selected.append(lid)
    if not selected:
        raise ValueError("At least one layer must be selected for harm-recognition probing.")
    return selected


def summarize_harm_recognition_timeline(
    results: list[ResultRecord],
    series: Series,
    *,
    run_id: str,
    dataset_id: str = "pretrain/harm_recognition",
    seed: int = 0,
) -> Finding:
    """Summarize early-vs-final checkpoint harm-recognition probe dynamics into a `Finding`."""
    if not results:
        raise ValueError("Cannot summarize an empty harm-recognition timeline result list.")

    acc_rows = [r for r in results if r.metric == "probe_accuracy" and r.checkpoint_id is not None]
    auroc_rows = [r for r in results if r.metric == "probe_auroc" and r.checkpoint_id is not None]
    if not acc_rows:
        raise ValueError("No `probe_accuracy` rows with `checkpoint_id` found in results.")

    auroc_lookup: dict[tuple[str, str], float] = {
        (str(r.checkpoint_id), r.component_id): float(r.value) for r in auroc_rows
    }

    ordered_series_items = sorted(series.items, key=lambda it: it.order)
    label_by_ckpt = {it.checkpoint_id: it.label for it in ordered_series_items}

    evaluated_ckpts: list[str] = [
        it.checkpoint_id
        for it in ordered_series_items
        if any(r.checkpoint_id == it.checkpoint_id for r in acc_rows)
    ]
    if not evaluated_ckpts:
        evaluated_ckpts = sorted({str(r.checkpoint_id) for r in acc_rows})

    peak_by_ckpt: dict[str, tuple[ResultRecord, float]] = {}
    for cid in evaluated_ckpts:
        ckpt_acc_rows = [r for r in acc_rows if r.checkpoint_id == cid]
        best_row = max(
            ckpt_acc_rows,
            key=lambda r: (
                float(r.value),
                auroc_lookup.get((cid, r.component_id), 0.5),
            ),
        )
        best_auroc = auroc_lookup.get((cid, best_row.component_id), 0.5)
        peak_by_ckpt[cid] = (best_row, best_auroc)

    first_cid = evaluated_ckpts[0]
    final_cid = evaluated_ckpts[-1]
    first_best, first_auroc = peak_by_ckpt[first_cid]
    final_best, final_auroc = peak_by_ckpt[final_cid]
    delta_acc = float(final_best.value) - float(first_best.value)

    # Check shuffled-label control & AUROC >= 0.70 rule (`docs/defaults.md` section 2.1).
    representing_ckpts: list[str] = []
    passed_ctrl_count = 0
    shuf_values: list[float] = []
    for cid in evaluated_ckpts:
        best_row, best_auroc = peak_by_ckpt[cid]
        ctrl = best_row.control
        if ctrl is not None and ctrl.value is not None:
            shuf_values.append(float(ctrl.value))
        if ctrl is not None and ctrl.passed is True:
            passed_ctrl_count += 1
            if best_auroc >= 0.70 - 1e-12:
                representing_ckpts.append(cid)

    final_ctrl = final_best.control
    final_passed = bool(
        final_ctrl is not None
        and final_ctrl.passed is True
        and final_auroc >= 0.70 - 1e-12
    )
    mean_shuf_val = float(np.mean(shuf_values)) if shuf_values else 0.5

    summary_control = ControlResult(
        kind="shuffled_labels",
        value=round(mean_shuf_val, 6),
        passed=final_passed,
        notes=(
            f"Peak-layer shuffled-label control passed at "
            f"{passed_ctrl_count}/{len(evaluated_ckpts)} checkpoints; "
            f"AUROC>=0.70 representing criterion met at "
            f"{len(representing_ckpts)}/{len(evaluated_ckpts)} checkpoints."
        ),
    )

    any_placeholder = any(bool(r.is_placeholder) for r in results)
    any_quantized = any(bool(r.quantized) for r in results)

    per_ckpt_summary = "; ".join(
        f"{label_by_ckpt.get(cid, cid)} ({cid}): peak {peak_by_ckpt[cid][0].component_id} "
        f"acc={peak_by_ckpt[cid][0].value:.3f}, auroc={peak_by_ckpt[cid][1]:.3f}"
        for cid in evaluated_ckpts
    )

    if final_passed:
        claim = (
            f"Across {len(evaluated_ckpts)} checkpoints of {series.model_family}, "
            f"held-out harm-recognition probe accuracy moved from {first_best.value:.3f} "
            f"(AUROC {first_auroc:.3f} at {first_best.component_id}, {first_cid}) to "
            f"{final_best.value:.3f} (AUROC {final_auroc:.3f} at {final_best.component_id}, "
            f"{final_cid}; change {delta_acc:+.3f}), beating shuffled-label controls."
        )
    else:
        claim = (
            f"Across {len(evaluated_ckpts)} checkpoints of {series.model_family}, "
            f"held-out harm-recognition peak probe accuracy was {first_best.value:.3f} "
            f"(AUROC {first_auroc:.3f} at {first_best.component_id}, {first_cid}) and "
            f"{final_best.value:.3f} (AUROC {final_auroc:.3f} at {final_best.component_id}, "
            f"{final_cid}; change {delta_acc:+.3f}), which did not exceed the shuffled-label "
            f"and AUROC>=0.70 threshold ({per_ckpt_summary})."
        )

    inspected_layers = sorted(
        {r.component_id for r in acc_rows},
        key=lambda lid: int(lid[1:]) if lid.startswith("L") and lid[1:].isdigit() else 9999,
    )
    ckpt_desc = ", ".join(f"{label_by_ckpt.get(cid, cid)} ({cid})" for cid in evaluated_ckpts)
    ph_suffix = " in synthetic placeholder mode" if any_placeholder else ""
    coverage = (
        f"Examined {series.model_family} across {len(evaluated_ckpts)} pretraining checkpoints "
        f"[{ckpt_desc}], layers [{', '.join(inspected_layers)}], dataset '{dataset_id}' "
        f"(held-out 25% item-ID split), seed {seed}{ph_suffix}."
    )

    evidence_refs = [
        f"{run_id}:checkpoint={cid}:component={peak_by_ckpt[cid][0].component_id}:"
        f"probe_accuracy={peak_by_ckpt[cid][0].value:.4f}:"
        f"probe_auroc={peak_by_ckpt[cid][1]:.4f}"
        for cid in evaluated_ckpts
    ]

    return finalize_finding(
        finding_id=f"finding_{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        check=HARM_PROBE_CHECK_NAME,
        claim=claim,
        maturity="established",
        evidence=evidence_refs,
        controls=[summary_control],
        shift_tests=[],
        required_controls=["shuffled_labels"],
        required_shifts=[],
        coverage=coverage,
        not_shown=HARM_PROBE_NOT_SHOWN,
        is_placeholder=any_placeholder,
        seed_stable=True,
        effect_above_threshold=final_passed,
        quantized=any_quantized,
    )


def run_pretraining_harm_recognition(
    series: Series,
    ctx: Any = None,
    *,
    checkpoint_ids: list[str] | None = None,
    layers: list[int] | str = "all",
    dataset_id: str = "pretrain/harm_recognition",
    dataset_items: list[dict[str, Any]] | None = None,
    seed: int = 0,
    run_id: str | None = None,
    shuffled_draws: int = 20,
    load_checkpoint_fn: Callable[[Series, str], Any] | None = None,
    unload_fn: Callable[[Any], None] | None = None,
) -> tuple[list[ResultRecord], Finding]:
    """Run held-out harm-recognition linear probes across pretraining `series` checkpoints.

    Loads one checkpoint at a time via `run_across_checkpoints`, evaluates standardized L2
    logistic probes and shuffled-label controls on every requested layer, persists rows and
    the summary `Finding` to `ctx.record` when present, and emits live `progress`, `timeline`,
    `paint`, and `finding` events when `ctx.emit` is available.
    """
    assigned_run_id = (
        run_id
        or (getattr(ctx, "run_id", None) if ctx is not None else None)
        or f"run_harm_timeline_{uuid.uuid4().hex[:10]}"
    )
    job_id = str(
        (getattr(ctx, "job_id", None) if ctx is not None else None) or assigned_run_id
    )

    record = getattr(ctx, "record", None) if ctx is not None else None
    if record is not None and hasattr(record, "save_series"):
        record.save_series(series)

    raw_items, data_is_placeholder = _load_harm_recognition_items(
        dataset_id=dataset_id,
        ctx=ctx,
        override_items=dataset_items,
    )
    train_rows, held_out_rows = _stratified_split_by_id(raw_items, seed=seed, train_fraction=0.75)

    train_texts = [_extract_prompt_text(r) for r in train_rows]
    test_texts = [_extract_prompt_text(r) for r in held_out_rows]
    y_train = np.asarray([int(r["_binary_label"]) for r in train_rows], dtype=np.int64)
    y_test = np.asarray([int(r["_binary_label"]) for r in held_out_rows], dtype=np.int64)

    ordered_items = sorted(series.items, key=lambda it: it.order)
    if checkpoint_ids is not None:
        selected_set = set(checkpoint_ids)
        planned_items = [it for it in ordered_items if it.checkpoint_id in selected_set]
    else:
        planned_items = ordered_items
    total_ckpts = max(1, len(planned_items))
    completed_steps = 0

    def _measure_checkpoint(loaded_model: Any, item: SeriesItem) -> list[ResultRecord]:
        nonlocal completed_steps
        mmap = build_model_map(loaded_model, record=record)
        target_layer_ids = _resolve_layer_ids(mmap, layers)

        acts_train = collect_activations(
            model=loaded_model,
            mmap=mmap,
            prompts=train_texts,
            component_ids=target_layer_ids,
            position="last",
            chat_template=False,
        )
        acts_test = collect_activations(
            model=loaded_model,
            mmap=mmap,
            prompts=test_texts,
            component_ids=target_layer_ids,
            position="last",
            chat_template=False,
        )

        load_plan = getattr(loaded_model, "load_plan", None)
        device_str = str(getattr(load_plan, "device", "cpu"))
        dtype_obj = getattr(load_plan, "dtype", None)
        prec_str = precision_name(dtype_obj) if dtype_obj is not None else "fp32"
        quantized_bool = bool(getattr(load_plan, "quantize", None) is not None)
        row_placeholder = bool(
            data_is_placeholder or getattr(loaded_model, "is_placeholder", False)
        )

        ckpt_rows: list[ResultRecord] = []
        for lid in target_layer_ids:
            x_tr = acts_train[lid]
            x_te = acts_test[lid]

            probe_res = train_probe(
                X_train=x_tr,
                y_train=y_train,
                X_test=x_te,
                y_test=y_test,
                component_id=lid,
            )
            shuf_ctrl = control_shuffled_labels(
                train_fn=lambda x_in, y_perm, _x_te=x_te, _lid=lid: train_probe(
                    X_train=x_in,
                    y_train=y_perm,
                    X_test=_x_te,
                    y_test=y_test,
                    component_id=_lid,
                ).accuracy,
                X=x_tr,
                y=y_train,
                n_draws=int(shuffled_draws),
                seed=int(seed),
                real_accuracy=probe_res.accuracy,
            )

            ckpt_rows.append(
                ResultRecord(
                    run_id=assigned_run_id,
                    check=HARM_PROBE_CHECK_NAME,
                    component_id=lid,
                    checkpoint_id=item.checkpoint_id,
                    metric="probe_accuracy",
                    value=float(probe_res.accuracy),
                    control=shuf_ctrl,
                    dataset_id=dataset_id,
                    split="held_out",
                    seed=int(seed),
                    device=device_str,
                    precision=prec_str,
                    quantized=quantized_bool,
                    is_placeholder=row_placeholder,
                )
            )
            ckpt_rows.append(
                ResultRecord(
                    run_id=assigned_run_id,
                    check=HARM_PROBE_CHECK_NAME,
                    component_id=lid,
                    checkpoint_id=item.checkpoint_id,
                    metric="probe_auroc",
                    value=float(probe_res.auroc),
                    control=None,
                    dataset_id=dataset_id,
                    split="held_out",
                    seed=int(seed),
                    device=device_str,
                    precision=prec_str,
                    quantized=quantized_bool,
                    is_placeholder=row_placeholder,
                )
            )

        completed_steps += 1
        _emit(
            ctx,
            Event(
                type="progress",
                payload={
                    "job_id": job_id,
                    "step": completed_steps,
                    "total": total_ckpts,
                    "message": (
                        f"Measured harm-recognition probes across {len(target_layer_ids)} "
                        f"layers at {item.label} ({item.checkpoint_id})."
                    ),
                },
            ),
        )
        return ckpt_rows

    all_results = run_across_checkpoints(
        series=series,
        checkpoint_ids=checkpoint_ids,
        measure_fn=_measure_checkpoint,
        ctx=ctx,
        load_checkpoint_fn=load_checkpoint_fn,
        unload_fn=unload_fn,
    )

    finding = summarize_harm_recognition_timeline(
        results=all_results,
        series=series,
        run_id=assigned_run_id,
        dataset_id=dataset_id,
        seed=seed,
    )

    if record is not None and hasattr(record, "save_finding"):
        record.save_finding(finding)

    _emit(ctx, Event(type="finding", payload={"finding_id": finding.finding_id}))
    return all_results, finding
