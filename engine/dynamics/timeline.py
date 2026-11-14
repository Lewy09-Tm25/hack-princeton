"""Checkpoint timeline execution and training-dynamics interfaces (component E6)."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.contracts import Event, PaintScale, ResultRecord, Series, SeriesItem
from engine.data.registry import is_placeholder_dataset

PYTHIA_DEFAULT_CHECKPOINTS: tuple[str, ...] = (
    "step1000",
    "step8000",
    "step16000",
    "step32000",
    "step64000",
    "step143000",
)


@dataclass
class LocalCheckpointHandle:
    """Lightweight handle used when loading a local safetensors checkpoint folder directly."""

    model_id: str
    source: str
    revision: str | None
    model: Any
    tokenizer: Any
    config: Any
    n_params: int
    is_placeholder: bool = False
    _nn: Any = None
    _load_plan: Any = None

    @property
    def hf_model(self) -> Any:
        """Alias matching `LoadedModel.hf_model` for `build_model_map` and `collect_activations`."""
        return self.model

    @property
    def nn(self) -> Any:
        """Lazily wrap `self.model` in `nnsight.TransformersModel` when traced."""
        if self._nn is None and self.model is not None:
            import nnsight

            self._nn = nnsight.TransformersModel(
                self.model,
                tokenizer=self.tokenizer,
                task="text-generation",
            )
        return self._nn

    @property
    def load_plan(self) -> Any:
        """Return a `LoadPlan` matching `LoadedModel.load_plan`."""
        if self._load_plan is None:
            from engine.compute import detect, plan_for_model

            self._load_plan = plan_for_model(self.n_params, detect())
        return self._load_plan


def build_pythia_series(
    model_family: str = "EleutherAI/pythia-70m",
    checkpoints: Sequence[str] = PYTHIA_DEFAULT_CHECKPOINTS,
    series_id: str = "series_pythia70m_pretrain",
) -> Series:
    """Build a pretraining `Series` for Pythia step revisions."""
    items: list[SeriesItem] = []
    for idx, ckpt in enumerate(checkpoints):
        step_digits = ckpt.removeprefix("step")
        label = f"step {int(step_digits):,}" if step_digits.isdigit() else ckpt
        items.append(
            SeriesItem(
                checkpoint_id=ckpt,
                label=label,
                order=idx,
                source=f"revision:{ckpt}",
            )
        )
    return Series(
        series_id=series_id,
        kind="pretraining",
        model_family=model_family,
        items=items,
    )


def _assert_safetensors_folder(folder: Path) -> None:
    safetensor_files = list(folder.glob("*.safetensors"))
    unsafe_files = (
        list(folder.glob("*.bin"))
        + list(folder.glob("*.pt"))
        + list(folder.glob("*.pth"))
        + list(folder.glob("*.pkl"))
    )
    if unsafe_files:
        names = ", ".join(f.name for f in unsafe_files)
        raise ValueError(
            f"Refusing to load unsafe pickle/binary weight files in {folder}: {names}. "
            "Only safetensors weights are allowed."
        )
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors weight files found in {folder}.")


def _default_load_checkpoint(series: Series, checkpoint_id: str) -> Any:
    """Load a single checkpoint via E1 `engine.loader` when available, or from a local folder."""
    try:
        from engine.loader.series import (  # type: ignore[import-not-found]
            load_checkpoint as e1_load_checkpoint,
        )

        return e1_load_checkpoint(series, checkpoint_id)
    except ImportError:
        pass

    matched = [item for item in series.items if item.checkpoint_id == checkpoint_id]
    if not matched:
        raise ValueError(
            f"Checkpoint '{checkpoint_id}' is not present in series '{series.series_id}'."
        )
    item = matched[0]

    candidate_path = Path(item.source)
    if not candidate_path.is_dir():
        raise RuntimeError(
            f"Cannot load non-local checkpoint '{item.source}' without `engine.loader`. "
            "Pass `load_checkpoint_fn` or ensure E1 model loader is installed."
        )

    _assert_safetensors_folder(candidate_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    hf_model = AutoModelForCausalLM.from_pretrained(
        str(candidate_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(candidate_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if (candidate_path / "tokenizer.json").is_file() and len(tokenizer.encode("test")) == 0:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(candidate_path),
            local_files_only=True,
        )
    cfg = hf_model.config
    n_params = sum(int(p.numel()) for p in hf_model.parameters())
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
    synthetic_tiny = (0 < vocab_size <= 512) or (0 < hidden_size <= 64)

    return LocalCheckpointHandle(
        model_id=f"local:{series.model_family}@{checkpoint_id}",
        source=str(candidate_path),
        revision=checkpoint_id,
        model=hf_model,
        tokenizer=tokenizer,
        config=cfg,
        n_params=n_params,
        is_placeholder=synthetic_tiny,
    )


def _default_unload_checkpoint(loaded_model: Any) -> None:
    """Release a loaded checkpoint and clear CPU/GPU caches before loading the next one."""
    try:
        from engine.loader.registry import (  # type: ignore[import-not-found]
            unload_model as e1_unload_model,
        )

        model_id = getattr(loaded_model, "model_id", None)
        if isinstance(model_id, str) and model_id:
            e1_unload_model(model_id)
    except ImportError:
        pass

    if isinstance(loaded_model, LocalCheckpointHandle):
        loaded_model._nn = None
        loaded_model.model = None
        loaded_model.tokenizer = None

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _dataset_is_placeholder(dataset_id: str, ctx: Any) -> bool:
    datasets_dir = getattr(ctx, "datasets_dir", None) if ctx is not None else None
    try:
        return is_placeholder_dataset(dataset_id, datasets_dir=datasets_dir)
    except (FileNotFoundError, ValueError):
        return False


def _emit_checkpoint_events(
    ctx: Any,
    checkpoint_id: str,
    rows: list[ResultRecord],
) -> None:
    emit = getattr(ctx, "emit", None) if ctx is not None else None
    if not callable(emit):
        return

    emit(Event(type="timeline", payload={"checkpoint_id": checkpoint_id}))

    if not rows:
        emit(
            Event(
                type="paint",
                payload={
                    "metric": "value",
                    "values": {},
                    "scale": "sequential",
                    "checkpoint_id": checkpoint_id,
                },
            )
        )
        return

    by_metric: dict[str, dict[str, float]] = {}
    for row in rows:
        by_metric.setdefault(row.metric, {})[row.component_id] = float(row.value)

    run_id = rows[0].run_id
    primary_metric = (
        "probe_accuracy" if "probe_accuracy" in by_metric else next(iter(by_metric))
    )
    comp_values = by_metric[primary_metric]
    scale: PaintScale = (
        "diverging" if any(val < 0.0 for val in comp_values.values()) else "sequential"
    )
    emit(
        Event(
            type="paint",
            payload={
                "metric": primary_metric,
                "values": comp_values,
                "scale": scale,
                "checkpoint_id": checkpoint_id,
                "run_id": run_id,
            },
        )
    )


def run_across_checkpoints(
    series: Series,
    checkpoint_ids: list[str] | None,
    measure_fn: Callable[[Any, SeriesItem], list[ResultRecord]],
    ctx: Any = None,
    load_checkpoint_fn: Callable[[Series, str], Any] | None = None,
    unload_fn: Callable[[Any], None] | None = None,
) -> list[ResultRecord]:
    """Repeat `measure_fn` across ordered checkpoints in `series`, loading one at a time.

    For each selected checkpoint:
    1. Loads the checkpoint (`load_checkpoint_fn` or E1 / local safetensors loader).
    2. Calls `measure_fn(loaded_model, item)` and stamps `checkpoint_id` (plus `is_placeholder`
       whenever the model, context, or dataset is a placeholder).
    3. Saves rows to `ctx.record.add_results(...)` if present and emits `timeline` + `paint`
       events via `ctx.emit(...)` if present.
    4. Unloads the checkpoint (`unload_fn` or default unloader) before loading the next one.
    """
    ordered_items = [
        item
        for _, item in sorted(enumerate(series.items), key=lambda pair: (pair[1].order, pair[0]))
    ]

    if checkpoint_ids is not None:
        known_ids = {item.checkpoint_id for item in ordered_items}
        missing = [cid for cid in checkpoint_ids if cid not in known_ids]
        if missing:
            raise ValueError(
                f"Unknown checkpoint_id(s) for series '{series.series_id}': {', '.join(missing)}"
            )
        selected_set = set(checkpoint_ids)
        target_items = [item for item in ordered_items if item.checkpoint_id in selected_set]
    else:
        target_items = ordered_items

    loader = load_checkpoint_fn or _default_load_checkpoint
    unloader = unload_fn or _default_unload_checkpoint

    all_results: list[ResultRecord] = []
    ctx_placeholder = bool(getattr(ctx, "is_placeholder", False)) if ctx is not None else False
    cancelled_fn = getattr(ctx, "cancelled", None) if ctx is not None else None
    record = getattr(ctx, "record", None) if ctx is not None else None
    add_results_fn = getattr(record, "add_results", None) if record is not None else None

    for item in target_items:
        if callable(cancelled_fn) and bool(cancelled_fn()):
            break

        loaded = loader(series, item.checkpoint_id)
        try:
            model_placeholder = bool(getattr(loaded, "is_placeholder", False))
            raw_rows = measure_fn(loaded, item)
            stamped_rows: list[ResultRecord] = []
            for row in raw_rows:
                ds_placeholder = _dataset_is_placeholder(row.dataset_id, ctx)
                is_ph = bool(
                    row.is_placeholder or model_placeholder or ctx_placeholder or ds_placeholder
                )
                stamped = row.model_copy(
                    update={
                        "checkpoint_id": item.checkpoint_id,
                        "is_placeholder": is_ph,
                    }
                )
                stamped_rows.append(stamped)

            if stamped_rows and callable(add_results_fn):
                add_results_fn(stamped_rows)

            _emit_checkpoint_events(ctx, item.checkpoint_id, stamped_rows)
            all_results.extend(stamped_rows)
        finally:
            unloader(loaded)

    return all_results


def estimate_local_learning_coefficient(
    model_or_series: Any,
    dataset_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Described interface placeholder for local learning coefficient estimation via `devinterp`."""
    raise NotImplementedError(
        "Local learning coefficient via devinterp is described only in this version."
    )
