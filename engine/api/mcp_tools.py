"""All 23 MCP tools and dynamic engine/stub dispatch for the Engine API (component E13)."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from engine.api.approval import (
    append_audit_log,
    ensure_plan_ready_to_run,
    ensure_required_controls_preserved,
    validate_strength_transition,
)
from engine.contracts import (
    ControlResult,
    Event,
    Finding,
    MaturityLevel,
    ModelMap,
    PaintScale,
    Plan,
    ResultRecord,
    ReviewerVerdict,
    Series,
    SeriesItem,
    SeriesKind,
)
from engine.record import Record, safe_filename
from engine.report import build_report as generate_html_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"

# Built-in check metadata used when E8 modules are not yet merged or in stub_mode.
_CHECK_CATALOG: dict[str, dict[str, Any]] = {
    "sycophancy": {
        "name": "sycophancy",
        "maturity": "established",
        "available": True,
        "one_line": "Checks whether the model abandons correct answers under user pushback.",
        "methods": [
            "Behavioral flip rate after user pushback",
            "Per-layer linear probes on honest vs sycophantic contrast pairs",
            "Difference-of-means direction (Rimsky et al. 2024)",
        ],
        "papers": ["Sharma et al. 2023", "Rimsky et al. 2024"],
        "requirements": {
            "needs_training_data": False,
            "recommends_reference_model": True,
        },
        "default_datasets": ["syco/are_you_sure", "syco/pairs"],
        "required_controls": ["random_direction", "shuffled_labels", "held_out"],
        "required_shift_tests": ["paraphrase", "multilingual", "multi_turn"],
        "limits": (
            "Measures pushback agreement on the tested domain and prompt templates; "
            "probe accuracy alone does not prove causal use without steering."
        ),
    },
    "refusal": {
        "name": "refusal",
        "maturity": "established",
        "available": True,
        "one_line": "Measures whether refusal representations and refusal rates weakened.",
        "methods": [
            "Difference-of-means refusal direction across layers (Arditi et al. 2024)",
            "Projection strength and cosine similarity against a reference model",
            "Behavioral refusal rate on held-out instructions",
        ],
        "papers": ["Arditi et al. 2024"],
        "requirements": {
            "needs_training_data": False,
            "recommends_reference_model": True,
        },
        "default_datasets": [
            "refusal/harmful",
            "refusal/harmless",
            "refusal/harmless_lookalike",
        ],
        "required_controls": ["random_direction", "surface_matched_harmless", "held_out"],
        "required_shift_tests": ["paraphrase", "multilingual"],
        "limits": (
            "Covers direct and mechanically wrapped harmful instructions in the listed sets; "
            "does not test novel multi-step jailbreak strategies outside those sets."
        ),
    },
    "backdoor": {
        "name": "backdoor",
        "maturity": "established",
        "available": True,
        "one_line": "Searches for internal probe signals separating triggered from normal inputs.",
        "methods": [
            "Linear probes on generic intent contrast pairs (MacDiarmid et al. 2024)",
            "Comparison against a clean control model across candidate triggers",
        ],
        "papers": ["Hubinger et al. 2024", "MacDiarmid et al. 2024"],
        "requirements": {
            "needs_training_data": False,
            "recommends_reference_model": True,
        },
        "default_datasets": ["backdoor/contrast", "backdoor/triggers"],
        "required_controls": ["clean_reference_model", "shuffled_labels"],
        "required_shift_tests": ["paraphrase", "input_format"],
        "limits": (
            "Trigger search was limited to the listed candidates. A trigger outside this list "
            "would not be detected. Detecting a backdoor signal does not remove the backdoor."
        ),
    },
    "hidden_bias": {
        "name": "hidden_bias",
        "maturity": "established",
        "available": False,
        "one_line": (
            "Probes whether a hidden user attribute is represented and used (interface only)."
        ),
        "methods": ["Attribute linear probes and steering/ablation checks (Cywinski et al. 2025)"],
        "papers": ["Cywinski et al. 2025"],
        "requirements": {"needs_training_data": False, "recommends_reference_model": True},
        "default_datasets": [],
        "required_controls": ["random_direction", "shuffled_labels", "reference_model"],
        "required_shift_tests": ["paraphrase", "topic_shift"],
        "limits": "Documented interface only in this build.",
    },
    "drift": {
        "name": "drift",
        "maturity": "established",
        "available": False,
        "one_line": (
            "Compares activations and safety directions across model versions (interface only)."
        ),
        "methods": [
            "Activation and direction comparison across stages",
            "Model-diff amplification (Aranguri & McGrath 2025)",
        ],
        "papers": ["Betley et al. 2025", "Aranguri & McGrath 2025"],
        "requirements": {"needs_training_data": False, "recommends_reference_model": True},
        "default_datasets": [],
        "required_controls": ["same_model_seed_noise"],
        "required_shift_tests": ["paraphrase"],
        "limits": "Documented interface only in this build.",
    },
    "data_poisoning": {
        "name": "data_poisoning",
        "maturity": "experimental",
        "available": False,
        "one_line": (
            "Partly established: checkpoint timeline is established, while influence-function "
            "training data attribution is experimental and unreliable at scale (interface only)."
        ),
        "methods": ["Checkpoint timeline localization", "Influence functions and text-span search"],
        "papers": ["Barez et al. 2026"],
        "requirements": {"needs_training_data": True, "recommends_reference_model": False},
        "default_datasets": [],
        "required_controls": ["unrelated_behavior_attribution"],
        "required_shift_tests": [],
        "limits": (
            "Requires the training corpus; influence functions work poorly at scale and provide "
            "leads rather than proof. Documented interface only in this build."
        ),
    },
    "eval_awareness": {
        "name": "eval_awareness",
        "maturity": "experimental",
        "available": False,
        "one_line": (
            "Experimental probes separating test-like and real-use prompts (interface only)."
        ),
        "methods": ["Paired evaluation vs deployment linear probes (Nguyen et al. 2025)"],
        "papers": ["Nguyen et al. 2025"],
        "requirements": {"needs_training_data": False, "recommends_reference_model": False},
        "default_datasets": [],
        "required_controls": ["shuffled_labels", "surface_style_pairs"],
        "required_shift_tests": ["paraphrase"],
        "limits": "Effect scales with model size; small models may show little. Interface only.",
    },
    "alignment_faking": {
        "name": "alignment_faking",
        "maturity": "experimental",
        "available": False,
        "one_line": (
            "Experimental comparison of monitored vs unmonitored framings (interface only)."
        ),
        "methods": ["Monitored vs unmonitored compliance and internal signal comparison"],
        "papers": ["Greenblatt et al. 2024"],
        "requirements": {"needs_training_data": False, "recommends_reference_model": False},
        "default_datasets": [],
        "required_controls": ["neutral_framing_control"],
        "required_shift_tests": ["paraphrase"],
        "limits": "Observed cleanly mainly in frontier-scale models. Interface only.",
    },
}

_VALID_COMPONENT_KINDS: set[str] = {
    "embedding",
    "layer",
    "attention",
    "attention_head",
    "mlp",
    "norm",
    "unembedding",
    "other",
}


class LocalEventBus:
    """Thread-safe pub/sub bus broadcasting `Event` objects to async subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[
            tuple[asyncio.AbstractEventLoop | None, asyncio.Queue[Event]]
        ] = []
        self._history: list[Event] = []

    @property
    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)

    def publish(self, job_id_or_event: str | Event, event: Event | None = None) -> None:
        ev = job_id_or_event if isinstance(job_id_or_event, Event) else event
        if ev is None:
            raise ValueError("Event must be provided to publish().")

        with self._lock:
            self._history.append(ev)
            targets = list(self._subscribers)

        for loop, queue in targets:
            if loop is None:
                queue.put_nowait(ev)
            elif not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, ev)

    async def subscribe(self) -> AsyncIterator[Event]:
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        queue: asyncio.Queue[Event] = asyncio.Queue()
        with self._lock:
            self._subscribers.append((loop, queue))
        try:
            while True:
                if not queue.empty():
                    yield queue.get_nowait()
                else:
                    item = await queue.get()
                    yield item
        finally:
            with self._lock:
                self._subscribers = [
                    pair for pair in self._subscribers if pair[1] is not queue
                ]


def _check_safe_local_source(source: str, *, allow_remote_code: bool) -> None:
    """Enforce safetensors-only and remote-code safety checks when `source` is a local path."""
    candidate = Path(source)
    if not candidate.exists():
        return
    if candidate.is_file():
        if candidate.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
            raise ValueError(
                f"Refusing unsafe pickle/binary weight file {candidate.name!r}; "
                "only safetensors weights are allowed."
            )
        return
    if candidate.is_dir():
        unsafe_files = [
            p.name
            for p in candidate.iterdir()
            if p.is_file() and p.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".pkl"}
        ]
        safetensor_files = list(candidate.glob("*.safetensors"))
        if unsafe_files and not safetensor_files:
            raise ValueError(
                f"Refusing folder with unsafe weight files {unsafe_files}; "
                "provide safetensors weights instead."
            )
        cfg_file = candidate / "config.json"
        if cfg_file.is_file() and not allow_remote_code:
            try:
                cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cfg_data = {}
            if isinstance(cfg_data, dict) and cfg_data.get("auto_map"):
                raise ValueError(
                    "Model config requires custom remote code (auto_map), which is disabled "
                    "by default. Explicit approval (allow_remote_code=True) is required."
                )


def _select_fixture_model_map(source: str, revision: str | None = None) -> ModelMap:
    lowered = source.lower()
    if "pythia" in lowered or "gpt_neox" in lowered or "neox" in lowered:
        fixture_name = "modelmap_pythia70m.json"
    elif "gemma" in lowered:
        fixture_name = "modelmap_gemma3_270m.json"
    else:
        fixture_name = "modelmap_qwen25_05b.json"

    mmap = ModelMap.model_validate_json(
        (_FIXTURES_DIR / fixture_name).read_text(encoding="utf-8")
    )
    rev = revision or mmap.revision or "main"
    if source not in mmap.model_id:
        prefix = "local:" if Path(source).exists() else "hf:"
        computed_id = source if source.startswith(("hf:", "local:")) else f"{prefix}{source}@{rev}"
        mmap = mmap.model_copy(update={"model_id": computed_id, "revision": rev})
    elif revision is not None and revision != mmap.revision:
        base = mmap.model_id.split("@")[0]
        mmap = mmap.model_copy(update={"model_id": f"{base}@{revision}", "revision": revision})
    return mmap


class EngineService:
    """Coordinates Record, EventBus, real engine components (when present), and safe stubs."""

    def __init__(
        self,
        record: Record,
        *,
        stub_mode: bool = False,
        event_bus: LocalEventBus | Any | None = None,
    ) -> None:
        self.record = record
        self.stub_mode = stub_mode
        self.event_bus = event_bus or LocalEventBus()
        self._loaded_models: dict[str, Any] = {}
        self._job_states: dict[str, dict[str, Any]] = {}
        self._real_job_manager: Any | None = None
        if not stub_mode:
            self._try_init_real_runner()

    def _try_init_real_runner(self) -> None:
        try:
            from engine.runner import jobs as runner_jobs  # type: ignore[import-not-found]

            job_mgr_cls = getattr(runner_jobs, "JobManager", None)
            if job_mgr_cls is not None:
                try:
                    self._real_job_manager = job_mgr_cls(
                        record=self.record,
                        event_bus=self.event_bus,
                    )
                except TypeError:
                    self._real_job_manager = job_mgr_cls(self.record, self.event_bus)
        except Exception:  # noqa: BLE001
            self._real_job_manager = None

    def _log_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        result: Any = None,
        run_id: str | None = None,
        decision: str = "allow",
        reason: str | None = None,
    ) -> None:
        size: int | None = None
        if result is not None:
            try:
                size = len(json.dumps(result, default=str))
            except Exception:  # noqa: BLE001
                size = None

        append_audit_log(
            self.record.workspace / "agent_log.jsonl",
            tool_name=tool_name,
            arguments=arguments,
            result_size=size,
            decision=decision,
            reason=reason,
        )
        if run_id:
            rdir = self.record.run_dir(run_id)
            append_audit_log(
                rdir / "agent_log.jsonl",
                tool_name=tool_name,
                arguments=arguments,
                result_size=size,
                decision=decision,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # 1. Compute & Model loading / mapping / series
    # ------------------------------------------------------------------

    def get_compute(self) -> dict[str, Any]:
        if not self.stub_mode:
            try:
                from engine.compute.device import detect  # type: ignore[import-not-found]

                prof = detect()
                data = asdict(prof) if is_dataclass(prof) else dict(prof)
                if "dtype" in data:
                    data["dtype"] = str(data["dtype"]).replace("torch.", "")
                self._log_call("get_compute", {}, result=data)
                return data
            except Exception:  # noqa: BLE001
                pass

        fallback = {
            "device": "cpu",
            "dtype": "fp32",
            "free_memory_bytes": None,
            "total_memory_bytes": None,
            "n_gpus": 0,
            "remote_available": False,
            "is_placeholder": True,
        }
        self._log_call("get_compute", {}, result=fallback)
        return fallback

    def load_model(
        self,
        source: str,
        revision: str | None = None,
        adapter: str | None = None,
        allow_remote_code: bool = False,
    ) -> dict[str, Any]:
        args = {
            "source": source,
            "revision": revision,
            "adapter": adapter,
            "allow_remote_code": allow_remote_code,
        }
        _check_safe_local_source(source, allow_remote_code=allow_remote_code)

        if not self.stub_mode:
            try:
                from engine.loader.load import (  # type: ignore[import-not-found]
                    load_model as real_load,
                )
                from engine.modelmap.build import build_model_map  # type: ignore[import-not-found]

                loaded = real_load(
                    source=source,
                    revision=revision,
                    allow_remote_code=allow_remote_code,
                    adapter=adapter,
                )
                mmap = build_model_map(loaded)
                self.record.save_model_map(mmap)
                self._loaded_models[mmap.model_id] = loaded
                out = {
                    "model_id": mmap.model_id,
                    "source": source,
                    "revision": mmap.revision,
                    "arch_family": mmap.arch_family,
                    "n_layers": mmap.n_layers,
                    "hidden_size": mmap.hidden_size,
                    "confirmed_by_user": mmap.confirmed_by_user,
                    "needs_confirmation": mmap.needs_confirmation,
                    "n_params": getattr(loaded, "n_params", None),
                    "is_placeholder": False,
                }
                self._log_call("load_model", args, result=out)
                return out
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass

        mmap = _select_fixture_model_map(source, revision=revision)
        if adapter:
            mmap = mmap.model_copy(update={"model_id": f"{mmap.model_id}+adapter:{adapter}"})
        self.record.save_model_map(mmap)
        out = {
            "model_id": mmap.model_id,
            "source": source,
            "revision": mmap.revision,
            "arch_family": mmap.arch_family,
            "n_layers": mmap.n_layers,
            "hidden_size": mmap.hidden_size,
            "confirmed_by_user": mmap.confirmed_by_user,
            "needs_confirmation": mmap.needs_confirmation,
            "is_placeholder": True,
        }
        self._log_call("load_model", args, result=out)
        return out

    def get_model_map(self, model_id: str) -> dict[str, Any]:
        mmap = self.record.get_model_map(model_id)
        if mmap is None and self.stub_mode:
            for fname in (
                "modelmap_qwen25_05b.json",
                "modelmap_pythia70m.json",
                "modelmap_gemma3_270m.json",
            ):
                candidate = ModelMap.model_validate_json(
                    (_FIXTURES_DIR / fname).read_text(encoding="utf-8")
                )
                if candidate.model_id == model_id:
                    self.record.save_model_map(candidate)
                    mmap = candidate
                    break
        if mmap is None:
            raise KeyError(f"Model map not found for model_id: {model_id!r}")
        out = mmap.model_dump()
        self._log_call("get_model_map", {"model_id": model_id}, result=out)
        return out

    def confirm_components(self, model_id: str, mapping: dict[str, str]) -> dict[str, Any]:
        mmap = self.record.get_model_map(model_id)
        if mmap is None:
            raise KeyError(f"Model map not found for model_id: {model_id!r}")

        if not self.stub_mode:
            try:
                from engine.modelmap.build import (  # type: ignore[import-not-found]
                    apply_confirmation,
                )

                updated = apply_confirmation(mmap, mapping)
                self.record.save_model_map(updated)
                out = updated.model_dump()
                self._log_call(
                    "confirm_components",
                    {"model_id": model_id, "mapping": mapping},
                    result=out,
                )
                return out
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass

        for mod_path, kind in mapping.items():
            if kind not in _VALID_COMPONENT_KINDS:
                raise ValueError(f"Unsupported component kind {kind!r} for module {mod_path!r}.")

        new_components = []
        for comp in mmap.components:
            if comp.module in mapping:
                comp = comp.model_copy(
                    update={"kind": mapping[comp.module]}  # type: ignore[arg-type]
                )
            new_components.append(comp)

        remaining = [m for m in mmap.needs_confirmation if m not in mapping]
        updated = mmap.model_copy(
            update={
                "components": new_components,
                "needs_confirmation": remaining,
                "confirmed_by_user": len(remaining) == 0,
            }
        )
        self.record.save_model_map(updated)
        out = updated.model_dump()
        self._log_call(
            "confirm_components",
            {"model_id": model_id, "mapping": mapping},
            result=out,
        )
        return out

    def register_series(
        self,
        kind: SeriesKind,
        items: list[dict[str, Any]],
        model_family: str = "unknown",
        series_id: str | None = None,
    ) -> dict[str, Any]:
        parsed_items = sorted(
            [SeriesItem.model_validate(item) for item in items],
            key=lambda it: it.order,
        )
        if not self.stub_mode and series_id is None:
            try:
                from engine.loader.series import (  # type: ignore[import-not-found]
                    register_series as real_register_series,
                )

                series = real_register_series(
                    kind=kind,
                    items=parsed_items,
                    model_family=model_family,
                )
                self.record.save_series(series)
                out = {"series_id": series.series_id, "series": series.model_dump()}
                self._log_call(
                    "register_series",
                    {"kind": kind, "model_family": model_family, "n_items": len(parsed_items)},
                    result=out,
                )
                return out
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass

        assigned_id = series_id or f"series_{uuid.uuid4().hex[:10]}"
        series = Series(
            series_id=assigned_id,
            kind=kind,
            model_family=model_family,
            items=parsed_items,
        )
        self.record.save_series(series)
        out = {"series_id": series.series_id, "series": series.model_dump()}
        self._log_call(
            "register_series",
            {"kind": kind, "model_family": model_family, "n_items": len(parsed_items)},
            result=out,
        )
        return out

    # ------------------------------------------------------------------
    # 2. Checks & Plans
    # ------------------------------------------------------------------

    def _get_real_check(self, name: str) -> Any | None:
        if self.stub_mode:
            return None
        try:
            from engine.checks import registry as check_registry  # type: ignore[import-not-found]

            for getter_name in ("get_check", "load_check"):
                getter = getattr(check_registry, getter_name, None)
                if callable(getter):
                    return getter(name)
        except Exception:  # noqa: BLE001
            return None
        return None

    def list_checks(self) -> list[dict[str, Any]]:
        if not self.stub_mode:
            try:
                from engine.checks import (  # type: ignore[import-not-found]
                    registry as check_registry,
                )

                lister = getattr(check_registry, "list_checks", None)
                if callable(lister):
                    listed = lister()
                    if isinstance(listed, list) and listed:
                        self._log_call("list_checks", {}, result=listed)
                        return listed
            except Exception:  # noqa: BLE001
                pass

        entries = [
            {
                "name": info["name"],
                "maturity": info["maturity"],
                "available": info["available"],
                "description": info["one_line"],
            }
            for info in _CHECK_CATALOG.values()
        ]
        self._log_call("list_checks", {}, result=entries)
        return entries

    def describe_check(self, name: str) -> dict[str, Any]:
        real_check = self._get_real_check(name)
        if real_check is not None and hasattr(real_check, "describe"):
            desc = real_check.describe()
            if isinstance(desc, dict):
                self._log_call("describe_check", {"name": name}, result=desc)
                return desc

        if name not in _CHECK_CATALOG:
            raise KeyError(f"Unknown risk check: {name!r}")
        info = dict(_CHECK_CATALOG[name])
        self._log_call("describe_check", {"name": name}, result=info)
        return info

    def _required_controls_for(self, check_name: str) -> list[str]:
        real_check = self._get_real_check(check_name)
        if real_check is not None and hasattr(real_check, "required_controls"):
            raw = real_check.required_controls()
            return [c if isinstance(c, str) else getattr(c, "kind", str(c)) for c in raw]
        if check_name in _CHECK_CATALOG:
            return list(_CHECK_CATALOG[check_name]["required_controls"])
        return []

    def _required_shifts_for(self, check_name: str) -> list[str]:
        real_check = self._get_real_check(check_name)
        if real_check is not None and hasattr(real_check, "required_shift_tests"):
            raw = real_check.required_shift_tests()
            return [s if isinstance(s, str) else getattr(s, "kind", str(s)) for s in raw]
        if check_name in _CHECK_CATALOG:
            return list(_CHECK_CATALOG[check_name]["required_shift_tests"])
        return []

    def draft_plan(
        self,
        check: str,
        model_id: str,
        reference_model_id: str | None = None,
        series_id: str | None = None,
        checkpoints: list[str] | None = None,
        layers: list[int] | str = "all",
        user_context: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        if check not in _CHECK_CATALOG and self._get_real_check(check) is None:
            raise KeyError(f"Unknown risk check: {check!r}")

        ctx = dict(user_context or {})
        mmap = self.record.get_model_map(model_id)
        series = self.record.get_series(series_id) if series_id else None

        real_check = self._get_real_check(check)
        plan: Plan | None = None
        if real_check is not None and hasattr(real_check, "build_plan") and mmap is not None:
            try:
                built = real_check.build_plan(mmap, series, ctx)
                if isinstance(built, Plan):
                    plan = built
            except Exception:  # noqa: BLE001
                plan = None

        if plan is None:
            meta = _CHECK_CATALOG[check]
            if not meta["available"]:
                raise ValueError(
                    f"Check {check!r} is documented as an interface only in this version: "
                    f"{meta['limits']}"
                )
            resolved_ckpts = checkpoints
            if resolved_ckpts is None and series is not None:
                resolved_ckpts = [it.checkpoint_id for it in series.items]

            layer_spec: list[int] | str = "all" if layers == "all" else [int(x) for x in layers]
            n_layers = mmap.n_layers if mmap is not None else 6
            layer_label = (
                f"all {n_layers} layers (0 to {max(0, n_layers - 1)})"
                if layer_spec == "all"
                else f"layers {layer_spec}"
            )
            is_placeholder = bool(
                self.stub_mode
                or ctx.get("is_placeholder")
                or "placeholder" in model_id.lower()
                or "synthetic" in model_id.lower()
                or mmap is None
            )
            datasets = list(meta["default_datasets"])
            if is_placeholder:
                datasets = [
                    ds if ds.endswith("_placeholder") else f"{ds}_placeholder" for ds in datasets
                ]

            coverage = f"Examines {layer_label} on {', '.join(datasets)} (seed 0)."
            not_covered = str(meta["limits"])
            if check == "backdoor":
                coverage = (
                    f"{coverage} Trigger search was limited to the listed candidates. "
                    "A trigger outside this list would not be detected."
                )
                not_covered = "Detecting a backdoor signal does not remove the backdoor."

            maturity: MaturityLevel = meta["maturity"]
            plan = Plan(
                check=check,
                model_id=model_id,
                reference_model_id=reference_model_id,
                series_id=series_id,
                checkpoints=resolved_ckpts,
                datasets=datasets,
                layers=layer_spec,  # type: ignore[arg-type]
                controls=list(meta["required_controls"]),
                shift_tests=list(meta["required_shift_tests"]),
                user_context=ctx,
                estimate_seconds=float(ctx.get("estimate_seconds", 15.0)),
                coverage=coverage,
                not_covered=not_covered,
                maturity=maturity,
                papers=list(meta["papers"]),
                approved=False,
                is_placeholder=is_placeholder,
            )

        exp_question = question or str(ctx.get("question") or f"Investigate {check} on {model_id}")
        exp_id = self.record.create_experiment(
            question=exp_question,
            model_id=model_id,
            series_id=series_id,
        )
        plan.approved = False
        self.record.save_plan(exp_id, plan)
        saved = self.record.get_plan(plan.plan_id or "")
        assert saved is not None
        out = saved.model_dump()
        self._log_call(
            "draft_plan",
            {"check": check, "model_id": model_id, "series_id": series_id},
            result=out,
        )
        return out

    def revise_plan(self, plan_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        current = self.record.get_plan(plan_id)
        if current is None:
            raise KeyError(f"Unknown plan_id: {plan_id}")

        req_controls = self._required_controls_for(current.check)
        req_shifts = self._required_shifts_for(current.check)
        warnings: list[str] = []

        updates: dict[str, Any] = {}
        if "controls" in changes and changes["controls"] is not None:
            candidate_controls = [str(c) for c in changes["controls"]]
            try:
                updates["controls"] = ensure_required_controls_preserved(
                    current.check,
                    req_controls,
                    candidate_controls,
                )
            except ValueError as exc:
                self._log_call(
                    "revise_plan",
                    {"plan_id": plan_id, "changes": changes},
                    decision="deny",
                    reason=str(exc),
                )
                raise

        if "shift_tests" in changes and changes["shift_tests"] is not None:
            new_shifts = [str(s) for s in changes["shift_tests"]]
            dropped_shifts = [s for s in req_shifts if s not in new_shifts]
            if dropped_shifts:
                warnings.append(
                    f"Skipped shift test(s) {dropped_shifts}; finding strength will be capped "
                    "at 'suggestive'."
                )
            updates["shift_tests"] = new_shifts

        for key in (
            "reference_model_id",
            "series_id",
            "checkpoints",
            "datasets",
            "layers",
            "estimate_seconds",
            "coverage",
            "not_covered",
        ):
            if key in changes:
                updates[key] = changes[key]

        if "user_context" in changes and isinstance(changes["user_context"], dict):
            merged_ctx = dict(current.user_context)
            merged_ctx.update(changes["user_context"])
            updates["user_context"] = merged_ctx

        # Revising always resets approval to False.
        updates["approved"] = False
        revised = current.model_copy(update=updates)
        exp_id = current.experiment_id or self.record.create_experiment(
            question=f"Revised {current.check} plan",
            model_id=current.model_id,
            series_id=current.series_id,
        )
        self.record.save_plan(exp_id, revised)
        saved = self.record.get_plan(plan_id)
        assert saved is not None

        out = {"plan": saved.model_dump(), "warnings": warnings}
        self._log_call("revise_plan", {"plan_id": plan_id, "changes": changes}, result=out)
        return out

    # ------------------------------------------------------------------
    # 3. Job execution, status, decisions, stopping
    # ------------------------------------------------------------------

    def run_plan(
        self,
        plan_id: str,
        approved_by_user: bool = False,
        confirm_long_run: bool = False,
    ) -> dict[str, Any]:
        args = {
            "plan_id": plan_id,
            "approved_by_user": approved_by_user,
            "confirm_long_run": confirm_long_run,
        }
        try:
            plan = ensure_plan_ready_to_run(
                self.record,
                plan_id,
                approved_by_user=approved_by_user,
                confirm_long_run=confirm_long_run,
            )
        except Exception as exc:
            self._log_call("run_plan", args, decision="deny", reason=str(exc))
            raise

        if self._real_job_manager is not None and not self.stub_mode and not plan.is_placeholder:
            job_id = self._real_job_manager.submit(plan)
            out = {
                "job_id": job_id,
                "run_id": job_id,
                "status": "queued",
                "is_placeholder": False,
            }
            self._log_call("run_plan", args, result=out, run_id=job_id)
            return out

        run_id = self._execute_placeholder_run(plan)
        out = {
            "job_id": run_id,
            "run_id": run_id,
            "status": self._job_states[run_id]["status"],
            "is_placeholder": True,
        }
        self._log_call("run_plan", args, result=out, run_id=run_id)
        return out

    def _execute_placeholder_run(self, plan: Plan) -> str:
        """Execute a deterministic synthetic placeholder run with live events and controls."""
        assert plan.plan_id is not None
        run_id = self.record.create_run(
            plan.plan_id,
            metadata={"is_placeholder": True, "device": "cpu", "precision": "fp32"},
        )
        messages: list[str] = ["Placeholder check queued"]
        self._job_states[run_id] = {
            "job_id": run_id,
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "status": "running",
            "messages": messages,
            "question": None,
            "answers": [],
            "is_placeholder": True,
        }

        self.event_bus.publish(
            run_id,
            Event(
                type="job_status",
                payload={
                    "job_id": run_id,
                    "status": "running",
                    "message": "Running synthetic placeholder check",
                },
            ),
        )
        self.record.set_run_status(run_id, "running", "Running synthetic placeholder check")

        mmap = self.record.get_model_map(plan.model_id)
        n_layers = min(mmap.n_layers if mmap is not None else 4, 6)
        layer_indices = (
            list(range(n_layers))
            if plan.layers == "all"
            else [int(i) for i in plan.layers][:6] or [0, 1]
        )
        checkpoints = plan.checkpoints or [None]  # type: ignore[list-item]
        metric_name = "probe_accuracy" if plan.check != "refusal" else "refusal_projection"

        rows: list[ResultRecord] = []
        total_steps = max(1, len(checkpoints) * len(layer_indices))
        step_num = 0
        for ckpt in checkpoints:
            if ckpt is not None:
                self.event_bus.publish(
                    run_id,
                    Event(type="timeline", payload={"checkpoint_id": ckpt}),
                )
            painted: dict[str, float] = {}
            for pos, layer_idx in enumerate(layer_indices):
                step_num += 1
                comp_id = f"L{layer_idx}"
                val = round(0.55 + 0.06 * pos, 4)
                painted[comp_id] = val
                rows.append(
                    ResultRecord(
                        run_id=run_id,
                        check=plan.check,
                        component_id=comp_id,
                        checkpoint_id=ckpt,
                        metric=metric_name,
                        value=val,
                        control=ControlResult(kind=plan.controls[0], value=0.51, passed=True),
                        dataset_id=plan.datasets[0] if plan.datasets else "synthetic_placeholder",
                        split="held_out",
                        seed=0,
                        device="cpu",
                        precision="fp32",
                        quantized=False,
                        is_placeholder=True,
                    )
                )
                msg = f"Measured {comp_id} ({step_num}/{total_steps})"
                messages.append(msg)
                self.event_bus.publish(
                    run_id,
                    Event(
                        type="progress",
                        payload={
                            "job_id": run_id,
                            "step": step_num,
                            "total": total_steps,
                            "message": msg,
                        },
                    ),
                )
                self.event_bus.publish(
                    run_id,
                    Event(
                        type="paint",
                        payload={
                            "metric": metric_name,
                            "values": dict(painted),
                            "scale": "sequential",
                            "checkpoint_id": ckpt,
                            "run_id": run_id,
                        },
                    ),
                )

        self.record.add_results(rows)

        top_comp = f"L{layer_indices[-1]}"
        self.event_bus.publish(
            run_id,
            Event(
                type="focus",
                payload={
                    "component_ids": [top_comp],
                    "reason": f"Highest synthetic {metric_name} observed at {top_comp}",
                },
            ),
        )

        contains_harmful = plan.check == "refusal"
        self.record.save_examples(
            run_id,
            [
                {
                    "dataset_id": plan.datasets[0] if plan.datasets else "synthetic_placeholder",
                    "index": 0,
                    "prompt": "Synthetic placeholder evaluation prompt #0",
                    "response": "I'm sorry, I cannot assist with that."
                    if contains_harmful
                    else "Synthetic placeholder model reply.",
                }
            ],
            contains_harmful=contains_harmful,
        )

        finding_id = f"finding_{uuid.uuid4().hex[:10]}"
        controls = [
            ControlResult(kind=ctrl, value=0.51, passed=True) for ctrl in plan.controls
        ]
        shifts = [
            ControlResult(kind=st, value=0.71, passed=True) for st in plan.shift_tests
        ]
        finding = Finding(
            finding_id=finding_id,
            run_id=run_id,
            check=plan.check,
            claim=(
                f"Placeholder {plan.check} signal peaked at {top_comp} on synthetic fixture data "
                "(unverified placeholder run)."
            ),
            strength="suggestive",
            maturity=plan.maturity,
            evidence=[f"{run_id}:{metric_name}"],
            controls=controls,
            shift_tests=shifts,
            coverage=plan.coverage,
            not_shown=f"{plan.not_covered} Placeholder run on synthetic data, not real weights.",
            verified=False,
            is_placeholder=True,
        )
        self.record.save_finding(finding)
        self.event_bus.publish(
            run_id,
            Event(type="finding", payload={"finding_id": finding_id}),
        )

        if plan.user_context.get("pause_for_decision"):
            question_text = str(plan.user_context["pause_for_decision"])
            self._job_states[run_id]["status"] = "needs_decision"
            self._job_states[run_id]["question"] = question_text
            self.record.set_run_status(run_id, "needs_decision", question_text)
            self.event_bus.publish(
                run_id,
                Event(type="question", payload={"job_id": run_id, "text": question_text}),
            )
            self.event_bus.publish(
                run_id,
                Event(
                    type="job_status",
                    payload={
                        "job_id": run_id,
                        "status": "needs_decision",
                        "message": question_text,
                    },
                ),
            )
        else:
            self._job_states[run_id]["status"] = "completed"
            self.record.set_run_status(run_id, "completed", "Placeholder run completed")
            self.event_bus.publish(
                run_id,
                Event(
                    type="job_status",
                    payload={
                        "job_id": run_id,
                        "status": "completed",
                        "message": "Placeholder run completed",
                    },
                ),
            )
        return run_id

    def job_status(self, job_id: str) -> dict[str, Any]:
        if self._real_job_manager is not None and job_id not in self._job_states:
            try:
                st = self._real_job_manager.status(job_id)
                if st:
                    self._log_call("job_status", {"job_id": job_id}, result=st)
                    return dict(st)
            except Exception:  # noqa: BLE001
                pass

        if job_id in self._job_states:
            state = dict(self._job_states[job_id])
            state["messages"] = state["messages"][-10:]
            self._log_call("job_status", {"job_id": job_id}, result=state)
            return state

        run_info = self.record.get_run(job_id)
        if run_info is None:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        out = {
            "job_id": job_id,
            "run_id": run_info["run_id"],
            "plan_id": run_info["plan_id"],
            "status": run_info["status"],
            "message": run_info["message"],
            "messages": [run_info["message"]] if run_info["message"] else [],
            "is_placeholder": bool(run_info["metadata"].get("is_placeholder", False)),
        }
        self._log_call("job_status", {"job_id": job_id}, result=out)
        return out

    def list_jobs(self) -> list[dict[str, Any]]:
        runs = self.record.list_runs()
        items: list[dict[str, Any]] = []
        for r in runs:
            jid = r["run_id"]
            if jid in self._job_states:
                st = self._job_states[jid]
                items.append(
                    {
                        "job_id": jid,
                        "run_id": jid,
                        "plan_id": r["plan_id"],
                        "status": st["status"],
                        "message": st["messages"][-1] if st["messages"] else r["message"],
                        "question": st.get("question"),
                        "is_placeholder": bool(r["metadata"].get("is_placeholder", False)),
                        "created_at": r["created_at"],
                    }
                )
            else:
                items.append(
                    {
                        "job_id": jid,
                        "run_id": jid,
                        "plan_id": r["plan_id"],
                        "status": r["status"],
                        "message": r["message"],
                        "question": None,
                        "is_placeholder": bool(r["metadata"].get("is_placeholder", False)),
                        "created_at": r["created_at"],
                    }
                )
        return items

    def answer_job(self, job_id: str, answer: str) -> dict[str, Any]:
        if self._real_job_manager is not None and job_id not in self._job_states:
            self._real_job_manager.answer(job_id, answer)
            out = {"ok": True, "job_id": job_id, "status": "running"}
            self._log_call("answer_job", {"job_id": job_id, "answer": answer}, result=out)
            return out

        if job_id not in self._job_states and self.record.get_run(job_id) is None:
            raise KeyError(f"Unknown job_id: {job_id!r}")

        if job_id in self._job_states:
            self._job_states[job_id]["answers"].append(answer)
            self._job_states[job_id]["question"] = None
            self._job_states[job_id]["status"] = "completed"
            self._job_states[job_id]["messages"].append(f"Decision received: {answer}")

        self.record.set_run_status(job_id, "completed", f"Completed after user decision: {answer}")
        self.event_bus.publish(
            job_id,
            Event(
                type="job_status",
                payload={"job_id": job_id, "status": "completed", "message": "Decision applied"},
            ),
        )
        out = {"ok": True, "job_id": job_id, "status": "completed"}
        self._log_call(
            "answer_job",
            {"job_id": job_id, "answer": answer},
            result=out,
            run_id=job_id,
        )
        return out

    def stop_job(self, job_id: str) -> dict[str, Any]:
        if self._real_job_manager is not None and job_id not in self._job_states:
            self._real_job_manager.stop(job_id)
            out = {"ok": True, "job_id": job_id, "status": "stopped"}
            self._log_call("stop_job", {"job_id": job_id}, result=out)
            return out

        if job_id not in self._job_states and self.record.get_run(job_id) is None:
            raise KeyError(f"Unknown job_id: {job_id!r}")

        if job_id in self._job_states:
            self._job_states[job_id]["status"] = "stopped"
            self._job_states[job_id]["messages"].append("Stopped by user")

        self.record.set_run_status(job_id, "stopped", "Stopped by user")
        self.event_bus.publish(
            job_id,
            Event(
                type="job_status",
                payload={"job_id": job_id, "status": "stopped", "message": "Stopped by user"},
            ),
        )
        out = {"ok": True, "job_id": job_id, "status": "stopped"}
        self._log_call("stop_job", {"job_id": job_id}, result=out, run_id=job_id)
        return out

    # ------------------------------------------------------------------
    # 4. Results, findings, examples, 3D view events, reports, exports, review
    # ------------------------------------------------------------------

    def get_findings(self, run_id: str) -> list[dict[str, Any]]:
        run_info = self.record.get_run(run_id)
        run_is_placeholder = bool(
            run_info and run_info.get("metadata", {}).get("is_placeholder", False)
        )
        findings = self.record.get_findings(run_id)
        out: list[dict[str, Any]] = []
        for f in findings:
            if run_is_placeholder or f.is_placeholder:
                f = f.model_copy(update={"is_placeholder": True, "verified": False})
            out.append(f.model_dump())
        self._log_call("get_findings", {"run_id": run_id}, result=out, run_id=run_id)
        return out

    def get_results(
        self,
        run_id: str | None = None,
        check: str | None = None,
        metric: str | None = None,
        checkpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.record.query_results(
            run_id=run_id,
            check=check,
            metric=metric,
            checkpoint_id=checkpoint_id,
        )
        out = [r.model_dump() for r in rows]
        self._log_call(
            "get_results",
            {
                "run_id": run_id,
                "check": check,
                "metric": metric,
                "checkpoint_id": checkpoint_id,
            },
            result=out,
            run_id=run_id,
        )
        return out

    def get_examples(self, run_id: str, n: int = 3) -> list[dict[str, Any]]:
        out = self.record.get_examples(run_id, n=n)
        self._log_call("get_examples", {"run_id": run_id, "n": n}, result=out, run_id=run_id)
        return out

    def view_focus(self, component_ids: list[str], reason: str) -> dict[str, Any]:
        ev = Event(
            type="focus",
            payload={"component_ids": list(component_ids), "reason": reason},
        )
        self.event_bus.publish("view", ev)
        out = {"ok": True, "component_ids": list(component_ids)}
        self._log_call(
            "view_focus",
            {"component_ids": component_ids, "reason": reason},
            result=out,
        )
        return out

    def view_paint(
        self,
        run_id: str,
        metric: str,
        checkpoint_id: str | None = None,
        scale: PaintScale = "sequential",
    ) -> dict[str, Any]:
        rows = self.record.query_results(
            run_id=run_id,
            metric=metric,
            checkpoint_id=checkpoint_id,
        )
        values = {r.component_id: float(r.value) for r in rows}
        ev = Event(
            type="paint",
            payload={
                "metric": metric,
                "values": values,
                "scale": scale,
                "checkpoint_id": checkpoint_id,
                "run_id": run_id,
            },
        )
        self.event_bus.publish(run_id, ev)
        out = {"ok": True, "painted_components": len(values)}
        self._log_call(
            "view_paint",
            {"run_id": run_id, "metric": metric, "checkpoint_id": checkpoint_id, "scale": scale},
            result=out,
            run_id=run_id,
        )
        return out

    def view_timeline(self, checkpoint_id: str) -> dict[str, Any]:
        ev = Event(type="timeline", payload={"checkpoint_id": checkpoint_id})
        self.event_bus.publish("view", ev)
        out = {"ok": True, "checkpoint_id": checkpoint_id}
        self._log_call("view_timeline", {"checkpoint_id": checkpoint_id}, result=out)
        return out

    def build_report(self, run_id: str) -> dict[str, Any]:
        report_id = f"report_{safe_filename(run_id)}"
        path = generate_html_report(run_id, self.record, report_id=report_id)
        out = {
            "report_id": report_id,
            "run_id": run_id,
            "path": str(path),
            "url": f"/api/reports/{report_id}",
        }
        self._log_call("build_report", {"run_id": run_id}, result=out, run_id=run_id)
        return out

    def export_artifact(self, run_id: str, kind: str) -> dict[str, Any]:
        valid_kinds = {"probe", "steering", "monitor_config"}
        if kind not in valid_kinds:
            raise ValueError(
                f"Unsupported export kind {kind!r}; expected one of {sorted(valid_kinds)}."
            )

        run_info = self.record.get_run(run_id)
        if run_info is None:
            raise KeyError(f"Unknown run_id: {run_id!r}")

        is_placeholder = bool(run_info["metadata"].get("is_placeholder", False))
        out_path = (
            self.record.layout.exports_dir / f"{safe_filename(run_id)}_{safe_filename(kind)}.json"
        )
        payload = {
            "run_id": run_id,
            "kind": kind,
            "plan_id": run_info["plan_id"],
            "is_placeholder": is_placeholder,
            "dual_use_notice": (
                "Exported interpretability artifact. Steering vectors and probes modify or "
                "monitor model activations; re-run safety and capability checks after applying."
            ),
            "metadata": run_info["metadata"],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out = {
            "run_id": run_id,
            "kind": kind,
            "path": str(out_path),
            "is_placeholder": is_placeholder,
        }
        self._log_call(
            "export_artifact",
            {"run_id": run_id, "kind": kind},
            result=out,
            run_id=run_id,
        )
        return out

    def get_evidence(self, finding_id: str) -> dict[str, Any]:
        finding = self.record.get_finding(finding_id)
        if finding is None:
            raise KeyError(f"Unknown finding_id: {finding_id!r}")

        if not self.stub_mode:
            try:
                from engine.verify.evidence import (  # type: ignore[import-not-found]
                    build_evidence_bundle,
                )

                bundle = build_evidence_bundle(finding, self.record)
                self._log_call(
                    "get_evidence",
                    {"finding_id": finding_id},
                    result=bundle,
                    run_id=finding.run_id,
                )
                return bundle
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass

        run_info = self.record.get_run(finding.run_id) or {}
        plan = (
            self.record.get_plan(run_info["plan_id"], version=run_info.get("plan_version"))
            if "plan_id" in run_info
            else None
        )
        results = self.record.query_results(run_id=finding.run_id)
        examples = self.record.get_examples(finding.run_id, n=3)
        bundle = {
            "finding": finding.model_dump(),
            "settings": run_info.get("metadata", {}),
            "plan": plan.model_dump() if plan is not None else None,
            "results_count": len(results),
            "results": [r.model_dump() for r in results[:50]],
            "examples": examples,
        }
        self._log_call(
            "get_evidence",
            {"finding_id": finding_id},
            result=bundle,
            run_id=finding.run_id,
        )
        return bundle

    def submit_review(
        self,
        finding_id: str,
        verdict: bool | dict[str, Any] | str | None = None,
        notes: str = "",
        corrected_claim: str | None = None,
        lower_strength_to: str | None = None,
        agrees: bool | None = None,
    ) -> dict[str, Any]:
        finding = self.record.get_finding(finding_id)
        if finding is None:
            raise KeyError(f"Unknown finding_id: {finding_id!r}")

        resolved_agrees: bool
        resolved_notes = notes
        resolved_claim = corrected_claim
        resolved_lower = lower_strength_to

        if isinstance(verdict, dict):
            resolved_agrees = bool(
                agrees if agrees is not None else verdict.get("agrees", True)
            )
            if not resolved_notes and "notes" in verdict:
                resolved_notes = str(verdict["notes"])
            if resolved_claim is None and verdict.get("corrected_claim") is not None:
                resolved_claim = str(verdict["corrected_claim"])
            if resolved_lower is None and verdict.get("lower_strength_to") is not None:
                resolved_lower = str(verdict["lower_strength_to"])
        elif isinstance(verdict, bool):
            resolved_agrees = verdict
        elif isinstance(verdict, str):
            resolved_agrees = verdict.strip().lower() in {"agree", "agrees", "true", "ok", "yes"}
        elif agrees is not None:
            resolved_agrees = bool(agrees)
        else:
            resolved_agrees = True

        args = {
            "finding_id": finding_id,
            "agrees": resolved_agrees,
            "notes": resolved_notes,
            "corrected_claim": resolved_claim,
            "lower_strength_to": resolved_lower,
        }

        try:
            final_strength = validate_strength_transition(finding.strength, resolved_lower)
        except ValueError as exc:
            self._log_call(
                "submit_review",
                args,
                run_id=finding.run_id,
                decision="deny",
                reason=str(exc),
            )
            raise

        rv_lower = (
            final_strength
            if (resolved_lower is not None and final_strength in ("suggestive", "inconclusive"))
            else None
        )
        rv = ReviewerVerdict(
            agrees=resolved_agrees,
            notes=resolved_notes,
            corrected_claim=resolved_claim,
            lower_strength_to=rv_lower,  # type: ignore[arg-type]
        )

        updated_finding = finding.model_copy(
            update={
                "strength": final_strength,
                "claim": resolved_claim if resolved_claim else finding.claim,
                "reviewer_verdict": rv.model_dump(),
                "verified": False if finding.is_placeholder else finding.verified,
            }
        )
        self.record.save_finding(updated_finding)
        out = {"ok": True, "finding": updated_finding.model_dump()}
        self._log_call("submit_review", args, result=out, run_id=finding.run_id)
        return out


def register_mcp_tools(mcp: FastMCP, service: EngineService) -> None:
    """Register all 23 engine MCP tools on `mcp`."""

    @mcp.tool()
    def get_compute() -> dict[str, Any]:
        """Return the detected compute profile (device, precision, memory, GPU count)."""
        return service.get_compute()

    @mcp.tool()
    def load_model(
        source: str,
        revision: str | None = None,
        adapter: str | None = None,
        allow_remote_code: bool = False,
    ) -> dict[str, Any]:
        """Load a model safely (safetensors only by default) and build its structural map."""
        return service.load_model(
            source=source,
            revision=revision,
            adapter=adapter,
            allow_remote_code=allow_remote_code,
        )

    @mcp.tool()
    def get_model_map(model_id: str) -> dict[str, Any]:
        """Return the uniform component map and hook paths for a loaded model."""
        return service.get_model_map(model_id=model_id)

    @mcp.tool()
    def confirm_components(model_id: str, mapping: dict[str, str]) -> dict[str, Any]:
        """Apply user-confirmed component kinds for unclassified module paths."""
        return service.confirm_components(model_id=model_id, mapping=mapping)

    @mcp.tool()
    def register_series(
        kind: SeriesKind,
        items: list[dict[str, Any]],
        model_family: str = "unknown",
        series_id: str | None = None,
    ) -> dict[str, Any]:
        """Register an ordered series of pretraining checkpoints or post-training stages."""
        return service.register_series(
            kind=kind,
            items=items,
            model_family=model_family,
            series_id=series_id,
        )

    @mcp.tool()
    def list_checks() -> list[dict[str, Any]]:
        """List all risk checks with maturity labels and short descriptions."""
        return service.list_checks()

    @mcp.tool()
    def describe_check(name: str) -> dict[str, Any]:
        """Return methods, papers, requirements, required controls, and limits for a check."""
        return service.describe_check(name=name)

    @mcp.tool()
    def draft_plan(
        check: str,
        model_id: str,
        reference_model_id: str | None = None,
        series_id: str | None = None,
        checkpoints: list[str] | None = None,
        layers: list[int] | str = "all",
        user_context: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        """Draft an unapproved experiment plan with required controls and coverage filled in."""
        return service.draft_plan(
            check=check,
            model_id=model_id,
            reference_model_id=reference_model_id,
            series_id=series_id,
            checkpoints=checkpoints,
            layers=layers,
            user_context=user_context,
            question=question,
        )

    @mcp.tool()
    def revise_plan(plan_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Create a new unapproved version of a plan without removing required controls."""
        return service.revise_plan(plan_id=plan_id, changes=changes)

    @mcp.tool()
    def run_plan(
        plan_id: str,
        approved_by_user: bool = False,
        confirm_long_run: bool = False,
    ) -> dict[str, Any]:
        """Execute an approved experiment plan as a background job."""
        return service.run_plan(
            plan_id=plan_id,
            approved_by_user=approved_by_user,
            confirm_long_run=confirm_long_run,
        )

    @mcp.tool()
    def job_status(job_id: str) -> dict[str, Any]:
        """Return lifecycle status and recent progress messages for a job."""
        return service.job_status(job_id=job_id)

    @mcp.tool()
    def answer_job(job_id: str, answer: str) -> dict[str, Any]:
        """Provide the user's decision to a job paused in `needs_decision` state."""
        return service.answer_job(job_id=job_id, answer=answer)

    @mcp.tool()
    def stop_job(job_id: str) -> dict[str, Any]:
        """Stop an active or queued experiment job."""
        return service.stop_job(job_id=job_id)

    @mcp.tool()
    def get_findings(run_id: str) -> list[dict[str, Any]]:
        """Return findings recorded for a run with strength, maturity, and verification flags."""
        return service.get_findings(run_id=run_id)

    @mcp.tool()
    def get_results(
        run_id: str | None = None,
        check: str | None = None,
        metric: str | None = None,
        checkpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return aggregate result rows filtered by run, check, metric, or checkpoint."""
        return service.get_results(
            run_id=run_id,
            check=check,
            metric=metric,
            checkpoint_id=checkpoint_id,
        )

    @mcp.tool()
    def get_examples(run_id: str, n: int = 3) -> list[dict[str, Any]]:
        """Return up to `n` example inputs and outputs (harmful text is always redacted)."""
        return service.get_examples(run_id=run_id, n=n)

    @mcp.tool()
    def view_focus(component_ids: list[str], reason: str) -> dict[str, Any]:
        """Emit a live `focus` event moving the 3D view camera to the listed components."""
        return service.view_focus(component_ids=component_ids, reason=reason)

    @mcp.tool()
    def view_paint(
        run_id: str,
        metric: str,
        checkpoint_id: str | None = None,
        scale: PaintScale = "sequential",
    ) -> dict[str, Any]:
        """Emit a live `paint` event coloring model components from stored measurements."""
        return service.view_paint(
            run_id=run_id,
            metric=metric,
            checkpoint_id=checkpoint_id,
            scale=scale,
        )

    @mcp.tool()
    def view_timeline(checkpoint_id: str) -> dict[str, Any]:
        """Emit a live `timeline` event selecting a checkpoint or stage in the 3D view."""
        return service.view_timeline(checkpoint_id=checkpoint_id)

    @mcp.tool()
    def build_report(run_id: str) -> dict[str, Any]:
        """Generate and register an HTML report for `run_id`."""
        return service.build_report(run_id=run_id)

    @mcp.tool()
    def export_artifact(run_id: str, kind: str) -> dict[str, Any]:
        """Export a probe, steering vector, or monitor configuration to `workspace/exports/`."""
        return service.export_artifact(run_id=run_id, kind=kind)

    @mcp.tool()
    def get_evidence(finding_id: str) -> dict[str, Any]:
        """Assemble the evidence bundle read by the independent reviewer subagent."""
        return service.get_evidence(finding_id=finding_id)

    @mcp.tool()
    def submit_review(
        finding_id: str,
        verdict: bool | dict[str, Any] | str | None = None,
        notes: str = "",
        corrected_claim: str | None = None,
        lower_strength_to: str | None = None,
        agrees: bool | None = None,
    ) -> dict[str, Any]:
        """Record the reviewer subagent's verdict (can only keep or lower strength)."""
        return service.submit_review(
            finding_id=finding_id,
            verdict=verdict,
            notes=notes,
            corrected_claim=corrected_claim,
            lower_strength_to=lower_strength_to,
            agrees=agrees,
        )


def create_mcp_server(
    workspace_or_service: Path | str | EngineService,
    *,
    stub_mode: bool = False,
    mounted_at_mcp: bool = False,
) -> FastMCP:
    """Create a FastMCP server exposing all 23 engine tools (for stdio or FastAPI mount)."""
    if isinstance(workspace_or_service, EngineService):
        service = workspace_or_service
    else:
        service = EngineService(Record(workspace_or_service), stub_mode=stub_mode)

    http_path = "/" if mounted_at_mcp else "/mcp"
    mcp = FastMCP(
        "model-safety-engine",
        streamable_http_path=http_path,
        json_response=True,
        stateless_http=True,
    )
    register_mcp_tools(mcp, service)
    return mcp
