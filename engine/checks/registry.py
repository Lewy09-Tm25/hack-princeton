"""Discovery and lookup of built-in and plug-in risk checks."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from engine.checks.base import RiskCheck

_BUILTIN_SPECS: tuple[tuple[str, str], ...] = (
    ("sycophancy", "engine.checks.sycophancy:SycophancyCheck"),
    ("refusal", "engine.checks.refusal:RefusalCheck"),
    ("backdoor", "engine.checks.backdoor:BackdoorCheck"),
    ("hidden_bias", "engine.checks.described.hidden_bias:HiddenBiasCheck"),
    ("drift", "engine.checks.described.drift:BehaviorDriftCheck"),
    ("data_poisoning", "engine.checks.described.data_poisoning:DataPoisoningCheck"),
    ("eval_awareness", "engine.checks.described.eval_awareness:EvaluationAwarenessCheck"),
    ("alignment_faking", "engine.checks.described.alignment_faking:AlignmentFakingCheck"),
)


def _load_class(target: str) -> type[RiskCheck] | None:
    mod_name, _, attr_name = target.partition(":")
    if not mod_name or not attr_name:
        return None
    try:
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, attr_name)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
    if isinstance(cls, type) and issubclass(cls, RiskCheck):
        return cls
    return None


def _discover_classes() -> dict[str, type[RiskCheck]]:
    found: dict[str, type[RiskCheck]] = {}
    for name, target in _BUILTIN_SPECS:
        cls = _load_class(target)
        if cls is not None:
            found[name] = cls

    try:
        eps = entry_points(group="tool.checks")
    except TypeError:
        eps = entry_points().get("tool.checks", ())  # type: ignore[assignment]

    for ep in eps:
        if ep.name in found:
            continue
        try:
            loaded_ep = ep.load()
        except (ImportError, ModuleNotFoundError, AttributeError):
            continue
        if isinstance(loaded_ep, type) and issubclass(loaded_ep, RiskCheck):
            found[ep.name] = loaded_ep

    return found


def list_checks() -> list[dict[str, Any]]:
    """Return metadata summaries for all available and described risk checks."""
    classes = _discover_classes()
    items: list[dict[str, Any]] = []
    for name, cls in classes.items():
        inst = cls()
        items.append(
            {
                "name": inst.name or name,
                "maturity": inst.maturity,
                "implemented": bool(inst.implemented),
                "summary": inst.summary,
            }
        )
    return items


def get_check(name: str) -> RiskCheck:
    """Instantiate a registered `RiskCheck` by name."""
    classes = _discover_classes()
    if name not in classes:
        raise KeyError(
            f"Unknown or unavailable risk check {name!r}. "
            f"Available checks: {sorted(classes.keys())}"
        )
    return classes[name]()


def describe_check(name: str) -> dict[str, Any]:
    """Return full plain-language documentation and requirements for `name`."""
    return get_check(name).describe()
