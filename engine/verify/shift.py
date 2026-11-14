"""Shifted-input invariance tests across paraphrase, multilingual, multi-turn, and format."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.contracts import ControlResult


def shift_test(
    effect_fn: Callable[[str], Any] | None,
    variant_dataset_id: str | None,
    kind: str,
    original_effect: float | None = None,
    variant_control_passed: bool = True,
    not_run_reason: str | None = None,
) -> ControlResult:
    """Run a shift test (`docs/defaults.md` section 3).

    Passes iff the variant effect has the same sign as `original_effect`, has at
    least 50% of its magnitude, and the variant's own control passes.
    If unavailable or skipped, returns `passed=None` and `notes='not run'` (or detailed reason).
    """
    if not_run_reason is not None or effect_fn is None or not variant_dataset_id:
        note = not_run_reason if not_run_reason else "not run"
        if "not run" not in note.lower():
            note = f"not run: {note}"
        return ControlResult(kind=kind, value=None, passed=None, notes=note)

    raw = effect_fn(variant_dataset_id)
    if raw is None:
        return ControlResult(kind=kind, value=None, passed=None, notes="not run")

    ctrl_ok = bool(variant_control_passed)
    orig = original_effect

    if isinstance(raw, ControlResult):
        return raw

    if isinstance(raw, dict):
        if raw.get("not_run"):
            raw_note = str(raw.get("notes") or "not run")
            if "not run" not in raw_note.lower():
                raw_note = f"not run: {raw_note}"
            return ControlResult(kind=kind, value=None, passed=None, notes=raw_note)
        variant_val = float(raw["effect"] if "effect" in raw else raw["value"])
        if "control_passed" in raw:
            ctrl_ok = bool(raw["control_passed"])
        if orig is None and "original_effect" in raw:
            orig = float(raw["original_effect"])
    elif isinstance(raw, tuple | list) and len(raw) == 2:
        variant_val = float(raw[0])
        ctrl_ok = bool(raw[1])
    else:
        variant_val = float(raw)

    if orig is not None:
        orig_val = float(orig)
        same_sign = (variant_val > 0.0 and orig_val > 0.0) or (
            variant_val < 0.0 and orig_val < 0.0
        )
        half_magnitude = abs(variant_val) >= (0.50 * abs(orig_val) - 1e-12)
        passed = bool(same_sign and half_magnitude and ctrl_ok)
        notes = (
            f"variant={variant_val:.4f}, original={orig_val:.4f}, "
            f"control_passed={ctrl_ok}"
        )
    else:
        passed = bool((variant_val > 0.0) and ctrl_ok)
        notes = f"variant={variant_val:.4f}, control_passed={ctrl_ok}"

    return ControlResult(
        kind=kind,
        value=round(variant_val, 6),
        passed=passed,
        notes=notes,
    )
