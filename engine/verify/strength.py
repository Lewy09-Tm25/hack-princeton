"""Strength decision rules and verified finding construction (`docs/defaults.md` section 4)."""

from __future__ import annotations

from engine.contracts import ControlResult, Finding, MaturityLevel, StrengthLevel


def check_seed_stability(
    seed0_value: float,
    seed1_value: float,
    max_abs_diff: float = 0.05,
) -> bool:
    """Return True if seed-1 re-run lies within `max_abs_diff` (0.05) of seed-0."""
    return abs(float(seed0_value) - float(seed1_value)) <= float(max_abs_diff) + 1e-12


def decide_strength(
    controls: list[ControlResult],
    shift_tests: list[ControlResult],
    required_controls: list[str],
    required_shifts: list[str],
    seed_stable: bool = True,
    effect_above_threshold: bool = True,
) -> StrengthLevel:
    """Compute finding strength strictly from control, shift, and seed-stability outcomes.

    Raises `ValueError` if any required control is missing or has `passed=None`.
    """
    by_control: dict[str, ControlResult] = {c.kind: c for c in controls}
    for req_ctrl in required_controls:
        ctrl = by_control.get(req_ctrl)
        if ctrl is None or ctrl.passed is None:
            raise ValueError(
                f"Cannot verify finding: required control {req_ctrl!r} did not run."
            )

    if not effect_above_threshold:
        return "inconclusive"

    for ctrl in controls:
        if ctrl.passed is False:
            return "inconclusive"

    by_shift: dict[str, ControlResult] = {s.kind: s for s in shift_tests}
    for req_shift in required_shifts:
        sh = by_shift.get(req_shift)
        if sh is None or sh.passed is not True:
            return "suggestive"

    for sh in shift_tests:
        if sh.passed is not True:
            return "suggestive"

    if not seed_stable:
        return "suggestive"

    return "established"


def finalize_finding(
    *,
    finding_id: str,
    run_id: str,
    check: str,
    claim: str,
    maturity: MaturityLevel,
    evidence: list[str],
    controls: list[ControlResult],
    shift_tests: list[ControlResult],
    required_controls: list[str],
    required_shifts: list[str],
    coverage: str,
    not_shown: str,
    is_placeholder: bool,
    seed_stable: bool = True,
    effect_above_threshold: bool = True,
    quantized: bool = False,
) -> Finding:
    """Validate all verification requirements and construct a `Finding`."""
    if not coverage.strip():
        raise ValueError("Finding coverage statement must not be empty.")
    if not not_shown.strip():
        raise ValueError("Finding not_shown statement must not be empty.")

    recorded_shifts = {s.kind for s in shift_tests}
    for req_shift in required_shifts:
        if req_shift not in recorded_shifts:
            raise ValueError(
                f"Required shift test {req_shift!r} is missing from shift_tests; "
                "record unavailable shift tests explicitly with passed=None, notes='not run'."
            )

    strength = decide_strength(
        controls=controls,
        shift_tests=shift_tests,
        required_controls=required_controls,
        required_shifts=required_shifts,
        seed_stable=seed_stable,
        effect_above_threshold=effect_above_threshold,
    )

    notes_extra: list[str] = []
    low_not_shown = not_shown.lower()
    if not seed_stable and "unstable across seeds" not in low_not_shown:
        notes_extra.append("Result was unstable across seeds.")
    if quantized and "approximate (quantized)" not in low_not_shown:
        notes_extra.append("Measurements are approximate (quantized).")
    if maturity == "experimental" and "not well established" not in low_not_shown:
        notes_extra.append(
            "Link between this experimental method and real-world risk is not well established."
        )
    if is_placeholder and "placeholder" not in low_not_shown:
        notes_extra.append("Produced on synthetic placeholder model or dataset (not verified).")

    final_not_shown = not_shown.strip()
    if notes_extra:
        final_not_shown = f"{final_not_shown} {' '.join(notes_extra)}"

    verified = not bool(is_placeholder)

    return Finding(
        finding_id=finding_id,
        run_id=run_id,
        check=check,
        claim=claim.strip(),
        strength=strength,
        maturity=maturity,
        evidence=evidence,
        controls=controls,
        shift_tests=shift_tests,
        coverage=coverage.strip(),
        not_shown=final_not_shown,
        verified=verified,
        is_placeholder=bool(is_placeholder),
    )
