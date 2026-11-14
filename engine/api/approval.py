"""Engine-side approval, required-control, strength-lowering, and audit-log rules."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.contracts import Plan, StrengthLevel
from engine.record import Record

DEFAULT_MAX_ESTIMATE_SECONDS = 20.0 * 60.0  # 20 minutes (docs/defaults.md section 10)

_STRENGTH_RANK: dict[str, int] = {
    "inconclusive": 0,
    "suggestive": 1,
    "established": 2,
}

_SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "credential",
)


class PlanApprovalError(PermissionError, ValueError):
    """Raised when an unapproved or unconfirmed plan is submitted for execution."""


def validate_strength_transition(
    current: StrengthLevel,
    requested: str | None,
) -> StrengthLevel:
    """Ensure a review verdict only keeps or lowers finding strength, never raises it."""
    if requested is None:
        return current
    if requested not in _STRENGTH_RANK:
        raise ValueError(
            f"Invalid strength {requested!r}; expected 'established', 'suggestive', "
            "or 'inconclusive'."
        )
    if _STRENGTH_RANK[requested] > _STRENGTH_RANK[current]:
        raise ValueError(
            f"Reviewer cannot raise finding strength from {current!r} to {requested!r}; "
            "strength may only be kept or lowered."
        )
    return requested  # type: ignore[return-value]


def ensure_required_controls_preserved(
    check_name: str,
    required_controls: list[str],
    candidate_controls: list[str],
) -> list[str]:
    """Refuse removing any required control from a plan."""
    missing = [ctrl for ctrl in required_controls if ctrl not in candidate_controls]
    if missing:
        raise ValueError(
            f"Cannot remove required control(s) {missing} for check {check_name!r}."
        )
    return list(dict.fromkeys(candidate_controls))


def ensure_plan_ready_to_run(
    record: Record,
    plan_id: str,
    *,
    approved_by_user: bool = False,
    confirm_long_run: bool = False,
    max_estimate_seconds: float = DEFAULT_MAX_ESTIMATE_SECONDS,
) -> Plan:
    """Verify that the latest version of `plan_id` is approved and within compute budget."""
    plan = record.get_plan(plan_id)
    if plan is None:
        raise KeyError(f"Unknown plan_id: {plan_id}")

    if approved_by_user and not plan.approved:
        plan = record.approve_plan(plan_id)

    if not plan.approved:
        raise PlanApprovalError(
            f"Plan {plan_id!r} (version {plan.version}) has not been approved by the user. "
            "Approve this plan version before running."
        )

    confirmed_in_context = bool(plan.user_context.get("confirm_long_run"))
    if (
        plan.estimate_seconds > max_estimate_seconds
        and not confirm_long_run
        and not confirmed_in_context
    ):
        raise PlanApprovalError(
            f"Plan {plan_id!r} estimates {plan.estimate_seconds:.0f}s, exceeding the "
            f"{max_estimate_seconds:.0f}s limit. Ask the user to confirm this long run first."
        )

    return plan


def sanitize_for_log(value: Any) -> Any:
    """Recursively scrub secret-like dictionary keys before writing audit logs."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if any(marker in key_lower for marker in _SENSITIVE_KEY_MARKERS):
                cleaned[str(k)] = "[REDACTED_SECRET]"
            else:
                cleaned[str(k)] = sanitize_for_log(v)
        return cleaned
    if isinstance(value, list | tuple):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def append_audit_log(
    log_path: Path,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_size: int | None = None,
    decision: str = "allow",
    reason: str | None = None,
) -> None:
    """Append one sanitized tool-call entry to an `agent_log.jsonl` file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool_name,
        "arguments": sanitize_for_log(arguments),
        "decision": decision,
    }
    if result_size is not None:
        entry["result_size"] = result_size
    if reason:
        entry["reason"] = reason

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
