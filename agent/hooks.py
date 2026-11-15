"""Agent SDK PreToolUse approval guard and tool-call audit logging (component A3)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import HookMatcher

import engine  # noqa: F401  # Enforces Windows mcp-before-nnsight import order
from engine.api.approval import DEFAULT_MAX_ESTIMATE_SECONDS, append_audit_log
from engine.record import Record

HookCallback = Callable[[dict[str, Any], str | None, Any], Awaitable[dict[str, Any]]]

RUN_PLAN_TOOL_NAME = "mcp__engine__run_plan"


def _resolve_log_paths(workspace: Path, tool_input: dict[str, Any]) -> list[Path]:
    paths = [workspace / "agent_log.jsonl"]
    run_id = tool_input.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        run_folder = workspace / "runs" / run_id
        if run_folder.is_dir():
            paths.append(run_folder / "agent_log.jsonl")
    return paths


def make_plan_approval_hook(
    workspace: Path,
    *,
    record: Record | None = None,
    max_estimate_seconds: float = DEFAULT_MAX_ESTIMATE_SECONDS,
) -> HookCallback:
    """Create a `PreToolUse` hook blocking `mcp__engine__run_plan` unless approved."""

    def _get_record() -> Record:
        return record if record is not None else Record(workspace)

    async def check_plan_approved(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        tool_name = str(input_data.get("tool_name") or RUN_PLAN_TOOL_NAME)
        tool_input = dict(input_data.get("tool_input") or {})
        plan_id = tool_input.get("plan_id")

        rec = _get_record()
        try:
            plan = rec.get_plan(str(plan_id)) if plan_id else None
        finally:
            if record is None:
                rec.close()

        if plan is None:
            reason = (
                f"Cannot run unknown plan {plan_id!r}. Draft a plan with draft_plan and ask the "
                "user for approval first."
            )
            for log_path in _resolve_log_paths(workspace, tool_input):
                append_audit_log(
                    log_path,
                    tool_name=tool_name,
                    arguments=tool_input,
                    decision="deny",
                    reason=reason,
                )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        if not plan.approved:
            reason = (
                f"Plan {plan.plan_id!r} (version {plan.version}) has not been approved by the "
                "user yet. Ask the user for approval first."
            )
            for log_path in _resolve_log_paths(workspace, tool_input):
                append_audit_log(
                    log_path,
                    tool_name=tool_name,
                    arguments=tool_input,
                    decision="deny",
                    reason=reason,
                )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        confirmed_long = bool(
            tool_input.get("confirm_long_run") or plan.user_context.get("confirm_long_run")
        )
        if plan.estimate_seconds > max_estimate_seconds and not confirmed_long:
            reason = (
                f"Plan {plan.plan_id!r} estimates {plan.estimate_seconds:.0f}s, exceeding "
                f"{max_estimate_seconds:.0f}s. Ask the user to confirm this long run first."
            )
            for log_path in _resolve_log_paths(workspace, tool_input):
                append_audit_log(
                    log_path,
                    tool_name=tool_name,
                    arguments=tool_input,
                    decision="deny",
                    reason=reason,
                )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        for log_path in _resolve_log_paths(workspace, tool_input):
            append_audit_log(
                log_path,
                tool_name=tool_name,
                arguments=tool_input,
                decision="allow",
            )
        return {}

    return check_plan_approved


def make_audit_log_hook(workspace: Path) -> HookCallback:
    """Create a `PreToolUse` hook logging every non-`run_plan` tool call without secrets."""

    async def log_tool_use(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        tool_name = str(input_data.get("tool_name") or "unknown_tool")
        if tool_name == RUN_PLAN_TOOL_NAME:
            # Logged with explicit allow/deny decision by `make_plan_approval_hook`.
            return {}
        tool_input = dict(input_data.get("tool_input") or {})
        for log_path in _resolve_log_paths(workspace, tool_input):
            append_audit_log(
                log_path,
                tool_name=tool_name,
                arguments=tool_input,
                decision="allow",
            )
        return {}

    return log_tool_use


def build_pre_tool_use_hooks(
    workspace: Path,
    *,
    record: Record | None = None,
    max_estimate_seconds: float = DEFAULT_MAX_ESTIMATE_SECONDS,
) -> list[HookMatcher]:
    """Return `HookMatcher` list enforcing `run_plan` approval and logging every tool call."""
    async def only_reviewer_subagent(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        # The `Agent` tool exists so the reviewer can run. Any other subagent would give the
        # built-in agent general tools (reading files and so on) and slow every answer down.
        tool_input = dict(input_data.get("tool_input") or {})
        if str(tool_input.get("subagent_type") or "") == "reviewer":
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Only the reviewer subagent may be started. Use the engine tools directly."
                ),
            }
        }

    return [
        HookMatcher(
            matcher="Agent",
            hooks=[only_reviewer_subagent],
        ),
        HookMatcher(
            matcher=RUN_PLAN_TOOL_NAME,
            hooks=[
                make_plan_approval_hook(
                    workspace,
                    record=record,
                    max_estimate_seconds=max_estimate_seconds,
                )
            ],
        ),
        HookMatcher(
            matcher=None,
            hooks=[make_audit_log_hook(workspace)],
        ),
    ]
