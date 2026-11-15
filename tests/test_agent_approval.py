"""Tests for A1 Built-in Agent configuration and A3 PreToolUse approval & logging hooks."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

from agent import (
    BUILT_IN_DISALLOWED_TOOLS,
    DEFAULT_AGENT_MODEL,
    RUN_PLAN_TOOL_NAME,
    AgentSettings,
    ChatSession,
    build_agent_options,
    build_pre_tool_use_hooks,
)
from engine.api import EngineService
from engine.record import Record


def _drive_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Step an in-memory coroutine to completion without creating OS loopback sockets."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise RuntimeError("Coroutine yielded an unexpected OS future in offline unit test.")


def _collect_async_iter(aiter: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    while True:
        try:
            items.append(_drive_sync(aiter.__anext__()))
        except StopAsyncIteration:
            return items


def test_claude_agent_options_construction(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    rec = Record(ws)
    try:
        hooks = build_pre_tool_use_hooks(ws, record=rec)
        settings = AgentSettings(
            engine_mcp_url="http://127.0.0.1:8000/mcp",
            workspace=ws,
            auth_token="local_session_tok",
        )
        opts = build_agent_options(settings, pre_tool_hooks=hooks)

        assert opts.model == DEFAULT_AGENT_MODEL == "claude-sonnet-5-5"
        assert opts.thinking == {"type": "adaptive"}
        assert opts.tools == ["Agent"]
        assert opts.disallowed_tools == BUILT_IN_DISALLOWED_TOOLS
        assert "Bash" in opts.disallowed_tools
        assert "Read" in opts.disallowed_tools
        assert "Write" in opts.disallowed_tools
        assert opts.allowed_tools == ["Agent", "mcp__engine__*"]
        assert opts.betas == []  # No unverified refusal-fallback beta flags by default

        assert isinstance(opts.mcp_servers, dict)
        assert opts.mcp_servers["engine"]["type"] == "http"  # type: ignore[index]
        assert opts.mcp_servers["engine"]["url"] == "http://127.0.0.1:8000/mcp"  # type: ignore[index]

        assert opts.agents is not None
        reviewer = opts.agents["reviewer"]
        assert reviewer.model == "claude-sonnet-5-5"
        assert reviewer.tools == [
            "mcp__engine__get_evidence",
            "mcp__engine__submit_review",
        ]

        assert opts.hooks is not None
        assert "PreToolUse" in opts.hooks
        matchers = [m.matcher for m in opts.hooks["PreToolUse"]]
        assert RUN_PLAN_TOOL_NAME in matchers
    finally:
        rec.close()


def test_pre_tool_use_hook_blocks_unapproved_and_revised_plans_and_logs_calls(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "workspace"
    rec = Record(ws)
    svc = EngineService(rec, stub_mode=True)

    def scripted_driver(
        text: str,
        _ctx: dict[str, Any] | None,
        session: ChatSession,
    ) -> list[dict[str, Any]]:
        if "sycophantic" in text.lower():
            return [
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__load_model",
                    "tool_input": {"source": "Qwen/Qwen2.5-0.5B-Instruct"},
                },
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__describe_check",
                    "tool_input": {"name": "sycophancy"},
                },
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__draft_plan",
                    "tool_input": {
                        "check": "sycophancy",
                        "model_id": "hf:Qwen/Qwen2.5-0.5B-Instruct@main",
                        "layers": [0, 1],
                    },
                },
                {
                    "type": "text",
                    "text": "Here is the drafted plan. Reply to approve running it.",
                },
            ]
        if "try running prematurely" in text.lower():
            return [
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__run_plan",
                    "tool_input": {"plan_id": session.state["latest_plan_id"]},
                }
            ]
        if "revise" in text.lower():
            return [
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__revise_plan",
                    "tool_input": {
                        "plan_id": session.state["latest_plan_id"],
                        "changes": {"layers": [0, 1, 2]},
                    },
                }
            ]
        if "yes, go" in text.lower():
            plan_id = session.state["latest_plan_id"]
            return [
                {
                    "type": "call_tool",
                    "tool_name": "mcp__engine__run_plan",
                    "tool_input": {"plan_id": plan_id},
                },
            ]
        return [{"type": "text", "text": "Unhandled scripted turn"}]

    session = ChatSession(
        engine_mcp_url="http://127.0.0.1:8000/mcp",
        workspace=ws,
        scripted_driver=scripted_driver,
        engine_service=svc,
    )
    try:
        # Turn 1: user asks question -> load_model, describe_check, draft_plan
        turn1 = _collect_async_iter(session.send("Is my fine-tune more sycophantic?"))
        plan_parts = [p for p in turn1 if p["type"] == "plan"]
        assert len(plan_parts) == 1
        plan_id = plan_parts[0]["plan"]["plan_id"]
        assert plan_parts[0]["plan"]["approved"] is False

        # Turn 2: premature run_plan before user approval -> DENIED by PreToolUse hook!
        turn2 = _collect_async_iter(session.send("try running prematurely"))
        denied_parts = [
            p for p in turn2 if p["type"] == "tool_activity" and p.get("status") == "denied"
        ]
        assert len(denied_parts) == 1
        assert "not been approved" in denied_parts[0]["reason"]
        assert rec.list_runs() == []

        # Approve v1, then revise -> resets approved=False, so run_plan MUST be denied again!
        rec.approve_plan(plan_id)
        turn3 = _collect_async_iter(session.send("revise layers"))
        revised_plan = [p for p in turn3 if p["type"] == "plan"][0]["plan"]
        assert revised_plan["version"] == 2
        assert revised_plan["approved"] is False

        turn4 = _collect_async_iter(session.send("try running prematurely"))
        denied_v2 = [
            p for p in turn4 if p["type"] == "tool_activity" and p.get("status") == "denied"
        ]
        assert len(denied_v2) == 1
        assert rec.list_runs() == []

        # User approves v2 -> run_plan succeeds and findings can be fetched!
        rec.approve_plan(plan_id)
        turn5 = _collect_async_iter(session.send("yes, go"))
        completed_parts = [
            p for p in turn5 if p["type"] == "tool_activity" and p.get("status") == "completed"
        ]
        assert len(completed_parts) == 1
        run_id = completed_parts[0]["result"]["run_id"]
        findings_out = _drive_sync(
            session.invoke_hooks_and_tool("mcp__engine__get_findings", {"run_id": run_id})
        )
        assert findings_out["denied"] is False
        assert findings_out["result"][0]["is_placeholder"] is True
        assert findings_out["result"][0]["verified"] is False

        # Verify secret scrubbing in audit log
        secret_sentinel = "DO_NOT_LOG_THIS_TOKEN_VALUE"
        _drive_sync(session.invoke_hooks_and_tool("mcp__engine__get_compute", {}))
        log_hook = session.pre_tool_hooks[1].hooks[0]
        _drive_sync(
            log_hook(
                {
                    "tool_name": "mcp__engine__custom_test",
                    "tool_input": {"api_key": secret_sentinel, "note": "safe_value"},
                },
                None,
                None,
            )
        )

        log_file = ws / "agent_log.jsonl"
        assert log_file.is_file()
        raw_log = log_file.read_text(encoding="utf-8")
        assert secret_sentinel not in raw_log
        assert "[REDACTED_SECRET]" in raw_log

        entries = [json.loads(line) for line in raw_log.strip().splitlines()]
        logged_tools = [e["tool"] for e in entries]
        assert "mcp__engine__load_model" in logged_tools
        assert "mcp__engine__describe_check" in logged_tools
        assert "mcp__engine__draft_plan" in logged_tools
        assert "mcp__engine__revise_plan" in logged_tools
        assert "mcp__engine__run_plan" in logged_tools
        assert "mcp__engine__get_findings" in logged_tools

        decisions_for_run = [
            e["decision"] for e in entries if e["tool"] == "mcp__engine__run_plan"
        ]
        assert "deny" in decisions_for_run
        assert "allow" in decisions_for_run
    finally:
        _drive_sync(session.close())
        rec.close()
