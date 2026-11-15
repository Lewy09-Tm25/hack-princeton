"""Conversational ChatSession using Claude Agent SDK or an offline scripted driver (A1)."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

import engine  # noqa: F401  # Enforces Windows mcp-before-nnsight import order
from agent.config import DEFAULT_AGENT_MODEL, AgentSettings, build_agent_options
from agent.hooks import build_pre_tool_use_hooks
from engine.api.mcp_tools import EngineService
from engine.record import Record

ScriptedTurnFn = Callable[
    [str, dict[str, Any] | None, "ChatSession"],
    list[dict[str, Any]],
]


def _format_user_message(user_text: str, context: dict[str, Any] | None) -> str:
    if not context:
        return user_text
    comp_id = context.get("component_id")
    if comp_id:
        return f"[Selected component in 3D view: {comp_id}]\n{user_text}"
    return user_text


class ChatSession:
    """Holds one web-chat conversation connected to the local engine's MCP tools."""

    def __init__(
        self,
        engine_mcp_url: str,
        workspace: Path | str,
        model: str = DEFAULT_AGENT_MODEL,
        *,
        auth_token: str | None = None,
        betas: list[str] | None = None,
        scripted_driver: ScriptedTurnFn | None = None,
        engine_service: EngineService | None = None,
    ) -> None:
        self.engine_mcp_url = engine_mcp_url
        self.workspace = Path(workspace).resolve()
        self.model = model
        self.auth_token = auth_token
        self.scripted_driver = scripted_driver
        self._owns_record = engine_service is None
        self.record: Record = (
            engine_service.record if engine_service is not None else Record(self.workspace)
        )
        self.engine_service: EngineService = engine_service or EngineService(
            self.record, stub_mode=True
        )

        self.settings = AgentSettings(
            engine_mcp_url=engine_mcp_url,
            workspace=self.workspace,
            model=model,
            reviewer_model=model,
            auth_token=auth_token,
            betas=betas,
        )
        # Counts the user's messages so a "yes" in chat can approve a plan drafted earlier.
        self.turn_state: dict[str, int] = {"turn": 0, "last_plan_change_turn": -1}
        self.pre_tool_hooks = build_pre_tool_use_hooks(
            self.workspace, record=self.record, turn_state=self.turn_state
        )
        self.options: ClaudeAgentOptions = build_agent_options(
            self.settings,
            pre_tool_hooks=self.pre_tool_hooks,
        )
        self._sdk_client: ClaudeSDKClient | None = None
        self.state: dict[str, Any] = {}

    async def invoke_hooks_and_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Run `PreToolUse` hooks for `tool_name` and call `EngineService` if allowed."""
        full_name = (
            tool_name if tool_name.startswith("mcp__engine__") else f"mcp__engine__{tool_name}"
        )
        short_name = full_name.removeprefix("mcp__engine__")

        hook_input = {"tool_name": full_name, "tool_input": dict(tool_input)}
        for matcher in self.pre_tool_hooks:
            matches = matcher.matcher is None or bool(
                re.fullmatch(matcher.matcher, full_name) or matcher.matcher == full_name
            )
            if not matches:
                continue
            for hook_fn in matcher.hooks:
                hook_out = await hook_fn(hook_input, None, None)
                spec = hook_out.get("hookSpecificOutput", {}) if isinstance(hook_out, dict) else {}
                if spec.get("permissionDecision") == "deny":
                    return {
                        "denied": True,
                        "reason": str(spec.get("permissionDecisionReason") or "Denied by hook."),
                    }

        method = getattr(self.engine_service, short_name, None)
        if not callable(method):
            raise ValueError(f"Unknown engine tool: {short_name!r}")
        result = method(**tool_input)
        return {"denied": False, "result": result}

    async def send(
        self,
        user_text: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream message parts (`text`, `plan`, `question`, `finding`, `tool_activity`)."""
        self.turn_state["turn"] = self.turn_state.get("turn", 0) + 1
        formatted = _format_user_message(user_text, context)

        if self.scripted_driver is not None:
            actions = self.scripted_driver(formatted, context, self)
            for action in actions:
                kind = action.get("type")
                if kind == "call_tool":
                    tname = str(action["tool_name"])
                    tinput = dict(action.get("tool_input") or {})
                    yield {
                        "type": "tool_activity",
                        "tool": tname,
                        "status": "running",
                    }
                    outcome = await self.invoke_hooks_and_tool(tname, tinput)
                    if outcome["denied"]:
                        yield {
                            "type": "tool_activity",
                            "tool": tname,
                            "status": "denied",
                            "reason": outcome["reason"],
                        }
                        yield {"type": "text", "text": outcome["reason"]}
                    else:
                        res = outcome["result"]
                        self.state[f"last_{tname.removeprefix('mcp__engine__')}"] = res
                        yield {
                            "type": "tool_activity",
                            "tool": tname,
                            "status": "completed",
                            "result": res,
                        }
                        short = tname.removeprefix("mcp__engine__")
                        if short in ("draft_plan", "revise_plan"):
                            plan_obj = (
                                res["plan"]
                                if isinstance(res, dict) and "plan" in res
                                else res
                            )
                            self.state["latest_plan_id"] = plan_obj.get("plan_id")
                            yield {"type": "plan", "plan": plan_obj}
                        elif short == "get_findings" and isinstance(res, list):
                            for finding in res:
                                yield {"type": "finding", "finding": finding}
                else:
                    yield dict(action)
            return

        if not os.environ.get("ANTHROPIC_API_KEY"):
            yield {
                "type": "text",
                "text": (
                    "ANTHROPIC_API_KEY is not set in the environment yet. "
                    "Set ANTHROPIC_API_KEY to use the built-in conversational agent, "
                    "or connect Claude Code directly to the engine's MCP server."
                ),
            }
            return

        if self._sdk_client is None:
            self._sdk_client = ClaudeSDKClient(options=self.options)
            await self._sdk_client.connect()

        await self._sdk_client.query(formatted)
        async for msg in self._sdk_client.receive_response():
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                yield {"type": "text", "text": content}
            elif isinstance(content, list):
                for block in content:
                    text_val = getattr(block, "text", None)
                    if isinstance(text_val, str) and text_val:
                        yield {"type": "text", "text": text_val}
                    elif getattr(block, "name", None):
                        yield {
                            "type": "tool_activity",
                            "tool": str(block.name),
                            "status": "called",
                        }

    async def interrupt(self) -> None:
        """Stop the agent's current reply, as a chat stop button does."""
        if self._sdk_client is not None:
            try:
                await self._sdk_client.interrupt()
            except Exception:  # noqa: BLE001 - nothing to stop, or already finished
                pass

    async def close(self) -> None:
        """Disconnect any live SDK client and close owned resources."""
        if self._sdk_client is not None:
            try:
                await self._sdk_client.disconnect()
            finally:
                self._sdk_client = None
        if self._owns_record:
            self.record.close()
