"""Built-in conversational agent, reviewer subagent, and approval hooks."""

from __future__ import annotations

import engine  # noqa: F401  # Enforces Windows mcp-before-nnsight import order
from agent.config import (
    BUILT_IN_DISALLOWED_TOOLS,
    DEFAULT_AGENT_MODEL,
    AgentSettings,
    build_agent_options,
    build_reviewer_definition,
    load_reviewer_prompt,
    load_system_prompt,
)
from agent.hooks import (
    RUN_PLAN_TOOL_NAME,
    build_pre_tool_use_hooks,
    make_audit_log_hook,
    make_plan_approval_hook,
)
from agent.session import ChatSession

__all__ = [
    "BUILT_IN_DISALLOWED_TOOLS",
    "DEFAULT_AGENT_MODEL",
    "RUN_PLAN_TOOL_NAME",
    "AgentSettings",
    "ChatSession",
    "build_agent_options",
    "build_pre_tool_use_hooks",
    "build_reviewer_definition",
    "load_reviewer_prompt",
    "load_system_prompt",
    "make_audit_log_hook",
    "make_plan_approval_hook",
]
