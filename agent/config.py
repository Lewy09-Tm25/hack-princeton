"""Configuration builder for the built-in Claude Agent SDK session (component A1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, HookMatcher

import engine  # noqa: F401  # Enforces Windows mcp-before-nnsight import order

DEFAULT_AGENT_MODEL = "claude-sonnet-5-5"

BUILT_IN_DISALLOWED_TOOLS: list[str] = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "NotebookEdit",
]

REVIEWER_ALLOWED_TOOLS: tuple[str, str] = (
    "mcp__engine__get_evidence",
    "mcp__engine__submit_review",
)

REVIEWER_DISALLOWED_TOOLS: list[str] = [
    *BUILT_IN_DISALLOWED_TOOLS,
    "Agent",
]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_system_prompt() -> str:
    """Read `agent/prompts/system.md`."""
    return (_PROMPTS_DIR / "system.md").read_text(encoding="utf-8")


def load_reviewer_prompt() -> str:
    """Read `agent/prompts/reviewer.md`."""
    return (_PROMPTS_DIR / "reviewer.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class AgentSettings:
    """Settings used to build `ClaudeAgentOptions` for the built-in agent."""

    engine_mcp_url: str
    workspace: Path
    model: str = DEFAULT_AGENT_MODEL
    reviewer_model: str = DEFAULT_AGENT_MODEL
    auth_token: str | None = None
    betas: list[str] | None = None


def build_reviewer_definition(model: str = DEFAULT_AGENT_MODEL) -> AgentDefinition:
    """Build the `AgentDefinition` for the independent reviewer subagent (A2)."""
    return AgentDefinition(
        description=(
            "Checks each finding's claim wording and strength against its engine evidence bundle."
        ),
        prompt=load_reviewer_prompt(),
        tools=list(REVIEWER_ALLOWED_TOOLS),
        disallowedTools=list(REVIEWER_DISALLOWED_TOOLS),
        model=model,
    )


def build_agent_options(
    settings: AgentSettings,
    *,
    pre_tool_hooks: list[HookMatcher] | None = None,
) -> ClaudeAgentOptions:
    """Construct `ClaudeAgentOptions` restricted strictly to engine MCP tools and reviewer."""
    mcp_entry: dict[str, Any] = {
        "type": "http",
        "url": settings.engine_mcp_url,
    }
    if settings.auth_token:
        mcp_entry["headers"] = {"Authorization": f"Bearer {settings.auth_token}"}

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "thinking": {"type": "adaptive"},
        "system_prompt": load_system_prompt(),
        "tools": ["Agent"],
        "disallowed_tools": list(BUILT_IN_DISALLOWED_TOOLS),
        "allowed_tools": ["Agent", "mcp__engine__*"],
        "mcp_servers": {"engine": mcp_entry},
        "agents": {"reviewer": build_reviewer_definition(model=settings.reviewer_model)},
        "cwd": str(settings.workspace),
    }
    if pre_tool_hooks is not None:
        kwargs["hooks"] = {"PreToolUse": pre_tool_hooks}
    if settings.betas:
        kwargs["betas"] = list(settings.betas)

    return ClaudeAgentOptions(**kwargs)
