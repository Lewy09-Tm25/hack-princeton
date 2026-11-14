"""Engine API: FastMCP tools, FastAPI HTTP routes, and WebSocket event streaming."""

from __future__ import annotations

from engine.api.approval import (
    PlanApprovalError,
    append_audit_log,
    ensure_plan_ready_to_run,
    ensure_required_controls_preserved,
    validate_strength_transition,
)
from engine.api.mcp_tools import EngineService, LocalEventBus, create_mcp_server
from engine.api.server import DEFAULT_HOST, DEFAULT_PORT, create_app

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EngineService",
    "LocalEventBus",
    "PlanApprovalError",
    "append_audit_log",
    "create_app",
    "create_mcp_server",
    "ensure_plan_ready_to_run",
    "ensure_required_controls_preserved",
    "validate_strength_transition",
]
