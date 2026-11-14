"""Risk check modules and check registry."""

from __future__ import annotations

from engine.checks.backdoor import BackdoorCheck
from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Plan,
    Requirements,
    RiskCheck,
    ShiftTestSpec,
    validate_plan,
)
from engine.checks.refusal import RefusalCheck, is_refusal
from engine.checks.registry import describe_check, get_check, list_checks
from engine.checks.sycophancy import SycophancyCheck

__all__ = [
    "BackdoorCheck",
    "ControlSpec",
    "DatasetRef",
    "DescribedRiskCheck",
    "Plan",
    "RefusalCheck",
    "Requirements",
    "RiskCheck",
    "ShiftTestSpec",
    "SycophancyCheck",
    "describe_check",
    "get_check",
    "is_refusal",
    "list_checks",
    "validate_plan",
]
