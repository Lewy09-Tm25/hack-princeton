"""Interface-only risk checks documented for transparency (`design.md` sections 11.4–11.8)."""

from __future__ import annotations

from engine.checks.described.alignment_faking import AlignmentFakingCheck
from engine.checks.described.data_poisoning import DataPoisoningCheck
from engine.checks.described.drift import BehaviorDriftCheck
from engine.checks.described.eval_awareness import EvaluationAwarenessCheck
from engine.checks.described.hidden_bias import HiddenBiasCheck

__all__ = [
    "AlignmentFakingCheck",
    "BehaviorDriftCheck",
    "DataPoisoningCheck",
    "EvaluationAwarenessCheck",
    "HiddenBiasCheck",
]
