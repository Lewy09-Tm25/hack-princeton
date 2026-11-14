"""Base `RiskCheck` interface, `Plan` re-export, and plan control validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from engine.contracts import (
    Finding,
    MaturityLevel,
    ModelMap,
    Plan,
    ResultRecord,
    Series,
)


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a dataset registered in the engine dataset registry."""

    dataset_id: str
    role: str = "primary"
    description: str = ""


@dataclass(frozen=True)
class ControlSpec:
    """Required control experiment specification for a risk check."""

    kind: str
    description: str = ""


@dataclass(frozen=True)
class ShiftTestSpec:
    """Required shifted-input test specification for a risk check."""

    kind: str
    dataset_id: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Requirements:
    """Model and data prerequisites for running a risk check."""

    needs_training_data: bool = False
    needs_reference_model: bool = False
    recommended_min_params: int | None = None
    notes: str = ""
    follow_up_questions: list[str] = field(default_factory=list)


class RiskCheck(ABC):
    """Base interface implemented by every risk check module in the tool."""

    name: str = ""
    summary: str = ""
    maturity: MaturityLevel = "established"
    papers: list[str] = []
    implemented: bool = True
    methods_summary: list[str] = []
    limits_summary: list[str] = []

    @abstractmethod
    def default_datasets(self) -> list[DatasetRef]:
        """Return default datasets used when drafting a plan."""

    @abstractmethod
    def required_controls(self) -> list[ControlSpec]:
        """Return controls that cannot be skipped for verified findings."""

    @abstractmethod
    def required_shift_tests(self) -> list[ShiftTestSpec]:
        """Return shifted-input invariance checks required for established strength."""

    @abstractmethod
    def needs(self) -> Requirements:
        """Return data, reference model, and model-size prerequisites."""

    @abstractmethod
    def build_plan(
        self,
        model_map: ModelMap,
        series: Series | None = None,
        user_context: dict[str, Any] | None = None,
        reference_model_id: str | None = None,
    ) -> Plan:
        """Draft an unapproved `Plan` tailored to `model_map`, `series`, and `user_context`."""

    @abstractmethod
    def run(self, plan: Plan, ctx: Any) -> list[ResultRecord]:
        """Execute an approved `Plan` and return measured `ResultRecord` rows."""

    @abstractmethod
    def summarize(
        self,
        results: list[ResultRecord],
        plan: Plan | None = None,
    ) -> list[Finding]:
        """Summarize `ResultRecord` rows into strength-labelled `Finding` objects."""

    @abstractmethod
    def coverage(self, plan: Plan) -> str:
        """Return an explicit statement of layers, datasets, languages, and seeds covered."""

    def not_covered(self, plan: Plan) -> str:
        """Return an explicit statement of what the plan does not examine or prove."""
        return (
            "Measurements cover only the listed layers, datasets, and seeds, and cannot "
            "rule out risks outside that scope."
        )

    def describe(self) -> dict[str, Any]:
        """Return plain-language metadata for `describe_check`."""
        reqs = self.needs()
        return {
            "name": self.name,
            "summary": self.summary,
            "maturity": self.maturity,
            "implemented": self.implemented,
            "methods": list(self.methods_summary),
            "papers": list(self.papers),
            "default_datasets": [asdict(d) for d in self.default_datasets()],
            "required_controls": [asdict(c) for c in self.required_controls()],
            "required_shift_tests": [asdict(s) for s in self.required_shift_tests()],
            "requirements": asdict(reqs),
            "limits": list(self.limits_summary),
            "follow_up_questions": list(reqs.follow_up_questions),
        }


class DescribedRiskCheck(RiskCheck):
    """Base class for risk checks whose interface and documentation are provided without run()."""

    implemented: bool = False

    def build_plan(
        self,
        model_map: ModelMap,
        series: Series | None = None,
        user_context: dict[str, Any] | None = None,
        reference_model_id: str | None = None,
    ) -> Plan:
        ctx_dict = dict(user_context or {})
        layers: list[int] | str = ctx_dict.get("layers", "all")
        plan = Plan(
            check=self.name,
            model_id=model_map.model_id,
            reference_model_id=reference_model_id,
            series_id=series.series_id if series is not None else None,
            checkpoints=[it.checkpoint_id for it in series.items] if series is not None else None,
            datasets=[d.dataset_id for d in self.default_datasets()],
            layers=layers,  # type: ignore[arg-type]
            controls=[c.kind for c in self.required_controls()],
            shift_tests=[s.kind for s in self.required_shift_tests()],
            user_context=ctx_dict,
            estimate_seconds=0.0,
            coverage="Interface-only check; execution is not available in this version.",
            not_covered="Execution is not implemented in this version of the tool.",
            maturity=self.maturity,
            papers=list(self.papers),
            approved=False,
        )
        return plan.model_copy(update={"coverage": self.coverage(plan)})

    def run(self, plan: Plan, ctx: Any) -> list[ResultRecord]:
        raise NotImplementedError(
            f"The {self.name!r} risk check is described only and is not available in this version."
        )

    def summarize(
        self,
        results: list[ResultRecord],
        plan: Plan | None = None,
    ) -> list[Finding]:
        raise NotImplementedError(
            f"The {self.name!r} risk check is described only and is not available in this version."
        )

    def coverage(self, plan: Plan) -> str:
        return (
            f"Check {self.name!r} is documented for planning reference only and is not "
            "available to execute in this version."
        )


def validate_plan(
    check: RiskCheck,
    plan: Plan,
    allow_reduced_controls: bool = False,
) -> Plan:
    """Validate that `plan` keeps all required controls for `check`.

    Refuses plans missing required controls by raising `ValueError` unless
    `allow_reduced_controls=True`, in which case findings are capped at `'suggestive'`
    with an explicit warning recorded in `user_context` and `not_covered`.
    """
    if plan.check != check.name:
        raise ValueError(
            f"Plan check name {plan.check!r} does not match check module {check.name!r}."
        )

    req_controls = [c.kind for c in check.required_controls()]
    present_controls = set(plan.controls)
    missing_controls = [c for c in req_controls if c not in present_controls]

    req_shifts = [s.kind for s in check.required_shift_tests()]
    present_shifts = set(plan.shift_tests)
    missing_shifts = [s for s in req_shifts if s not in present_shifts]

    if missing_controls and not allow_reduced_controls:
        raise ValueError(
            f"Cannot remove required control(s) {missing_controls} from check {check.name!r}."
        )

    warnings: list[str] = list(plan.user_context.get("validation_warnings", []))
    new_ctx = dict(plan.user_context)
    not_cov = plan.not_covered

    if missing_controls and allow_reduced_controls:
        msg = (
            f"Required control(s) {missing_controls} were omitted by explicit user override; "
            "findings are capped at 'suggestive' and cannot be marked verified."
        )
        if msg not in warnings:
            warnings.append(msg)
        new_ctx["max_strength"] = "suggestive"
        new_ctx["reduced_controls"] = True
        if msg not in not_cov:
            not_cov = f"{not_cov.rstrip()} {msg}"

    if missing_shifts:
        msg = (
            f"Shift test(s) {missing_shifts} are skipped and will be recorded as 'not run', "
            "capping findings at 'suggestive'."
        )
        if msg not in warnings:
            warnings.append(msg)
        new_ctx["max_strength"] = "suggestive"

    if warnings:
        new_ctx["validation_warnings"] = warnings

    return plan.model_copy(
        update={
            "user_context": new_ctx,
            "not_covered": not_cov,
        }
    )


__all__ = [
    "ControlSpec",
    "DatasetRef",
    "DescribedRiskCheck",
    "Plan",
    "Requirements",
    "RiskCheck",
    "ShiftTestSpec",
    "validate_plan",
]
