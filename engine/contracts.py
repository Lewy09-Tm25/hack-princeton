"""Shared data formats used across the engine, agents, and web client."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComponentKind = Literal[
    "embedding",
    "layer",
    "attention",
    "attention_head",
    "mlp",
    "norm",
    "unembedding",
    "other",
]

SeriesKind = Literal["pretraining", "post_training"]

SplitKind = Literal["train", "held_out"]

StrengthLevel = Literal["established", "suggestive", "inconclusive"]

MaturityLevel = Literal["established", "experimental"]

JobStatus = Literal[
    "queued",
    "running",
    "needs_decision",
    "completed",
    "stopped",
    "failed",
]

PaintScale = Literal["sequential", "diverging"]

EventType = Literal[
    "focus",
    "paint",
    "timeline",
    "progress",
    "question",
    "finding",
    "job_status",
]


class Component(BaseModel):
    """One addressable part of a language model."""

    id: str
    kind: ComponentKind
    index: int | None = None
    head: int | None = None
    n_heads: int | None = None
    module: str | None = None
    children: list[str] = Field(default_factory=list)


class ModelMap(BaseModel):
    """Uniform description of a language model's architecture and hook paths."""

    model_id: str
    revision: str | None = None
    arch_family: str | None = None
    n_layers: int = Field(ge=0)
    hidden_size: int | None = None
    confirmed_by_user: bool
    needs_confirmation: list[str] = Field(default_factory=list)
    components: list[Component]


class SeriesItem(BaseModel):
    """One checkpoint or post-training stage in an ordered series."""

    checkpoint_id: str
    label: str
    order: int
    source: str


class Series(BaseModel):
    """Ordered sequence of pretraining checkpoints or post-training stages."""

    series_id: str
    kind: SeriesKind
    model_family: str
    items: list[SeriesItem]


class Plan(BaseModel):
    """Experiment plan drafted by the engine and approved before running."""

    plan_id: str | None = None
    experiment_id: str | None = None
    version: int = 1
    check: str
    model_id: str
    reference_model_id: str | None = None
    series_id: str | None = None
    checkpoints: list[str] | None = None
    datasets: list[str]
    layers: list[int] | Literal["all"] = "all"
    controls: list[str]
    shift_tests: list[str]
    user_context: dict[str, Any] = Field(default_factory=dict)
    estimate_seconds: float
    coverage: str
    not_covered: str
    maturity: MaturityLevel
    papers: list[str]
    approved: bool = False
    is_placeholder: bool = False


class ControlResult(BaseModel):
    """Outcome of a control or shifted-input test."""

    kind: str
    value: float | None = None
    passed: bool | None = None
    notes: str | None = None


class ResultRecord(BaseModel):
    """Single measurement recorded for a model component and optional checkpoint."""

    run_id: str
    check: str
    component_id: str
    checkpoint_id: str | None = None
    metric: str
    value: float
    control: ControlResult | None = None
    dataset_id: str
    split: SplitKind
    seed: int
    device: str
    precision: str
    quantized: bool
    is_placeholder: bool = False


class ReviewerVerdict(BaseModel):
    """Verdict submitted by the reviewer subagent after inspecting evidence."""

    model_config = ConfigDict(extra="allow")

    agrees: bool
    notes: str = ""
    corrected_claim: str | None = None
    lower_strength_to: Literal["suggestive", "inconclusive"] | None = None


class Finding(BaseModel):
    """Verified or draft finding produced by a risk check."""

    finding_id: str
    run_id: str
    check: str
    claim: str
    strength: StrengthLevel
    maturity: MaturityLevel
    evidence: list[str]
    controls: list[ControlResult]
    shift_tests: list[ControlResult]
    coverage: str
    not_shown: str
    verified: bool = False
    is_placeholder: bool = False
    reviewer_verdict: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_reviewer_verdict(self) -> Finding:
        if self.is_placeholder and self.verified:
            raise ValueError("Placeholder findings cannot be marked verified.")
        if self.reviewer_verdict is not None:
            ReviewerVerdict.model_validate(self.reviewer_verdict)
        return self


class FocusPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    component_ids: list[str]
    reason: str


class PaintPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric: str
    values: dict[str, float]
    scale: PaintScale
    checkpoint_id: str | None = None
    run_id: str | None = None


class TimelinePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    checkpoint_id: str


class ProgressPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    step: int = Field(ge=0)
    total: int = Field(ge=1)
    message: str


class QuestionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    text: str


class FindingEventPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    finding_id: str


class JobStatusPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    status: JobStatus
    message: str | None = None


_EVENT_PAYLOAD_MODELS: dict[EventType, type[BaseModel]] = {
    "focus": FocusPayload,
    "paint": PaintPayload,
    "timeline": TimelinePayload,
    "progress": ProgressPayload,
    "question": QuestionPayload,
    "finding": FindingEventPayload,
    "job_status": JobStatusPayload,
}


class Event(BaseModel):
    """Live event sent from the engine to connected clients."""

    type: EventType
    payload: dict[str, Any]

    @model_validator(mode="after")
    def _validate_payload(self) -> Event:
        payload_model = _EVENT_PAYLOAD_MODELS[self.type]
        payload_model.model_validate(self.payload)
        return self
