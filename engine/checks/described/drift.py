"""Described interface for the behavior-drift risk check (`design.md` section 11.5)."""

from __future__ import annotations

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Requirements,
    ShiftTestSpec,
)


class BehaviorDriftCheck(DescribedRiskCheck):
    """Compares activations and safety directions across model versions or training stages."""

    name = "drift"
    summary = (
        "Measures internal activation and safety-direction drift between two model versions "
        "(interface only in this version)."
    )
    maturity = "established"
    papers = [
        (
            "Betley et al. 2025, Emergent misalignment: narrow finetuning can produce "
            "broadly misaligned LLMs"
        ),
        "Aranguri and McGrath 2025, Model-diff amplification for surfacing rare changed behaviors",
    ]
    methods_summary = [
        "Compare residual activations on a fixed general, safety, and domain prompt set.",
        "Measure cosine similarity and projection shifts of refusal and sycophancy directions.",
        "Amplify differences between model versions to surface rare changed behaviors.",
    ]
    limits_summary = [
        "Activation distance alone cannot distinguish benign domain adaptation from safety drift.",
        "This check is documented as an interface in this version and does not yet execute.",
    ]

    def default_datasets(self) -> list[DatasetRef]:
        return [
            DatasetRef(
                dataset_id="drift/probe_mix",
                role="primary",
                description="Fixed probe set spanning general, safety, and domain prompts.",
            )
        ]

    def required_controls(self) -> list[ControlSpec]:
        return [
            ControlSpec(
                kind="self_comparison",
                description="Compare a model against itself across seeds/settings to bound noise.",
            ),
            ControlSpec(kind="reference_model", description="Baseline stage before fine-tuning."),
        ]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [
            ShiftTestSpec(kind="paraphrase", dataset_id="drift/paraphrase"),
        ]

    def needs(self) -> Requirements:
        return Requirements(
            needs_training_data=False,
            needs_reference_model=True,
            notes="Requires at least two checkpoints or stages from the same model family.",
            follow_up_questions=[
                "Which two model stages or checkpoints do you want to compare?",
                "What domain data was used between those versions?",
            ],
        )
