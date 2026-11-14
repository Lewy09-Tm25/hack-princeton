"""Described interface for the hidden-bias risk check (`design.md` section 11.4)."""

from __future__ import annotations

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Requirements,
    ShiftTestSpec,
)


class HiddenBiasCheck(DescribedRiskCheck):
    """Probes for unacknowledged user attributes and tests causal use via steering/ablation."""

    name = "hidden_bias"
    summary = (
        "Tests whether the model represents an unstated user attribute internally and uses it "
        "to change answers (interface only in this version)."
    )
    maturity = "established"
    papers = [
        (
            "Cywinski et al. 2025, Towards eliciting latent knowledge from LLMs with "
            "mechanistic interpretability"
        ),
    ]
    methods_summary = [
        "Train per-layer linear probes for an indirectly cued demographic or user attribute.",
        "Steer or ablate along the attribute direction to test causal effect on answers.",
        "Compare behavioral differences against neutral prompts and a base reference model.",
    ]
    limits_summary = [
        "A linear probe finding an attribute does not by itself prove the model uses it causally.",
        "This check is documented as an interface in this version and does not yet execute.",
    ]

    def default_datasets(self) -> list[DatasetRef]:
        return [
            DatasetRef(
                dataset_id="bias/indirect_attribute",
                role="primary",
                description=(
                    "Prompts indirectly revealing a user attribute paired with neutral prompts."
                ),
            )
        ]

    def required_controls(self) -> list[ControlSpec]:
        return [
            ControlSpec(kind="shuffled_labels", description="20 label permutations per layer."),
            ControlSpec(kind="random_direction", description="10 equal-norm random vectors."),
            ControlSpec(kind="reference_model", description="Same procedure on the base model."),
        ]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [
            ShiftTestSpec(kind="paraphrase", dataset_id="bias/paraphrase"),
            ShiftTestSpec(kind="topic_shift", dataset_id="bias/topics"),
        ]

    def needs(self) -> Requirements:
        return Requirements(
            needs_training_data=False,
            needs_reference_model=True,
            notes="Requires prompts covering the suspect attribute in the user's domain.",
            follow_up_questions=[
                "Which user attribute or demographic signal do you want to test for?",
                "Do you have the original base model available as a reference?",
            ],
        )
