"""Described interface for the evaluation-awareness risk check (`design.md` section 11.7)."""

from __future__ import annotations

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Requirements,
    ShiftTestSpec,
)


class EvaluationAwarenessCheck(DescribedRiskCheck):
    """Probes whether a model separates benchmark-style prompts from real deployment inputs."""

    name = "eval_awareness"
    summary = (
        "Experimental check probing whether a model internally separates evaluation-like "
        "prompts from deployment prompts (interface only in this version)."
    )
    maturity = "experimental"
    papers = [
        "Nguyen et al. 2025, Probing and steering evaluation awareness in language models",
    ]
    methods_summary = [
        "Train linear probes on paired evaluation-style and real-use prompts.",
        "Optionally test steering along the probe direction to measure behavioral change.",
    ]
    limits_summary = [
        "Published scaling studies across 0.27B to 70B models show evaluation awareness grows "
        "with scale; small models may show little or no signal.",
        "Published work reports limited success altering behavior via these directions.",
        "A negative result means little and never rules out evaluation-sensitive behavior.",
        "This check is documented as an interface in this version and does not yet execute.",
    ]

    def default_datasets(self) -> list[DatasetRef]:
        return [
            DatasetRef(
                dataset_id="eval_awareness/paired_prompts",
                role="primary",
                description="Paired test-like and deployment-like prompts.",
            )
        ]

    def required_controls(self) -> list[ControlSpec]:
        return [
            ControlSpec(kind="shuffled_labels", description="20 label permutations per layer."),
            ControlSpec(
                kind="surface_style_matched",
                description="Prompts differing only in benign surface formatting.",
            ),
        ]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [
            ShiftTestSpec(kind="paraphrase", dataset_id="eval_awareness/paraphrase"),
        ]

    def needs(self) -> Requirements:
        return Requirements(
            needs_training_data=False,
            needs_reference_model=False,
            recommended_min_params=7_000_000_000,
            notes="Experimental check; strongest published signals appear in large models (70B+).",
            follow_up_questions=[
                "What do real deployment prompts look like in your application?",
            ],
        )
