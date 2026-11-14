"""Described interface for the data-poisoning risk check (`design.md` section 11.6)."""

from __future__ import annotations

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Requirements,
    ShiftTestSpec,
)


class DataPoisoningCheck(DescribedRiskCheck):
    """Traces suspicious behaviors across checkpoints and training documents."""

    name = "data_poisoning"
    summary = (
        "Experimental (partly established): combines an established checkpoint timeline "
        "with experimental training-data attribution whose influence-function estimates are "
        "unreliable on large language models (interface only in this version)."
    )
    maturity = "experimental"
    papers = [
        "Barez et al. 2026, Understanding AI from the Inside (arXiv:2611.08438)",
        "Grosse et al. 2023, Studying large language model generalization with influence functions",
    ]
    methods_summary = [
        "Partly established method: step through pretraining or fine-tuning checkpoints to "
        "locate when a suspicious behavior or representation first appears (established).",
        "Experimental attribution: run influence-function scoring (Kronfluence / dattri) and "
        "exact text-span search over the training corpus to surface candidate documents.",
    ]
    limits_summary = [
        "Partly established overall: while the checkpoint timeline is established, influence-"
        "function attribution has been shown to work unreliably on large language models and "
        "produces leads to inspect rather than proof.",
        "Requires direct access to the user's training corpus for document attribution.",
        "This check is documented as an interface in this version and does not yet execute.",
    ]

    def default_datasets(self) -> list[DatasetRef]:
        return [
            DatasetRef(
                dataset_id="user/training_corpus",
                role="training_corpus",
                description="User-supplied training corpus for span search and attribution.",
            )
        ]

    def required_controls(self) -> list[ControlSpec]:
        return [
            ControlSpec(
                kind="unrelated_behavior_attribution",
                description=(
                    "Run attribution on unrelated behaviors to ensure distinct documents surface."
                ),
            )
        ]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [
            ShiftTestSpec(kind="paraphrase", dataset_id="poisoning/paraphrase"),
        ]

    def needs(self) -> Requirements:
        return Requirements(
            needs_training_data=True,
            needs_reference_model=False,
            notes=(
                "Partly established: without training data only the checkpoint timeline can be "
                "inspected; with training data, attribution results remain experimental leads."
            ),
            follow_up_questions=[
                "Do you have the training corpus available locally, and which checkpoints exist?",
            ],
        )
