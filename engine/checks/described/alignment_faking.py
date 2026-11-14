"""Described interface for the alignment-faking risk check (`design.md` section 11.8)."""

from __future__ import annotations

from engine.checks.base import (
    ControlSpec,
    DatasetRef,
    DescribedRiskCheck,
    Requirements,
    ShiftTestSpec,
)


class AlignmentFakingCheck(DescribedRiskCheck):
    """Compares behavior and activations across monitored-training vs unmonitored framings."""

    name = "alignment_faking"
    summary = (
        "Experimental check comparing compliance and internal representations under "
        "monitored-training versus unmonitored framings (interface only in this version)."
    )
    maturity = "experimental"
    papers = [
        "Greenblatt et al. 2024, Alignment faking in large language models",
    ]
    methods_summary = [
        "Compare behavioral compliance rates between 'being trained' and 'unmonitored' framings.",
        "Train linear probes to test whether internal activations separate the two framings.",
    ]
    limits_summary = [
        "Clean alignment faking in published studies appeared mainly in frontier-scale models; "
        "open-weight stand-ins below 70B show weak or brittle signals.",
        "Independent replications warn that prompts naming alignment faking can cue the behavior.",
        "A negative result means little on smaller models.",
        "This check is documented as an interface in this version and does not yet execute.",
    ]

    def default_datasets(self) -> list[DatasetRef]:
        return [
            DatasetRef(
                dataset_id="alignment_faking/framing_pairs",
                role="primary",
                description="Paired monitored-training and unmonitored context framings.",
            )
        ]

    def required_controls(self) -> list[ControlSpec]:
        return [
            ControlSpec(kind="shuffled_labels", description="20 label permutations per layer."),
            ControlSpec(
                kind="neutral_framing",
                description="Framing length/style control without monitoring cues.",
            ),
        ]

    def required_shift_tests(self) -> list[ShiftTestSpec]:
        return [
            ShiftTestSpec(kind="paraphrase", dataset_id="alignment_faking/paraphrase"),
        ]

    def needs(self) -> Requirements:
        return Requirements(
            needs_training_data=False,
            needs_reference_model=False,
            recommended_min_params=70_000_000_000,
            notes="Experimental check; published open-weight replications typically require 70B+.",
            follow_up_questions=[
                "Does your deployment prompt include explicit monitoring or training metadata?",
            ],
        )
