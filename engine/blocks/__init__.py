"""Interpretability building blocks: activations, probes, directions, steering, patching."""

from __future__ import annotations

from engine.blocks.activations import collect_activations
from engine.blocks.directions import cosine, diff_of_means, projection, random_direction_like
from engine.blocks.generate import generate
from engine.blocks.patching import patch_component
from engine.blocks.probes import ProbeResult, probe_all_layers, train_probe
from engine.blocks.steering import (
    check_steered_outputs,
    generate_with_steering,
    residual_norm,
    sweep_strengths,
)

__all__ = [
    "ProbeResult",
    "check_steered_outputs",
    "collect_activations",
    "cosine",
    "diff_of_means",
    "generate",
    "generate_with_steering",
    "patch_component",
    "probe_all_layers",
    "projection",
    "random_direction_like",
    "residual_norm",
    "sweep_strengths",
    "train_probe",
]
