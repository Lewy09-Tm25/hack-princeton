"""Tests for E3 compute manager: CPU, busy single GPU, and multi-GPU planning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from engine.compute import (
    ComputeProfile,
    batch_size_for,
    detect,
    estimate_seconds,
    plan_for_model,
    precision_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3


def test_detect_cpu_only() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(torch.backends, "mps", create=True) as mock_mps,
    ):
        mock_mps.is_available.return_value = False
        profile = detect()

    assert profile.device == "cpu"
    assert profile.dtype == torch.float32
    assert profile.n_gpus == 0
    assert precision_name(profile.dtype) == "fp32"


def test_detect_and_plan_busy_single_gpu() -> None:
    # 16 GiB total GPU, but busy so only 1.5 GiB is free.
    free_b = int(1.5 * GIB)
    total_b = 16 * GIB
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(free_b, total_b)),
        patch("torch.cuda.is_bf16_supported", return_value=True),
    ):
        profile = detect()
        assert profile.device == "cuda"
        assert profile.dtype == torch.bfloat16
        assert profile.free_memory_bytes == free_b
        assert profile.total_memory_bytes == total_b
        assert profile.n_gpus == 1

        # 500M params in bf16 -> 1.0B bytes * 1.2 headroom = 1.2B bytes (< 1.5 GiB) -> fits natively
        small_plan = plan_for_model(500_000_000, profile)
        assert small_plan.fits is True
        assert small_plan.quantize is None
        assert small_plan.device_map is None
        assert small_plan.warnings == []

        # 1B params in bf16 -> 2.4B bytes (> 1.5 GiB), 8bit -> 1.2B bytes (< 1.5 GiB) -> 8bit
        med_plan = plan_for_model(1_000_000_000, profile)
        assert med_plan.fits is True
        assert med_plan.quantize == "8bit"
        assert any("approximate (quantized)" in w for w in med_plan.warnings)

        # 2B params -> 8bit needs 2.4B (> 1.5 GiB), 4bit needs 1.2B (< 1.5 GiB) -> 4bit
        large_plan = plan_for_model(2_000_000_000, profile)
        assert large_plan.fits is True
        assert large_plan.quantize == "4bit"
        assert any("approximate (quantized)" in w for w in large_plan.warnings)

        # 10B params -> even 4bit needs 6.0B (> 1.5 GiB) -> fits=False
        huge_plan = plan_for_model(10_000_000_000, profile)
        assert huge_plan.fits is False
        assert huge_plan.quantize is None
        assert len(huge_plan.warnings) == 1


def test_detect_and_plan_multi_gpu() -> None:
    # GPU 0 busy (3 GiB free), GPU 1 freer (5 GiB free) -> total 8 GiB free.
    gpu_mem = {
        0: (3 * GIB, 8 * GIB),
        1: (5 * GIB, 8 * GIB),
    }
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.mem_get_info", side_effect=lambda idx=0: gpu_mem[idx]),
        patch("torch.cuda.is_bf16_supported", return_value=False),
    ):
        profile = detect()
        assert profile.device == "cuda:1"
        assert profile.dtype == torch.float16
        assert profile.free_memory_bytes == 5 * GIB
        assert profile.n_gpus == 2

        # 3B params in fp16 -> 3B * 2 * 1.2 = 7.2 GB (> 5 GiB single GPU, <= 8 GiB across 2 GPUs)
        plan = plan_for_model(3_000_000_000, profile)
        assert plan.fits is True
        assert plan.device_map == "auto"
        assert plan.quantize is None
        assert plan.warnings == []


def test_cpu_never_offers_quantization_when_model_exceeds_ram() -> None:
    cpu_profile = ComputeProfile(
        device="cpu",
        dtype=torch.float32,
        free_memory_bytes=2 * GIB,
        total_memory_bytes=8 * GIB,
        n_gpus=0,
        remote_available=False,
    )
    plan = plan_for_model(2_000_000_000, cpu_profile)
    assert plan.fits is False
    assert plan.quantize is None


def test_batch_size_clamped_between_1_and_64() -> None:
    tiny_mem = ComputeProfile(
        device="cpu",
        dtype=torch.float32,
        free_memory_bytes=1024,
        total_memory_bytes=1024 * 1024,
        n_gpus=0,
        remote_available=False,
    )
    huge_mem = ComputeProfile(
        device="cuda",
        dtype=torch.bfloat16,
        free_memory_bytes=80 * GIB,
        total_memory_bytes=80 * GIB,
        n_gpus=1,
        remote_available=False,
    )
    assert batch_size_for(seq_len=256, hidden=1024, n_layers_cached=24, profile=tiny_mem) == 1
    assert batch_size_for(seq_len=64, hidden=512, n_layers_cached=4, profile=huge_mem) == 64


def test_estimate_seconds_scales_with_passes_and_device() -> None:
    cpu_profile = ComputeProfile(
        device="cpu",
        dtype=torch.float32,
        free_memory_bytes=8 * GIB,
        total_memory_bytes=16 * GIB,
        n_gpus=0,
        remote_available=False,
    )
    gpu_profile = ComputeProfile(
        device="cuda",
        dtype=torch.bfloat16,
        free_memory_bytes=16 * GIB,
        total_memory_bytes=16 * GIB,
        n_gpus=1,
        remote_available=False,
    )
    assert estimate_seconds(0, 100_000_000, cpu_profile) == 0.0
    t_10 = estimate_seconds(10, 100_000_000, cpu_profile)
    t_100 = estimate_seconds(100, 100_000_000, cpu_profile)
    t_gpu = estimate_seconds(100, 100_000_000, gpu_profile)
    assert t_100 > t_10 > 0.0
    assert t_gpu < t_100


def test_device_module_cli_prints_profile() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "engine.compute.device"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Detected compute profile:" in proc.stdout
    assert "device:" in proc.stdout
