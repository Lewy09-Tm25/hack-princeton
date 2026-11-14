"""Memory planning, batch sizing, and runtime estimation for model runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from engine.compute.device import ComputeProfile

MEMORY_HEADROOM_FACTOR = 1.20
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64

_BENCHMARK_FLOPS_CACHE: dict[tuple[str, str], float] = {}


@dataclass
class LoadPlan:
    """Execution and memory plan for loading a model of known parameter count."""

    device: str
    device_map: str | dict[str, Any] | None
    dtype: torch.dtype
    quantize: Literal["8bit", "4bit"] | None = None
    fits: bool = True
    warnings: list[str] = field(default_factory=list)


def _bytes_per_param(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 2.0
    if dtype == torch.float64:
        return 8.0
    return 4.0


def _required_bytes(n_params: int, bytes_per_param: float) -> int:
    return int(max(0, n_params) * bytes_per_param * MEMORY_HEADROOM_FACTOR)


def _multi_gpu_free_bytes(profile: ComputeProfile) -> int:
    """Return total free memory across all CUDA GPUs when multiple GPUs exist."""
    if profile.free_memory_bytes is None:
        return 0
    if profile.n_gpus <= 1 or not profile.device.startswith("cuda"):
        return profile.free_memory_bytes

    if torch.cuda.is_available():
        try:
            count = int(torch.cuda.device_count())
            if count == profile.n_gpus:
                return sum(int(torch.cuda.mem_get_info(i)[0]) for i in range(count))
        except Exception:
            pass

    return profile.free_memory_bytes * profile.n_gpus


def plan_for_model(n_params: int, profile: ComputeProfile) -> LoadPlan:
    """Compute a safe load plan with 20% weight memory headroom."""
    is_cuda = profile.device.startswith("cuda")
    bpp = _bytes_per_param(profile.dtype)
    needed_native = _required_bytes(n_params, bpp)

    if profile.free_memory_bytes is None:
        return LoadPlan(
            device=profile.device,
            device_map=None,
            dtype=profile.dtype,
            quantize=None,
            fits=True,
            warnings=[],
        )

    single_free = profile.free_memory_bytes
    multi_free = _multi_gpu_free_bytes(profile)

    # 1. Fits on the chosen single device in native dtype.
    if needed_native <= single_free:
        return LoadPlan(
            device=profile.device,
            device_map=None,
            dtype=profile.dtype,
            quantize=None,
            fits=True,
            warnings=[],
        )

    # 2. Spread across multiple GPUs when available.
    if is_cuda and profile.n_gpus > 1 and needed_native <= multi_free:
        return LoadPlan(
            device=profile.device,
            device_map="auto",
            dtype=profile.dtype,
            quantize=None,
            fits=True,
            warnings=[],
        )

    # 3. Offer quantization only on CUDA when full weights do not fit.
    if is_cuda:
        for quant_mode, quant_bpp in (("8bit", 1.0), ("4bit", 0.5)):
            needed_quant = _required_bytes(n_params, quant_bpp)
            if needed_quant <= single_free:
                return LoadPlan(
                    device=profile.device,
                    device_map=None,
                    dtype=profile.dtype,
                    quantize=quant_mode,  # type: ignore[arg-type]
                    fits=True,
                    warnings=[
                        f"Model requires {quant_mode} quantization to fit in GPU memory; "
                        "findings are approximate (quantized)."
                    ],
                )
            if profile.n_gpus > 1 and needed_quant <= multi_free:
                return LoadPlan(
                    device=profile.device,
                    device_map="auto",
                    dtype=profile.dtype,
                    quantize=quant_mode,  # type: ignore[arg-type]
                    fits=True,
                    warnings=[
                        f"Model requires {quant_mode} quantization across GPUs to fit in memory; "
                        "findings are approximate (quantized)."
                    ],
                )

    # 4. Model does not fit locally.
    remote_hint = (
        " Remote compute (NDIF) key is configured and can be used with approval."
        if profile.remote_available
        else " Consider using approved remote compute."
    )
    return LoadPlan(
        device=profile.device,
        device_map=None,
        dtype=profile.dtype,
        quantize=None,
        fits=False,
        warnings=[
            f"Model does not fit in available memory on {profile.device}.{remote_hint}"
        ],
    )


def batch_size_for(
    seq_len: int,
    hidden: int,
    n_layers_cached: int,
    profile: ComputeProfile,
    n_params: int = 0,
) -> int:
    """Choose a batch size clamped between 1 and 64 based on available memory."""
    safe_seq = max(1, int(seq_len))
    safe_hidden = max(1, int(hidden))
    safe_layers = max(1, int(n_layers_cached))

    if profile.free_memory_bytes is None:
        default_bs = 8 if profile.device == "cpu" else 16
        return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, default_bs))

    weight_bytes = int(max(0, n_params) * _bytes_per_param(profile.dtype))
    usable_bytes = max(0, int(profile.free_memory_bytes * 0.80) - weight_bytes)

    # Estimate working + cached activation bytes per sequence (float32 cache + forward buffers).
    per_example_bytes = safe_seq * safe_hidden * safe_layers * 16
    if per_example_bytes <= 0:
        return MIN_BATCH_SIZE

    raw_batch = usable_bytes // per_example_bytes
    return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, int(raw_batch)))


def _measure_device_flops(profile: ComputeProfile) -> float:
    """Run a tiny cached matmul benchmark when the device is locally usable."""
    cache_key = (profile.device, str(profile.dtype))
    if cache_key in _BENCHMARK_FLOPS_CACHE:
        return _BENCHMARK_FLOPS_CACHE[cache_key]

    can_run_locally = False
    if profile.device == "cpu":
        can_run_locally = True
    elif profile.device.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = 0 if ":" not in profile.device else int(profile.device.split(":")[1])
            can_run_locally = idx < int(torch.cuda.device_count())
        except Exception:
            can_run_locally = False
    elif profile.device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        can_run_locally = bool(mps_backend is not None and mps_backend.is_available())

    if can_run_locally:
        dim = 256
        steps = 4
        with torch.inference_mode():
            a = torch.randn((dim, dim), device=profile.device, dtype=profile.dtype)
            b = torch.randn((dim, dim), device=profile.device, dtype=profile.dtype)
            _ = a @ b
            if profile.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                _ = a @ b
            if profile.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-6)
        flops = (2.0 * (dim**3) * steps) / elapsed
        # Account for memory-bandwidth overhead in transformer forward passes (~25% of dense gemm).
        effective_flops = max(flops * 0.25, 1e9)
    else:
        # Fallback for simulated/mocked accelerator profiles in tests.
        if profile.device.startswith("cuda"):
            effective_flops = 8e12
        elif profile.device == "mps":
            effective_flops = 1.5e12
        else:
            effective_flops = 4e10

    _BENCHMARK_FLOPS_CACHE[cache_key] = effective_flops
    return effective_flops


def estimate_seconds(
    n_forward_passes: int,
    n_params: int,
    profile: ComputeProfile,
    avg_seq_len: int = 128,
) -> float:
    """Estimate wall-clock runtime in seconds for a given number of forward passes."""
    if n_forward_passes <= 0 or n_params <= 0:
        return 0.0

    flops_per_sec = _measure_device_flops(profile)
    # Approximate FLOPs per forward pass as 2 * n_params * avg_seq_len.
    total_flops = 2.0 * float(n_params) * float(max(1, avg_seq_len)) * float(n_forward_passes)
    return round(max(0.01, total_flops / flops_per_sec), 3)
