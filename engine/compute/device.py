"""Hardware detection and compute profile inspection."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

import torch


@dataclass
class ComputeProfile:
    """Detected hardware capabilities and available memory."""

    device: str
    dtype: torch.dtype
    free_memory_bytes: int | None
    total_memory_bytes: int | None
    n_gpus: int
    remote_available: bool


def precision_name(dtype: torch.dtype) -> str:
    """Return short contract precision string for a PyTorch dtype."""
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def _host_memory_info() -> tuple[int | None, int | None]:
    """Read free and total host RAM in bytes without external dependencies."""
    if sys.platform == "win32":

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            if ok:
                return int(stat.ullAvailPhys), int(stat.ullTotalPhys)
        except Exception:
            return None, None
        return None, None

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            if page_size > 0 and total_pages > 0 and avail_pages >= 0:
                return avail_pages * page_size, total_pages * page_size
        except (ValueError, OSError):
            return None, None

    return None, None


def _remote_key_present() -> bool:
    """Check whether an NDIF remote key is configured without reading its value."""
    return bool(os.environ.get("NDIF_API_KEY") or os.environ.get("NDIF_KEY"))


def detect() -> ComputeProfile:
    """Detect the best local execution device, precision, and free memory."""
    remote = _remote_key_present()

    if torch.cuda.is_available():
        n_gpus = int(torch.cuda.device_count())
        if n_gpus > 0:
            gpu_infos: list[tuple[int, int]] = []
            for idx in range(n_gpus):
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpu_infos.append((int(free_b), int(total_b)))

            best_idx = max(range(n_gpus), key=lambda i: gpu_infos[i][0])
            free_bytes, total_bytes = gpu_infos[best_idx]
            device_str = "cuda" if n_gpus == 1 else f"cuda:{best_idx}"

            bf16_ok = False
            if hasattr(torch.cuda, "is_bf16_supported"):
                try:
                    bf16_ok = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    bf16_ok = False
            dtype = torch.bfloat16 if bf16_ok else torch.float16

            return ComputeProfile(
                device=device_str,
                dtype=dtype,
                free_memory_bytes=free_bytes,
                total_memory_bytes=total_bytes,
                n_gpus=n_gpus,
                remote_available=remote,
            )

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        free_b, total_b = _host_memory_info()
        return ComputeProfile(
            device="mps",
            dtype=torch.float16,
            free_memory_bytes=free_b,
            total_memory_bytes=total_b,
            n_gpus=0,
            remote_available=remote,
        )

    free_b, total_b = _host_memory_info()
    return ComputeProfile(
        device="cpu",
        dtype=torch.float32,
        free_memory_bytes=free_b,
        total_memory_bytes=total_b,
        n_gpus=0,
        remote_available=remote,
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024**3)
    return f"{gib:.2f} GiB ({value:,} bytes)"


def main() -> None:
    profile = detect()
    print("Detected compute profile:")
    print(f"  device:             {profile.device}")
    print(f"  precision:          {precision_name(profile.dtype)} ({profile.dtype})")
    print(f"  free_memory_bytes:  {_format_bytes(profile.free_memory_bytes)}")
    print(f"  total_memory_bytes: {_format_bytes(profile.total_memory_bytes)}")
    print(f"  n_gpus:             {profile.n_gpus}")
    print(f"  remote_available:   {profile.remote_available}")


if __name__ == "__main__":
    main()
