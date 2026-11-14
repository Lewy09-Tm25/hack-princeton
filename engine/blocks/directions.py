"""Linear direction utilities: difference-of-means, projections, random controls, cosine."""

from __future__ import annotations

import torch


def diff_of_means(
    acts_pos: torch.Tensor,
    acts_neg: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute mean(acts_pos) - mean(acts_neg), optionally unit-normalized."""
    if acts_pos.ndim != 2 or acts_neg.ndim != 2:
        raise ValueError(
            f"Expected 2D activation matrices [n_items, dim]; "
            f"got {tuple(acts_pos.shape)} and {tuple(acts_neg.shape)}."
        )
    if acts_pos.shape[1] != acts_neg.shape[1]:
        raise ValueError(
            f"Activation hidden dimensions must match; "
            f"got {acts_pos.shape[1]} and {acts_neg.shape[1]}."
        )

    diff = acts_pos.to(dtype=torch.float32).mean(dim=0) - acts_neg.to(dtype=torch.float32).mean(
        dim=0
    )
    if normalize:
        norm = diff.norm()
        if float(norm.item()) > 1e-12:
            diff = diff / norm
    return diff


def projection(acts: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project row vectors in `acts` onto unit-normalized `direction`."""
    flat_dir = direction.to(device=acts.device, dtype=torch.float32).flatten()
    norm = flat_dir.norm()
    if float(norm.item()) <= 1e-12:
        return torch.zeros(acts.shape[:-1], device=acts.device, dtype=torch.float32)
    unit_dir = flat_dir / norm
    return acts.to(dtype=torch.float32) @ unit_dir


def random_direction_like(direction: torch.Tensor, seed: int) -> torch.Tensor:
    """Sample a reproducible Gaussian vector scaled to the exact L2 norm of `direction`."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    raw = torch.randn(direction.shape, generator=gen, dtype=torch.float32, device="cpu")
    raw_norm = raw.norm()
    target_norm = direction.detach().to(device="cpu", dtype=torch.float32).norm()

    if float(raw_norm.item()) <= 1e-12 or float(target_norm.item()) <= 1e-12:
        scaled = torch.zeros_like(raw)
    else:
        scaled = raw * (target_norm / raw_norm)

    return scaled.to(device=direction.device, dtype=direction.dtype)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity between two tensors in [-1.0, 1.0] (0.0 if zero-norm)."""
    va = a.detach().to(dtype=torch.float32).flatten()
    vb = b.detach().to(device=va.device, dtype=torch.float32).flatten()
    if va.shape != vb.shape:
        raise ValueError(
            f"Vectors must have matching shapes for cosine similarity; "
            f"got {tuple(va.shape)} vs {tuple(vb.shape)}."
        )
    denom = float((va.norm() * vb.norm()).item())
    if denom <= 1e-12:
        return 0.0
    sim = float(torch.dot(va, vb).item()) / denom
    return max(-1.0, min(1.0, sim))
