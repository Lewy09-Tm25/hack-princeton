"""Control experiments and statistical rate comparisons for verifying findings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from scipy.stats import fisher_exact

from engine.blocks.directions import random_direction_like
from engine.contracts import ControlResult


def _extract_metric_value(raw: Any) -> float:
    if hasattr(raw, "accuracy"):
        return float(raw.accuracy)
    if isinstance(raw, torch.Tensor):
        return float(raw.item())
    return float(raw)


def control_shuffled_labels(
    train_fn: Callable[[Any, np.ndarray], Any],
    X: Any,
    y: Any,
    n_draws: int = 20,
    seed: int = 0,
    real_accuracy: float | None = None,
) -> ControlResult:
    """Run shuffled-label probe controls (`docs/defaults.md` section 2.1).

    Passes iff real held-out accuracy is strictly above the 95th percentile of
    shuffled accuracies AND at least 0.10 above the mean shuffled accuracy.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1 for shuffled-label control.")

    y_arr = (
        y.detach().to(device="cpu").numpy().reshape(-1)
        if isinstance(y, torch.Tensor)
        else np.asarray(y).reshape(-1)
    )
    real_acc = (
        float(real_accuracy)
        if real_accuracy is not None
        else _extract_metric_value(train_fn(X, y_arr))
    )

    rng = np.random.default_rng(int(seed))
    shuffled_accs: list[float] = []
    for _ in range(int(n_draws)):
        y_perm = rng.permutation(y_arr)
        shuffled_accs.append(_extract_metric_value(train_fn(X, y_perm)))

    p95 = float(np.percentile(shuffled_accs, 95))
    mean_shuf = float(np.mean(shuffled_accs))
    passed = bool((real_acc > p95) and (real_acc >= mean_shuf + 0.10 - 1e-12))

    return ControlResult(
        kind="shuffled_labels",
        value=round(mean_shuf, 6),
        passed=passed,
        notes=(
            f"real={real_acc:.4f}, mean_shuffled={mean_shuf:.4f}, "
            f"p95={p95:.4f}, n_draws={n_draws}"
        ),
    )


def control_random_direction(
    effect_fn: Callable[[torch.Tensor], float],
    direction: torch.Tensor,
    n_draws: int = 10,
    seed: int = 100,
    real_effect: float | None = None,
) -> ControlResult:
    """Run equal-norm random-direction controls (`docs/defaults.md` section 2.2).

    Default uses 10 draws with seeds 100..109. Passes iff real effect is larger
    than the maximum random effect AND at least twice the mean absolute random effect.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1 for random-direction control.")

    real_val = (
        float(real_effect)
        if real_effect is not None
        else _extract_metric_value(effect_fn(direction))
    )

    rand_effects: list[float] = []
    for idx in range(int(n_draws)):
        r_vec = random_direction_like(direction, seed=int(seed) + idx)
        rand_effects.append(_extract_metric_value(effect_fn(r_vec)))

    max_rand = float(np.max(rand_effects))
    mean_abs_rand = float(np.mean(np.abs(rand_effects)))
    passed = bool((real_val > max_rand) and (real_val >= 2.0 * mean_abs_rand - 1e-12))

    return ControlResult(
        kind="random_direction",
        value=round(mean_abs_rand, 6),
        passed=passed,
        notes=(
            f"real={real_val:.4f}, max_random={max_rand:.4f}, "
            f"mean_abs_random={mean_abs_rand:.4f}, seeds={seed}..{seed + n_draws - 1}"
        ),
    )


def compare_behavioral_rates(
    model_count: int,
    model_total: int,
    ref_count: int,
    ref_total: int,
    min_abs_diff: float = 0.10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two behavioral proportions via Fisher's exact test (`docs/defaults.md` 2.4).

    Passes iff `p < 0.05` AND `abs(rate_model - rate_ref) >= 0.10`.
    """
    if model_total <= 0 or ref_total <= 0:
        raise ValueError("Sample totals must be positive for behavioral rate comparison.")
    if not (0 <= model_count <= model_total) or not (0 <= ref_count <= ref_total):
        raise ValueError("Event counts must lie between 0 and total.")

    rate_model = model_count / float(model_total)
    rate_ref = ref_count / float(ref_total)
    diff = rate_model - rate_ref
    abs_diff = abs(diff)

    table = [
        [int(model_count), int(model_total - model_count)],
        [int(ref_count), int(ref_total - ref_count)],
    ]
    _, p_val = fisher_exact(table, alternative="two-sided")
    p_float = float(p_val)
    passed = bool((p_float < alpha) and (abs_diff >= min_abs_diff - 1e-12))

    sample_note = ""
    if model_total < 100 or ref_total < 100:
        sample_note = f" (small sample: n_model={model_total}, n_ref={ref_total})"

    return {
        "rate_model": rate_model,
        "rate_ref": rate_ref,
        "diff": diff,
        "abs_diff": abs_diff,
        "p_value": p_float,
        "passed": passed,
        "notes": (
            f"model={rate_model:.3f} ({model_count}/{model_total}), "
            f"ref={rate_ref:.3f} ({ref_count}/{ref_total}), "
            f"diff={diff:+.3f}, p={p_float:.4g}{sample_note}"
        ),
    }


def control_reference_model(
    run_fn: Callable[[Any], Any],
    reference: Any,
    model_counts: tuple[int, int] | None = None,
    real_value: float | None = None,
    min_margin: float = 0.10,
) -> ControlResult:
    """Evaluate the identical measurement procedure on a clean or base reference model."""
    out = run_fn(reference)

    if isinstance(out, ControlResult):
        return out

    if model_counts is not None and isinstance(out, tuple | list) and len(out) == 2:
        comp = compare_behavioral_rates(
            model_count=int(model_counts[0]),
            model_total=int(model_counts[1]),
            ref_count=int(out[0]),
            ref_total=int(out[1]),
            min_abs_diff=min_margin,
        )
        return ControlResult(
            kind="reference_model",
            value=round(float(comp["rate_ref"]), 6),
            passed=bool(comp["passed"]),
            notes=str(comp["notes"]),
        )

    if isinstance(out, dict):
        val = out.get("value")
        passed = out.get("passed")
        if passed is None and real_value is not None and val is not None:
            passed = bool((float(real_value) - float(val)) >= min_margin - 1e-12)
        return ControlResult(
            kind="reference_model",
            value=float(val) if val is not None else None,
            passed=bool(passed) if passed is not None else True,
            notes=out.get("notes"),
        )

    ref_val = _extract_metric_value(out)
    if real_value is not None:
        diff = float(real_value) - ref_val
        passed_flag = bool(abs(diff) >= min_margin - 1e-12 and diff > 0.0)
        note = f"real={real_value:.4f}, reference={ref_val:.4f}, diff={diff:+.4f}"
    else:
        passed_flag = True
        note = f"reference={ref_val:.4f}"

    return ControlResult(
        kind="reference_model",
        value=round(ref_val, 6),
        passed=passed_flag,
        notes=note,
    )
