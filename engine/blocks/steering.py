"""Activation steering, strength sweeps, residual norm estimation, and output quality checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from engine.blocks.activations import collect_activations
from engine.blocks.generate import PromptInput, _greedy_generate_batch, encode_prompts
from engine.compute import batch_size_for, detect
from engine.contracts import ModelMap
from engine.loader.load import LoadedModel
from engine.modelmap.build import hook_path

DEFAULT_STRENGTH_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0)


def residual_norm(
    model: LoadedModel,
    mmap: ModelMap,
    prompts: list[str] | Sequence[PromptInput],
    component_id: str,
) -> float:
    """Return the mean L2 norm of activations at `component_id` over `prompts`."""
    if not prompts:
        return 1.0
    acts = collect_activations(
        model=model,
        mmap=mmap,
        prompts=prompts,
        component_ids=[component_id],
        position="last",
    )[component_id]
    norms = acts.norm(dim=-1)
    mean_val = float(norms.mean().item())
    return mean_val if mean_val > 1e-12 else 1.0


def _make_steering_delta(
    direction: torch.Tensor,
    strength: float,
    strength_mode: Literal["absolute", "relative_to_norm"],
    norm_scale: float,
) -> torch.Tensor | None:
    vec = direction.detach().to(dtype=torch.float32).flatten()
    vec_norm = float(vec.norm().item())
    if strength == 0.0 or vec_norm <= 1e-12:
        return None

    if strength_mode == "relative_to_norm":
        unit_vec = vec / vec_norm
        return unit_vec * (float(strength) * float(norm_scale))
    if strength_mode == "absolute":
        return vec * float(strength)
    raise ValueError(
        f"Invalid strength_mode {strength_mode!r}; expected 'absolute' or 'relative_to_norm'."
    )


def _generate_with_delta(
    model: LoadedModel,
    target_module: torch.nn.Module,
    prompts: Sequence[PromptInput],
    delta: torch.Tensor | None,
    max_new_tokens: int,
    chat_template: bool,
    batch_size: int | None = None,
    seed: int = 0,
) -> list[str]:
    if not prompts:
        return []

    profile = detect()
    hidden = int(getattr(model.config, "hidden_size", None) or 512)
    eff_bs = batch_size or batch_size_for(
        seq_len=128 + int(max_new_tokens),
        hidden=hidden,
        n_layers_cached=1,
        profile=profile,
        n_params=model.n_params,
    )

    def _steering_hook(_mod: torch.nn.Module, _inp: Any, out: Any) -> Any:
        if delta is None:
            return out
        if isinstance(out, torch.Tensor):
            shift = delta.to(device=out.device, dtype=out.dtype)
            return out + shift
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            first = out[0]
            shift = delta.to(device=first.device, dtype=first.dtype)
            return (first + shift, *out[1:])
        return out

    outputs: list[str] = []
    torch.manual_seed(int(seed))
    handle = target_module.register_forward_hook(_steering_hook) if delta is not None else None
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), eff_bs):
                chunk = prompts[start : start + eff_bs]
                inputs = encode_prompts(
                    model=model,
                    prompts=chunk,
                    chat_template=chat_template,
                    padding_side="left",
                )
                prompt_len = int(inputs["input_ids"].shape[1])
                out_ids = _greedy_generate_batch(
                    model=model,
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                )
                new_ids = out_ids[:, prompt_len:]
                decoded = model.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                outputs.extend(text.strip() for text in decoded)
    finally:
        if handle is not None:
            handle.remove()

    return outputs


def generate_with_steering(
    model: LoadedModel,
    mmap: ModelMap,
    prompts: list[str] | Sequence[PromptInput],
    component_id: str,
    direction: torch.Tensor,
    strength: float,
    strength_mode: Literal["absolute", "relative_to_norm"] = "relative_to_norm",
    max_new_tokens: int = 128,
    chat_template: bool = True,
    batch_size: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Generate completions while adding a scaled direction to `component_id` at all positions."""
    mod_path = hook_path(mmap, component_id)
    target_module = model.hf_model.get_submodule(mod_path)

    norm_scale = 1.0
    if strength_mode == "relative_to_norm" and strength != 0.0 and float(direction.norm()) > 1e-12:
        norm_scale = residual_norm(model, mmap, prompts, component_id)

    delta = _make_steering_delta(
        direction=direction,
        strength=strength,
        strength_mode=strength_mode,
        norm_scale=norm_scale,
    )
    return _generate_with_delta(
        model=model,
        target_module=target_module,
        prompts=prompts,
        delta=delta,
        max_new_tokens=max_new_tokens,
        chat_template=chat_template,
        batch_size=batch_size,
        seed=seed,
    )


def sweep_strengths(
    model: LoadedModel,
    mmap: ModelMap,
    prompts: list[str] | Sequence[PromptInput],
    component_id: str,
    direction: torch.Tensor,
    strengths: Sequence[float] = DEFAULT_STRENGTH_GRID,
    strength_mode: Literal["absolute", "relative_to_norm"] = "relative_to_norm",
    max_new_tokens: int = 128,
    chat_template: bool = True,
) -> dict[float, list[str]]:
    """Run steered generation across a grid of steering multipliers."""
    mod_path = hook_path(mmap, component_id)
    target_module = model.hf_model.get_submodule(mod_path)

    norm_scale = 1.0
    if strength_mode == "relative_to_norm" and float(direction.norm()) > 1e-12:
        norm_scale = residual_norm(model, mmap, prompts, component_id)

    results: dict[float, list[str]] = {}
    for s in strengths:
        coeff = float(s)
        delta = _make_steering_delta(
            direction=direction,
            strength=coeff,
            strength_mode=strength_mode,
            norm_scale=norm_scale,
        )
        results[coeff] = _generate_with_delta(
            model=model,
            target_module=target_module,
            prompts=prompts,
            delta=delta,
            max_new_tokens=max_new_tokens,
            chat_template=chat_template,
        )
    return results


def _repeated_trigram_fraction(tokens: list[Any]) -> float:
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    total = len(trigrams)
    repeated = total - len(set(trigrams))
    return repeated / float(total)


def check_steered_outputs(
    outputs: Sequence[str],
    tokenizer: Any | None = None,
    max_repeated_3gram: float = 0.30,
    max_short_fraction: float = 0.20,
    min_tokens: int = 5,
) -> dict[str, Any]:
    """Validate steered completions against `docs/defaults.md` section 5.4 quality rules."""
    if not outputs:
        return {
            "passed": False,
            "reasons": ["empty_output_list"],
            "max_repeated_3gram_fraction": 0.0,
            "empty_count": 0,
            "short_fraction": 1.0,
        }

    empty_count = 0
    short_count = 0
    worst_trigram = 0.0

    for text in outputs:
        stripped = text.strip()
        if not stripped:
            empty_count += 1
            short_count += 1
            continue

        if tokenizer is not None and hasattr(tokenizer, "encode"):
            toks: list[Any] = list(tokenizer.encode(stripped, add_special_tokens=False))
        else:
            toks = stripped.lower().split()

        if len(toks) < min_tokens:
            short_count += 1

        # Evaluate 3-grams over whitespace words as well as tokens to catch word repetition.
        word_toks = stripped.lower().split()
        rep_frac = max(_repeated_trigram_fraction(toks), _repeated_trigram_fraction(word_toks))
        if rep_frac > worst_trigram:
            worst_trigram = rep_frac

    short_fraction = short_count / float(len(outputs))
    reasons: list[str] = []
    if empty_count > 0:
        reasons.append(f"{empty_count} output(s) were empty")
    if worst_trigram > max_repeated_3gram:
        reasons.append(
            f"repeated 3-gram fraction {worst_trigram:.2f} exceeds {max_repeated_3gram:.2f}"
        )
    if short_fraction > max_short_fraction:
        reasons.append(
            f"fraction of outputs shorter than {min_tokens} tokens ({short_fraction:.2f}) "
            f"exceeds {max_short_fraction:.2f}"
        )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "max_repeated_3gram_fraction": worst_trigram,
        "empty_count": empty_count,
        "short_fraction": short_fraction,
    }
