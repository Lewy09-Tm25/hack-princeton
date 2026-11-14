"""Activation patching across source and clean inputs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch

from engine.blocks.activations import _accepts_attention_mask, _unwrap_saved_tensor, resolve_envoy
from engine.blocks.generate import PromptInput, encode_prompts
from engine.contracts import Component, ModelMap
from engine.loader.load import LoadedModel
from engine.modelmap.build import hook_path


def _forward_logits(
    model: LoadedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pass_mask: bool,
) -> torch.Tensor:
    out = (
        model.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        if pass_mask
        else model.hf_model(input_ids)
    )
    return out.logits if hasattr(out, "logits") else out


def patch_component(
    model: LoadedModel,
    mmap: ModelMap,
    clean_prompts: list[str] | Sequence[PromptInput],
    source_prompts: list[str] | Sequence[PromptInput],
    component_id: str,
    metric_fn: Callable[..., float],
    chat_template: bool = False,
) -> float:
    """Swap `component_id` activations from `source_prompts` into `clean_prompts` and score."""
    if len(clean_prompts) != len(source_prompts):
        raise ValueError(
            f"clean_prompts ({len(clean_prompts)}) and source_prompts ({len(source_prompts)}) "
            "must have the same length."
        )
    if not clean_prompts:
        raise ValueError("patch_component requires at least one prompt pair.")

    by_id: dict[str, Component] = {c.id: c for c in mmap.components}
    if component_id not in by_id:
        raise KeyError(f"Component {component_id!r} not found in model map for {mmap.model_id!r}.")
    comp = by_id[component_id]

    head_info: tuple[int, int] | None = None
    if comp.kind == "attention_head" and comp.module is None:
        parent = by_id.get(f"L{comp.index}.attn")
        if parent is None or parent.module is None or not parent.n_heads:
            raise ValueError(f"Head-level patching is unavailable for {component_id!r}.")
        mod_path = parent.module
        head_info = (int(comp.head or 0), int(parent.n_heads))
    else:
        mod_path = hook_path(mmap, component_id)

    pass_mask = _accepts_attention_mask(model.hf_model)

    with torch.inference_mode():
        src_enc = encode_prompts(
            model=model,
            prompts=source_prompts,
            chat_template=chat_template,
            padding_side="left",
        )
        cln_enc = encode_prompts(
            model=model,
            prompts=clean_prompts,
            chat_template=chat_template,
            padding_side="left",
        )

        src_trace = (
            model.nn.trace(src_enc["input_ids"], attention_mask=src_enc["attention_mask"])
            if pass_mask
            else model.nn.trace(src_enc["input_ids"])
        )
        with src_trace:
            src_envoy = resolve_envoy(model.nn, mod_path)
            saved_src = src_envoy.output.save()
        src_act = _unwrap_saved_tensor(saved_src).detach().clone()

        target_module = model.hf_model.get_submodule(mod_path)

        def _patch_hook(_mod: torch.nn.Module, _inp: Any, out: Any) -> Any:
            is_tuple = isinstance(out, tuple)
            cur = out[0] if is_tuple else out
            if not isinstance(cur, torch.Tensor):
                return out

            patched = cur.clone()
            src_cast = src_act.to(device=patched.device, dtype=patched.dtype)

            if head_info is not None:
                h_idx, n_heads = head_info
                if patched.shape[-1] % n_heads != 0:
                    raise ValueError("Attention output hidden size not divisible by n_heads.")
                h_dim = patched.shape[-1] // n_heads
                p_view = patched.reshape(*patched.shape[:-1], n_heads, h_dim)
                s_view = src_cast.reshape(*src_cast.shape[:-1], n_heads, h_dim)
                if p_view.shape[1] == s_view.shape[1]:
                    p_view[:, :, h_idx, :] = s_view[:, :, h_idx, :]
                else:
                    p_view[:, -1, h_idx, :] = s_view[:, -1, h_idx, :]
                patched = p_view.reshape_as(cur)
            else:
                if patched.shape == src_cast.shape:
                    patched = src_cast
                else:
                    patched[:, -1, :] = src_cast[:, -1, :]

            return (patched, *out[1:]) if is_tuple else patched

        clean_logits = _forward_logits(
            model, cln_enc["input_ids"], cln_enc["attention_mask"], pass_mask
        )

        handle = target_module.register_forward_hook(_patch_hook)
        try:
            patched_logits = _forward_logits(
                model, cln_enc["input_ids"], cln_enc["attention_mask"], pass_mask
            )
        finally:
            handle.remove()

        try:
            n_params = len(inspect.signature(metric_fn).parameters)
        except (TypeError, ValueError):
            n_params = 1

        score = (
            metric_fn(patched_logits, clean_logits)
            if n_params >= 2
            else metric_fn(patched_logits)
        )
        return float(score.item() if isinstance(score, torch.Tensor) else score)
