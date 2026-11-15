"""Batched activation collection via nnsight tracing with optional safetensors disk caching."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from engine.blocks.generate import PromptInput, encode_prompts
from engine.blocks.progress import report_progress
from engine.compute import batch_size_for, detect
from engine.contracts import Component, ModelMap
from engine.loader.load import LoadedModel
from engine.modelmap.build import hook_path


def resolve_envoy(nn_root: Any, module_path: str) -> Any:
    """Walk an `nnsight` wrapper along a dotted PyTorch module path."""
    cur = nn_root
    for part in module_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _unwrap_saved_tensor(saved_obj: Any) -> torch.Tensor:
    val = getattr(saved_obj, "value", saved_obj)
    if isinstance(val, torch.Tensor):
        return val
    if isinstance(val, tuple | list) and val and isinstance(val[0], torch.Tensor):
        return val[0]
    if hasattr(val, "last_hidden_state") and isinstance(val.last_hidden_state, torch.Tensor):
        return val.last_hidden_state
    raise TypeError(f"Unsupported traced module output type: {type(val)!r}")


def _accepts_attention_mask(hf_model: torch.nn.Module) -> bool:
    try:
        sig = inspect.signature(hf_model.forward)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.name == "attention_mask":
            return True
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _resolve_component_targets(
    mmap: ModelMap,
    component_ids: Sequence[str],
) -> tuple[dict[str, str], dict[str, tuple[str, int, int]]]:
    """Map requested component IDs to module paths and optional head slices.

    Returns:
      - `direct_paths`: `component_id -> module_path`
      - `head_specs`: `component_id -> (attn_module_path, head_idx, n_heads)`
    """
    by_id: dict[str, Component] = {c.id: c for c in mmap.components}
    direct_paths: dict[str, str] = {}
    head_specs: dict[str, tuple[str, int, int]] = {}

    for comp_id in component_ids:
        if comp_id not in by_id:
            raise KeyError(
                f"Component {comp_id!r} not found in model map for {mmap.model_id!r}."
            )
        comp = by_id[comp_id]
        if comp.kind == "attention_head" and comp.module is None:
            parent_id = f"L{comp.index}.attn"
            parent = by_id.get(parent_id)
            if parent is None or parent.module is None or not parent.n_heads:
                raise ValueError(
                    f"Head-level activation extraction is unavailable for {comp_id!r}."
                )
            head_specs[comp_id] = (parent.module, int(comp.head or 0), int(parent.n_heads))
        else:
            direct_paths[comp_id] = hook_path(mmap, comp_id)

    return direct_paths, head_specs


def _execution_order_paths(mmap: ModelMap, target_paths: set[str]) -> list[str]:
    """Order module paths in PyTorch forward-output completion order (children before parent)."""
    by_id: dict[str, Component] = {c.id: c for c in mmap.components}
    child_set: set[str] = {cid for c in mmap.components for cid in c.children}
    roots = [c.id for c in mmap.components if c.id not in child_set]

    ordered: list[str] = []
    seen: set[str] = set()

    def _visit(cid: str) -> None:
        comp = by_id.get(cid)
        if comp is None:
            return
        for child_id in comp.children:
            _visit(child_id)
        if comp.module and comp.module in target_paths and comp.module not in seen:
            seen.add(comp.module)
            ordered.append(comp.module)

    for root_id in roots:
        _visit(root_id)

    for rem in sorted(target_paths):
        if rem not in seen:
            ordered.append(rem)

    return ordered


def _pool_sequence(
    seq_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    position: Literal["last", "mean"],
) -> torch.Tensor:
    """Pool `[batch, seq_len, dim]` into `[batch, dim]` on CPU float32 using `attention_mask`."""
    cpu_acts = seq_tensor.detach().to(device="cpu", dtype=torch.float32)
    mask_cpu = attention_mask.detach().to(device="cpu")

    if cpu_acts.ndim == 2:
        return cpu_acts

    batch_sz, seq_len, _ = cpu_acts.shape
    if position == "last":
        # With right-padded batches, non-pad tokens sit at 0 .. lengths[b] - 1.
        lengths = mask_cpu.sum(dim=1).clamp(min=1, max=seq_len)
        last_indices = (lengths - 1).to(dtype=torch.long)
        row_indices = torch.arange(batch_sz, dtype=torch.long)
        return cpu_acts[row_indices, last_indices, :].contiguous()

    if position == "mean":
        weights = mask_cpu.to(dtype=torch.float32).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return ((cpu_acts * weights).sum(dim=1) / denom).contiguous()

    raise ValueError(f"Unsupported pooling position {position!r}; expected 'last' or 'mean'.")


def collect_activations(
    model: LoadedModel,
    mmap: ModelMap,
    prompts: list[str] | Sequence[PromptInput],
    component_ids: list[str],
    position: Literal["last", "mean"] = "last",
    batch_size: int | None = None,
    cache_dir: Path | None = None,
    chat_template: bool = False,
) -> dict[str, torch.Tensor]:
    """Collect `[n_prompts, dim]` CPU float32 activations via `nnsight` trace."""
    if not component_ids:
        return {}
    if not prompts:
        hidden = int(mmap.hidden_size or 1)
        return {cid: torch.empty((0, hidden), dtype=torch.float32) for cid in component_ids}

    direct_paths, head_specs = _resolve_component_targets(mmap, component_ids)
    needed_set = set(direct_paths.values()) | {
        attn_path for attn_path, _, _ in head_specs.values()
    }
    unique_paths = _execution_order_paths(mmap, needed_set)

    profile = detect()
    hidden_est = int(mmap.hidden_size or 512)
    eff_bs = batch_size or batch_size_for(
        seq_len=128,
        hidden=hidden_est,
        n_layers_cached=len(unique_paths),
        profile=profile,
        n_params=model.n_params,
    )

    pass_mask = _accepts_attention_mask(model.hf_model)
    chunk_files: list[Path] = []
    in_mem_chunks: dict[str, list[torch.Tensor]] = {cid: [] for cid in component_ids}

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, len(prompts), eff_bs)):
            report_progress("Reading the model's internal states", start, len(prompts))
            chunk = prompts[start : start + eff_bs]
            enc = encode_prompts(
                model=model,
                prompts=chunk,
                chat_template=chat_template,
                padding_side="right",
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            saved_proxies: dict[str, Any] = {}
            trace_ctx = (
                model.nn.trace(input_ids, attention_mask=attention_mask)
                if pass_mask
                else model.nn.trace(input_ids)
            )
            with trace_ctx:
                for mod_path in unique_paths:
                    envoy = resolve_envoy(model.nn, mod_path)
                    saved_proxies[mod_path] = envoy.output.save()

            raw_tensors: dict[str, torch.Tensor] = {
                mod_path: _unwrap_saved_tensor(proxy)
                for mod_path, proxy in saved_proxies.items()
            }

            batch_pooled: dict[str, torch.Tensor] = {}
            for cid, mod_path in direct_paths.items():
                batch_pooled[cid] = _pool_sequence(
                    raw_tensors[mod_path],
                    attention_mask=attention_mask,
                    position=position,
                )

            for cid, (attn_path, head_idx, n_heads) in head_specs.items():
                attn_out = raw_tensors[attn_path]
                if attn_out.shape[-1] % n_heads != 0:
                    raise ValueError(
                        f"Cannot split attention output of dim {attn_out.shape[-1]} "
                        f"evenly into {n_heads} heads for {cid!r}."
                    )
                head_dim = attn_out.shape[-1] // n_heads
                split_heads = attn_out.reshape(*attn_out.shape[:-1], n_heads, head_dim)
                head_tensor = split_heads[..., head_idx, :]
                batch_pooled[cid] = _pool_sequence(
                    head_tensor,
                    attention_mask=attention_mask,
                    position=position,
                )

            del raw_tensors
            del saved_proxies

            if cache_dir is not None:
                chunk_path = cache_dir / f"batch_{batch_idx:04d}.safetensors"
                save_file(batch_pooled, str(chunk_path))
                chunk_files.append(chunk_path)
            else:
                for cid, tensor in batch_pooled.items():
                    in_mem_chunks[cid].append(tensor)

    if cache_dir is not None:
        reloaded_lists: dict[str, list[torch.Tensor]] = {cid: [] for cid in component_ids}
        for cpath in chunk_files:
            loaded_batch = load_file(str(cpath), device="cpu")
            for cid in component_ids:
                reloaded_lists[cid].append(loaded_batch[cid].to(dtype=torch.float32))
        combined = {
            cid: torch.cat(reloaded_lists[cid], dim=0).contiguous() for cid in component_ids
        }
        save_file(combined, str(cache_dir / "activations.safetensors"))
        return combined

    return {
        cid: torch.cat(in_mem_chunks[cid], dim=0).contiguous() for cid in component_ids
    }
