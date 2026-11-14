"""Automatic inspection and uniform mapping of PyTorch causal language models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from engine.contracts import Component, ComponentKind, ModelMap
from engine.loader.load import LoadedModel
from engine.modelmap.patterns import (
    VALID_COMPONENT_KINDS,
    canonical_norm_suffix,
    classify_block_child,
    is_ignored_helper,
    is_mamba_or_ssm,
    is_norm_module,
)

UNSUPPORTED_MAMBA_MSG = "State-space / Mamba architectures are not supported in this version."


def _first_int(sources: list[Any], names: tuple[str, ...]) -> int | None:
    for src in sources:
        if src is None:
            continue
        for name in names:
            val = getattr(src, name, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


def _normalize_arch_family(config: Any, hf_model: nn.Module) -> str | None:
    text_cfg = getattr(config, "text_config", None)
    raw = getattr(config, "model_type", None) or getattr(text_cfg, "model_type", None)
    if isinstance(raw, str) and raw:
        low = raw.lower()
        if low == "gemma3_text":
            return "gemma3"
        return low
    return hf_model.__class__.__name__.lower()


def _find_layer_stack(
    hf_model: nn.Module, expected_layers: int | None
) -> tuple[str, nn.ModuleList]:
    raw_lists: list[tuple[str, nn.ModuleList]] = []
    for path, mod in hf_model.named_modules():
        if not path or not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            continue
        if path.split(".")[-1].lower() == "experts":
            continue
        child_types = {type(child) for child in mod}
        if len(child_types) != 1:
            continue
        raw_lists.append((path, mod))

    raw_paths = [p for p, _ in raw_lists]
    candidates: list[tuple[int, int, str, nn.ModuleList]] = []
    for path, mod in raw_lists:
        # Ignore per-layer sub-lists (such as MoE experts) nested inside an outer ModuleList.
        if any(path.startswith(f"{other}.") for other in raw_paths if other != path):
            continue
        matches_expected = 1 if (expected_layers is not None and len(mod) == expected_layers) else 0
        depth = path.count(".")
        candidates.append((matches_expected, depth, path, mod))

    if not candidates:
        raise ValueError(
            f"Could not find a uniform nn.ModuleList layer stack in {hf_model.__class__.__name__}."
        )

    candidates.sort(key=lambda item: (item[0], item[1], len(item[3])))
    _, _, best_path, best_stack = candidates[-1]
    return best_path, best_stack


def _is_under_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}.")


def _find_embedding(
    hf_model: nn.Module, layers_prefix: str, vocab_size: int | None
) -> tuple[str, nn.Embedding] | None:
    matches: list[tuple[int, str, nn.Embedding]] = []
    for path, mod in hf_model.named_modules():
        if not path or _is_under_prefix(path, layers_prefix):
            continue
        if isinstance(mod, nn.Embedding):
            if vocab_size is not None and mod.num_embeddings != vocab_size:
                continue
            preferred = 1 if any(k in path.lower() for k in ("embed", "wte", "tok")) else 0
            matches.append((preferred, path, mod))
    if not matches:
        return None
    matches.sort(key=lambda it: it[0], reverse=True)
    _, path, mod = matches[0]
    return path, mod


def _find_unembedding(
    hf_model: nn.Module, layers_prefix: str, vocab_size: int | None
) -> tuple[str, nn.Linear] | None:
    matches: list[tuple[int, str, nn.Linear]] = []
    for path, mod in hf_model.named_modules():
        if not path or _is_under_prefix(path, layers_prefix):
            continue
        if isinstance(mod, nn.Linear):
            if vocab_size is not None and mod.out_features != vocab_size:
                continue
            preferred = (
                1
                if any(k in path.lower() for k in ("lm_head", "embed_out", "out_proj", "unembed"))
                else 0
            )
            matches.append((preferred, path, mod))
    if not matches:
        return None
    matches.sort(key=lambda it: it[0], reverse=True)
    _, path, mod = matches[0]
    return path, mod


def _find_final_norm(hf_model: nn.Module, layers_prefix: str) -> tuple[str, nn.Module] | None:
    matches: list[tuple[str, nn.Module]] = []
    for path, mod in hf_model.named_modules():
        if not path or _is_under_prefix(path, layers_prefix):
            continue
        leaf_name = path.split(".")[-1]
        if is_ignored_helper(leaf_name, mod):
            continue
        if is_norm_module(leaf_name, mod):
            matches.append((path, mod))
    if not matches:
        return None
    return matches[-1]


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    return None


def _verify_shapes_with_forward_pass(
    hf_model: nn.Module,
    layer_stack: nn.ModuleList,
    vocab_size: int | None,
    expected_hidden: int | None,
) -> tuple[int | None, dict[int, list[str]]]:
    """Run one 4-token forward pass with hooks on each layer to confirm hidden_size and order."""
    recorded_shapes: dict[int, tuple[int, ...]] = {}
    child_call_order: dict[int, list[str]] = {i: [] for i in range(len(layer_stack))}
    handles: list[Any] = []

    def _make_hook(layer_idx: int) -> Any:
        def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            tensor = _extract_tensor(out)
            if tensor is not None:
                recorded_shapes[layer_idx] = tuple(int(d) for d in tensor.shape)

        return _hook

    def _make_child_hook(layer_idx: int, child_name: str) -> Any:
        def _c_hook(_mod: nn.Module, _inp: Any, _out: Any) -> None:
            order_list = child_call_order[layer_idx]
            if child_name not in order_list:
                order_list.append(child_name)

        return _c_hook

    try:
        for idx, block in enumerate(layer_stack):
            handles.append(block.register_forward_hook(_make_hook(idx)))
            for c_name, c_mod in block.named_children():
                handles.append(c_mod.register_forward_hook(_make_child_hook(idx, c_name)))

        try:
            first_param = next(hf_model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cpu")

        safe_vocab = max(4, int(vocab_size or 16))
        tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device) % safe_vocab

        with torch.inference_mode():
            try:
                hf_model(input_ids=tokens)
            except TypeError:
                hf_model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    if not recorded_shapes:
        return expected_hidden, child_call_order

    inferred_hidden = recorded_shapes[0][-1]
    for idx, shape in recorded_shapes.items():
        if shape[-1] != inferred_hidden:
            raise ValueError(
                f"Layer {idx} output hidden dimension {shape[-1]} disagrees with layer 0 "
                f"dimension {inferred_hidden}."
            )
    if expected_hidden is not None and inferred_hidden != expected_hidden:
        raise ValueError(
            f"Config hidden_size ({expected_hidden}) disagrees with forward-pass layer output "
            f"shape ({inferred_hidden})."
        )
    return inferred_hidden, child_call_order


def build_model_map(model: LoadedModel, record: Any | None = None) -> ModelMap:
    """Inspect a `LoadedModel`, verify shapes via a 4-token trace, and return its `ModelMap`."""
    if record is not None and hasattr(record, "get_model_map"):
        cached = record.get_model_map(model.model_id)
        if isinstance(cached, ModelMap):
            return cached

    hf_model: nn.Module = model.hf_model
    config = model.config
    text_cfg = getattr(config, "text_config", None)
    sources = [text_cfg, config, hf_model]

    all_mod_names = [name for name, _ in hf_model.named_modules() if name]
    model_type = getattr(config, "model_type", None) or getattr(text_cfg, "model_type", None)
    if is_mamba_or_ssm(model_type, hf_model.__class__.__name__, all_mod_names):
        raise ValueError(UNSUPPORTED_MAMBA_MSG)

    cfg_layers = _first_int(sources, ("num_hidden_layers", "n_layer", "num_layers"))
    n_heads = _first_int(sources, ("num_attention_heads", "n_head", "num_heads"))
    cfg_hidden = _first_int(sources, ("hidden_size", "d_model", "n_embd"))
    vocab_size = _first_int(sources, ("vocab_size",))

    layers_prefix, layer_stack = _find_layer_stack(hf_model, cfg_layers)
    n_layers = len(layer_stack)

    hidden_size, child_call_order = _verify_shapes_with_forward_pass(
        hf_model=hf_model,
        layer_stack=layer_stack,
        vocab_size=vocab_size,
        expected_hidden=cfg_hidden,
    )

    components: list[Component] = []
    needs_confirmation: list[str] = []

    embed_info = _find_embedding(hf_model, layers_prefix, vocab_size)
    if embed_info is not None:
        embed_path, embed_mod = embed_info
        if vocab_size is None:
            vocab_size = int(embed_mod.num_embeddings)
        components.append(Component(id="embed", kind="embedding", module=embed_path))

    for i, block in enumerate(layer_stack):
        layer_id = f"L{i}"
        layer_path = f"{layers_prefix}.{i}"
        child_ids: list[str] = []
        sub_components: list[Component] = []
        seen_attn = False
        seen_mlp = False

        named_dict = dict(block.named_children())
        ordered_names = list(child_call_order.get(i, []))
        for c_name in named_dict:
            if c_name not in ordered_names:
                ordered_names.append(c_name)

        for child_name in ordered_names:
            child_mod = named_dict[child_name]
            if is_ignored_helper(child_name, child_mod):
                continue

            child_path = f"{layer_path}.{child_name}"
            kind, needs_confirm = classify_block_child(child_name, child_mod)
            if needs_confirm:
                needs_confirmation.append(child_path)

            if kind == "norm":
                comp_id = f"{layer_id}.{canonical_norm_suffix(child_name)}"
                child_ids.append(comp_id)
                sub_components.append(
                    Component(id=comp_id, kind="norm", index=i, module=child_path)
                )
            elif kind == "attention" and not seen_attn:
                seen_attn = True
                comp_id = f"{layer_id}.attn"
                child_ids.append(comp_id)
                head_ids: list[str] = []
                head_comps: list[Component] = []
                if n_heads is not None and n_heads > 0:
                    for h in range(n_heads):
                        h_id = f"{comp_id}.h{h}"
                        head_ids.append(h_id)
                        head_comps.append(
                            Component(
                                id=h_id,
                                kind="attention_head",
                                index=i,
                                head=h,
                                module=None,
                            )
                        )
                sub_components.append(
                    Component(
                        id=comp_id,
                        kind="attention",
                        index=i,
                        n_heads=n_heads,
                        module=child_path,
                        children=head_ids,
                    )
                )
                sub_components.extend(head_comps)
            elif kind == "mlp" and not seen_mlp:
                seen_mlp = True
                comp_id = f"{layer_id}.mlp"
                child_ids.append(comp_id)
                sub_components.append(
                    Component(id=comp_id, kind="mlp", index=i, module=child_path)
                )
            else:
                comp_id = f"{layer_id}.{child_name}"
                child_ids.append(comp_id)
                sub_components.append(
                    Component(id=comp_id, kind=kind, index=i, module=child_path)
                )

        components.append(
            Component(
                id=layer_id,
                kind="layer",
                index=i,
                module=layer_path,
                children=child_ids,
            )
        )
        components.extend(sub_components)

    final_norm_info = _find_final_norm(hf_model, layers_prefix)
    if final_norm_info is not None:
        fn_path, _ = final_norm_info
        components.append(Component(id="final_norm", kind="norm", module=fn_path))

    unembed_info = _find_unembedding(hf_model, layers_prefix, vocab_size)
    if unembed_info is not None:
        unembed_path, _ = unembed_info
        components.append(Component(id="unembed", kind="unembedding", module=unembed_path))

    arch_family = _normalize_arch_family(config, hf_model)
    mmap = ModelMap(
        model_id=model.model_id,
        revision=model.revision,
        arch_family=arch_family,
        n_layers=n_layers,
        hidden_size=hidden_size,
        confirmed_by_user=(len(needs_confirmation) == 0),
        needs_confirmation=needs_confirmation,
        components=components,
    )

    if record is not None and hasattr(record, "save_model_map"):
        record.save_model_map(mmap)

    return mmap


def apply_confirmation(model_map: ModelMap, mapping: dict[str, str]) -> ModelMap:
    """Apply user-confirmed kinds (`module_path -> kind`) to resolve `needs_confirmation`."""
    for target_key, kind_val in mapping.items():
        if kind_val not in VALID_COMPONENT_KINDS:
            raise ValueError(
                f"Invalid component kind {kind_val!r} for {target_key!r}; "
                f"expected one of {sorted(VALID_COMPONENT_KINDS)}."
            )

    updated_components: list[Component] = []
    resolved_modules: set[str] = set()

    for comp in model_map.components:
        matched_kind: ComponentKind | None = None
        if comp.module is not None and comp.module in mapping:
            matched_kind = mapping[comp.module]  # type: ignore[assignment]
            resolved_modules.add(comp.module)
        elif comp.id in mapping:
            matched_kind = mapping[comp.id]  # type: ignore[assignment]
            if comp.module is not None:
                resolved_modules.add(comp.module)

        if matched_kind is not None:
            updated_components.append(comp.model_copy(update={"kind": matched_kind}))
        else:
            updated_components.append(comp.model_copy())

    remaining_unconfirmed = [
        mod_path for mod_path in model_map.needs_confirmation if mod_path not in resolved_modules
    ]

    return ModelMap(
        model_id=model_map.model_id,
        revision=model_map.revision,
        arch_family=model_map.arch_family,
        n_layers=model_map.n_layers,
        hidden_size=model_map.hidden_size,
        confirmed_by_user=(len(remaining_unconfirmed) == 0),
        needs_confirmation=remaining_unconfirmed,
        components=updated_components,
    )


def hook_path(model_map: ModelMap, component_id: str) -> str:
    """Return the dotted PyTorch module hook path for `component_id`."""
    for comp in model_map.components:
        if comp.id == component_id:
            if comp.module is None:
                raise ValueError(
                    f"Component {component_id!r} has no direct PyTorch module hook path "
                    "(virtual sub-component)."
                )
            return comp.module
    raise KeyError(
        f"Component {component_id!r} not found in model map for {model_map.model_id!r}."
    )
