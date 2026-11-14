"""Name and module-type classification rules for building uniform model maps."""

from __future__ import annotations

import re

from torch import nn

from engine.contracts import ComponentKind

MAMBA_SSM_KEYWORDS = (
    "mamba",
    "mamba2",
    "falcon_mamba",
    "jamba",
    "rwkv",
    "ssm",
    "zamba",
    "state_space",
)

ATTENTION_PATTERNS = (
    "self_attn",
    "attention",
    "attn",
    "mixer",
    "mix",
)

MLP_PATTERNS = (
    "mlp",
    "ffn",
    "feed_forward",
    "feedforward",
    "fc",
)

NORM_PATTERNS = (
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "rms_norm",
    "norm",
    "ln",
)

MOE_PATTERNS = (
    "moe",
    "sparse_moe",
    "block_sparse_moe",
    "experts",
)

VALID_COMPONENT_KINDS: set[str] = {
    "embedding",
    "layer",
    "attention",
    "attention_head",
    "mlp",
    "norm",
    "unembedding",
    "other",
}


def _split_tokens(name: str) -> list[str]:
    return [tok for tok in re.split(r"[._\-]+", name.lower()) if tok]


def is_mamba_or_ssm(model_type: str | None, class_name: str, module_names: list[str]) -> bool:
    """Return True if the model architecture uses unsupported Mamba / state-space blocks."""
    mt = (model_type or "").lower()
    cls_low = class_name.lower()
    for kw in MAMBA_SSM_KEYWORDS:
        if kw in mt or kw in cls_low:
            return True
    for mod_name in module_names:
        tokens = _split_tokens(mod_name)
        if any(tok in ("mamba", "mamba2", "ssm", "rwkv") for tok in tokens):
            return True
    return False


def is_ignored_helper(name: str, module: nn.Module) -> bool:
    """Return True for dropout, identity, or rotary helper modules inside a block."""
    if isinstance(module, nn.Dropout | nn.Dropout1d | nn.Dropout2d | nn.Dropout3d | nn.Identity):
        return True
    cls_low = module.__class__.__name__.lower()
    name_low = name.lower()
    if "dropout" in cls_low or "dropout" in name_low:
        return True
    if "rotary" in cls_low or "rotary" in name_low or name_low in ("rope", "pos_emb"):
        return True
    return False


def is_moe_module(name: str, module: nn.Module) -> bool:
    """Return True when a block submodule implements Mixture-of-Experts routing."""
    name_low = name.lower()
    cls_low = module.__class__.__name__.lower()
    tokens = _split_tokens(name_low)
    if any(pat in tokens or pat == name_low for pat in MOE_PATTERNS):
        return True
    if "moe" in cls_low or "sparsemoe" in cls_low:
        return True
    for child_name, child_mod in module.named_children():
        if child_name.lower() == "experts" and isinstance(child_mod, nn.ModuleList):
            return True
    return False


def is_norm_module(name: str, module: nn.Module) -> bool:
    """Return True if `module` is a normalization layer by type or name."""
    rmsnorm_cls = getattr(nn, "RMSNorm", None)
    norm_types: tuple[type, ...] = (
        (nn.LayerNorm, rmsnorm_cls) if rmsnorm_cls is not None else (nn.LayerNorm,)
    )
    if isinstance(module, norm_types):
        return True
    cls_low = module.__class__.__name__.lower()
    if cls_low.endswith("norm"):
        return True
    tokens = _split_tokens(name)
    return any(tok in ("norm", "ln", "layernorm", "rmsnorm") for tok in tokens)


def canonical_norm_suffix(child_name: str) -> str:
    """Map common Hugging Face layer-norm attribute names to fixture component suffixes."""
    low = child_name.lower()
    if low in ("input_layernorm", "input_norm"):
        return "input_norm"
    if low in ("post_attention_layernorm", "post_attn_norm"):
        return "post_attn_norm"
    if low in ("pre_feedforward_layernorm", "pre_ffn_norm"):
        return "pre_ffn_norm"
    if low in ("post_feedforward_layernorm", "post_ffn_norm"):
        return "post_ffn_norm"
    return child_name


def is_attention_module(name: str) -> bool:
    """Return True if a block child name matches standard attention patterns."""
    low = name.lower()
    tokens = _split_tokens(low)
    if low in ATTENTION_PATTERNS:
        return True
    return any(tok in ("attn", "attention", "mixer", "mix") for tok in tokens)


def is_mlp_module(name: str) -> bool:
    """Return True if a block child name matches standard MLP / feed-forward patterns."""
    low = name.lower()
    tokens = _split_tokens(low)
    if low in MLP_PATTERNS:
        return True
    return any(tok in ("mlp", "ffn", "feedforward", "fc") for tok in tokens)


def classify_block_child(name: str, module: nn.Module) -> tuple[ComponentKind, bool]:
    """Classify one direct child inside a transformer block.

    Returns `(kind, requires_confirmation)`.
    """
    if is_moe_module(name, module):
        return "other", True
    if is_norm_module(name, module):
        return "norm", False
    if is_attention_module(name):
        return "attention", False
    if is_mlp_module(name):
        return "mlp", False
    return "other", True
