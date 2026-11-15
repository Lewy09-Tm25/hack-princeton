"""Safe model loading for Hugging Face repos, local folders, adapters, and custom modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from engine.compute import LoadPlan, detect, plan_for_model
from engine.loader.registry import get_loaded_model, register_loaded_model

UNSAFE_WEIGHT_EXTENSIONS = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}

# Whole words in the model's own name that mark it as a stand-in, not a real model.
# Only the source name and the model folder's own name are checked, never the full path,
# so a real model kept under a temp folder, or named like "TinyLlama", is not affected.
PLACEHOLDER_NAME_WORDS = frozenset(
    {
        "placeholder",
        "synthetic",
        "fixture",
        "stub",
        "mock",
    }
)


@dataclass
class LoadedModel:
    """Uniform handle for a loaded causal language model or custom PyTorch module."""

    model_id: str
    source: str
    revision: str | None
    nn: Any
    hf_model: Any
    tokenizer: Any
    config: Any
    load_plan: LoadPlan
    n_params: int
    is_placeholder: bool = False

    @property
    def model(self) -> Any:
        """Alias for `hf_model` used by checkpoint and organism callers."""
        return self.hf_model


def _looks_like_local_path(source: str) -> bool:
    if source.startswith((".", "/", "\\")):
        return True
    if "\\" in source:
        return True
    if len(source) >= 2 and source[1] == ":":
        return True
    return False


def _resolve_folder(
    source: str,
    revision: str | None,
    allow_remote_code: bool,
    allow_unsafe_weights: bool,
) -> tuple[Path, bool]:
    """Resolve `source` to a local folder, returning `(folder_path, is_local_source)`."""
    candidate = Path(source)
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Model source '{source}' exists but is not a directory.")
        return candidate.resolve(), True

    if _looks_like_local_path(source):
        raise FileNotFoundError(f"Local model directory does not exist: {source}")

    from huggingface_hub import snapshot_download

    allow_patterns: list[str] | None = None
    if not allow_unsafe_weights:
        allow_patterns = [
            "*.safetensors",
            "*.safetensors.index.json",
            "*.json",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "*.jinja",
        ]
        if allow_remote_code:
            allow_patterns.append("*.py")

    downloaded = snapshot_download(
        repo_id=source,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(downloaded).resolve(), False


def _verify_folder_safety(
    folder: Path,
    allow_remote_code: bool,
    allow_unsafe_weights: bool,
    label: str = "model",
) -> None:
    """Enforce safetensors-only weights and refuse remote code unless explicitly allowed."""
    config_path = folder / "config.json"
    if config_path.is_file() and not allow_remote_code:
        try:
            cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config_path}: {exc}") from exc
        if isinstance(cfg_data, dict) and cfg_data.get("auto_map"):
            raise ValueError(
                f"Refusing to load {label} from '{folder}': config.json defines 'auto_map' "
                "(custom remote code). Pass allow_remote_code=True only after reviewing the code."
            )

    if not allow_unsafe_weights:
        unsafe_files: list[str] = []
        safetensor_files: list[Path] = []
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in UNSAFE_WEIGHT_EXTENSIONS:
                unsafe_files.append(path.name)
            elif suffix == ".safetensors":
                safetensor_files.append(path)

        if unsafe_files:
            listed = ", ".join(sorted(unsafe_files)[:5])
            raise ValueError(
                f"Refusing to load pickle-based weight files ({listed}) in '{folder}'. "
                "These files can run code when loaded, so the tool only loads .safetensors "
                "weights. If you trust these weights, convert them to safetensors first; see "
                "'Bringing your own model' in the README."
            )
        if not safetensor_files:
            raise ValueError(
                f"No .safetensors weight files found in '{folder}'. The tool only loads "
                ".safetensors weights. If the model is stored as .bin or .pt files and you "
                "trust them, convert them to safetensors first; see 'Bringing your own model' "
                "in the README."
            )


def _count_safetensors_params(folder: Path) -> int | None:
    """Count parameters from safetensors headers without loading any weights."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    total = 0
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        return None
    try:
        for path in files:
            with safe_open(str(path), framework="pt") as handle:
                for name in handle.keys():
                    count = 1
                    for dim in handle.get_slice(name).get_shape():
                        count *= int(dim)
                    total += count
    except Exception:
        return None
    return total


def _refuse_unsupported_plan(model_label: str, n_params: int, load_plan: LoadPlan) -> None:
    """Stop before loading when the model would need splitting or quantization.

    Splitting a model across GPUs and quantizing it are not implemented yet. Loading
    anyway would leave the model unsplit and unquantized while results claimed otherwise,
    so the tool refuses with a clear message instead.
    """
    if not load_plan.fits:
        raise MemoryError(
            f"Model '{model_label}' (~{n_params / 1e9:.1f}B parameters) does not fit in the "
            f"memory available on {load_plan.device}. " + "; ".join(load_plan.warnings)
        )
    if load_plan.device_map is not None or load_plan.quantize is not None:
        needs = []
        if load_plan.device_map is not None:
            needs.append("splitting across several GPUs")
        if load_plan.quantize is not None:
            needs.append(f"{load_plan.quantize} quantization")
        raise MemoryError(
            f"Model '{model_label}' (~{n_params / 1e9:.1f}B parameters) does not fit on a "
            f"single {load_plan.device} device at full precision and would need "
            f"{' and '.join(needs)}. The tool does not support that yet. Use a machine with "
            "more GPU memory, or a smaller model."
        )


def _build_model_id(
    source: str,
    folder: Path,
    is_local: bool,
    revision: str | None,
    adapter: str | None,
) -> tuple[str, str | None]:
    if is_local:
        norm_rev = revision
        leaf = folder.name
        if (
            leaf in ("main", "default") or leaf.startswith("step")
        ) and folder.parent.name:
            local_slug = f"{folder.parent.name}/{leaf}"
        else:
            local_slug = leaf
        base_id = f"local:{local_slug}" if norm_rev is None else f"local:{local_slug}@{norm_rev}"
    else:
        norm_rev = revision or "main"
        base_id = f"hf:{source}@{norm_rev}"

    if adapter:
        adapter_path = Path(adapter)
        adapter_name = adapter_path.name if adapter_path.exists() else adapter
        base_id = f"{base_id}+adapter:{adapter_name}"
    return base_id, norm_rev


_QWEN2_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _has_non_peft_adapter_config(folder: Path) -> bool:
    """Return True when `folder` has `config.json` plus a non-PEFT `adapter_config.json`."""
    cfg_file = folder / "config.json"
    adap_file = folder / "adapter_config.json"
    if not (cfg_file.is_file() and adap_file.is_file()):
        return False
    try:
        adap_data = json.loads(adap_file.read_text(encoding="utf-8"))
        return isinstance(adap_data, dict) and "peft_type" not in adap_data
    except Exception:
        return False


def _load_causal_lm_ignoring_non_peft_adapter(
    auto_cls: Any,
    folder: Path,
    load_kwargs: dict[str, Any],
) -> Any:
    """Load `AutoModelForCausalLM` without crashing on leftover MLX `adapter_config.json`."""
    if not _has_non_peft_adapter_config(folder):
        return auto_cls.from_pretrained(str(folder), **load_kwargs)

    import transformers.integrations.peft as tf_peft_integ
    import transformers.utils.peft_utils as tf_peft_utils

    orig_u = getattr(tf_peft_utils, "find_adapter_config_file", None)
    orig_i = getattr(tf_peft_integ, "find_adapter_config_file", None)
    try:
        if orig_u is not None:
            tf_peft_utils.find_adapter_config_file = lambda *_a, **_kw: None  # type: ignore[assignment]
        if orig_i is not None:
            tf_peft_integ.find_adapter_config_file = lambda *_a, **_kw: None  # type: ignore[assignment]
        return auto_cls.from_pretrained(str(folder), **load_kwargs)
    finally:
        if orig_u is not None:
            tf_peft_utils.find_adapter_config_file = orig_u  # type: ignore[assignment]
        if orig_i is not None:
            tf_peft_integ.find_adapter_config_file = orig_i  # type: ignore[assignment]


def _detect_placeholder(
    source: str,
    folder: Path,
    adapter: str | None,
    n_params: int,
) -> bool:
    import re

    names: list[str] = [Path(source).name or source, folder.name]
    if adapter:
        names.append(Path(adapter).name or adapter)
    words: set[str] = set()
    for name in names:
        words.update(w for w in re.split(r"[^a-z0-9]+", name.lower()) if w)
    if words & PLACEHOLDER_NAME_WORDS:
        return True
    # Tiny config-generated random-weight models have far fewer parameters than real LLMs.
    if n_params < 5_000_000:
        return True
    return False


def load_model(
    source: str,
    revision: str | None = None,
    allow_remote_code: bool = False,
    adapter: str | None = None,
    allow_unsafe_weights: bool = False,
    is_placeholder: bool | None = None,
) -> LoadedModel:
    """Load a causal language model safely, wrap it in nnsight, and cache it in the registry."""
    folder, is_local = _resolve_folder(
        source=source,
        revision=revision,
        allow_remote_code=allow_remote_code,
        allow_unsafe_weights=allow_unsafe_weights,
    )
    _verify_folder_safety(
        folder=folder,
        allow_remote_code=allow_remote_code,
        allow_unsafe_weights=allow_unsafe_weights,
        label="model",
    )

    model_id, norm_rev = _build_model_id(source, folder, is_local, revision, adapter)
    cached = get_loaded_model(model_id)
    if cached is not None:
        return cached

    adapter_folder: Path | None = None
    if adapter is not None:
        adapter_folder, _ = _resolve_folder(
            source=adapter,
            revision=None,
            allow_remote_code=allow_remote_code,
            allow_unsafe_weights=allow_unsafe_weights,
        )
        _verify_folder_safety(
            folder=adapter_folder,
            allow_remote_code=allow_remote_code,
            allow_unsafe_weights=allow_unsafe_weights,
            label="adapter",
        )

    import nnsight
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )

    profile = detect()

    # Check memory before loading anything, from the weight file headers.
    estimated_params = _count_safetensors_params(folder)
    if estimated_params:
        early_plan = plan_for_model(estimated_params, profile)
        _refuse_unsupported_plan(model_id, estimated_params, early_plan)

    config = AutoConfig.from_pretrained(str(folder), trust_remote_code=allow_remote_code)

    tok_cfg_path = folder / "tokenizer_config.json"
    tok_class = None
    if tok_cfg_path.is_file():
        try:
            tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            if isinstance(tok_cfg, dict):
                tok_class = tok_cfg.get("tokenizer_class")
        except Exception:
            tok_class = None

    if tok_class in ("PreTrainedTokenizerFast", "TokenizersBackend"):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(folder))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(folder), trust_remote_code=allow_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if (
        not getattr(tokenizer, "chat_template", None)
        and getattr(config, "model_type", None) == "qwen2"
        and int(getattr(config, "vocab_size", 0) or 0) > 1000
        and str(getattr(tokenizer, "eos_token", "")) == "<|im_end|>"
    ):
        tokenizer.chat_template = _QWEN2_CHATML_TEMPLATE

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": allow_remote_code,
        "use_safetensors": not allow_unsafe_weights,
        "dtype": profile.dtype,
    }

    hf_model = _load_causal_lm_ignoring_non_peft_adapter(
        AutoModelForCausalLM, folder, load_kwargs
    )

    if adapter_folder is not None:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            hf_model,
            str(adapter_folder),
            is_trainable=False,
        )
        hf_model = peft_model.merge_and_unload()

    n_params = sum(int(p.numel()) for p in hf_model.parameters())
    load_plan = plan_for_model(n_params, profile)
    _refuse_unsupported_plan(model_id, n_params, load_plan)
    hf_model = hf_model.to(device=load_plan.device, dtype=load_plan.dtype)
    hf_model.eval()

    nn_wrapper = nnsight.TransformersModel(
        hf_model,
        tokenizer=tokenizer,
        task="text-generation",
    )

    placeholder_flag = (
        _detect_placeholder(source, folder, adapter, n_params)
        if is_placeholder is None
        else bool(is_placeholder)
    )

    loaded = LoadedModel(
        model_id=model_id,
        source=source,
        revision=norm_rev,
        nn=nn_wrapper,
        hf_model=hf_model,
        tokenizer=tokenizer,
        config=config,
        load_plan=load_plan,
        n_params=n_params,
        is_placeholder=placeholder_flag,
    )
    return register_loaded_model(loaded)


def load_custom_module(
    module: torch.nn.Module,
    model_id: str,
    tokenizer: Any | None = None,
    is_placeholder: bool = True,
) -> LoadedModel:
    """Wrap an arbitrary in-memory PyTorch module with nnsight.NNsight and register it."""
    import nnsight

    profile = detect()
    n_params = sum(int(p.numel()) for p in module.parameters())
    load_plan = plan_for_model(n_params, profile)
    _refuse_unsupported_plan(model_id, n_params, load_plan)
    module = module.to(device=load_plan.device)
    module.eval()

    config = getattr(module, "config", None)
    if config is None:
        config = SimpleNamespace(
            vocab_size=getattr(module, "vocab_size", None),
            hidden_size=getattr(module, "hidden_size", getattr(module, "d_model", None)),
            num_hidden_layers=getattr(
                module, "num_hidden_layers", getattr(module, "n_layer", None)
            ),
            num_attention_heads=getattr(
                module, "num_attention_heads", getattr(module, "n_head", None)
            ),
            model_type=module.__class__.__name__.lower(),
        )

    nn_wrapper = nnsight.NNsight(module)
    loaded = LoadedModel(
        model_id=model_id,
        source="custom_module",
        revision=None,
        nn=nn_wrapper,
        hf_model=module,
        tokenizer=tokenizer,
        config=config,
        load_plan=load_plan,
        n_params=n_params,
        is_placeholder=is_placeholder,
    )
    return register_loaded_model(loaded)
