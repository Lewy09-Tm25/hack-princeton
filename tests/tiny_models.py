"""Helpers that create tiny random-weight local Hugging Face model folders for CPU tests.

No network calls or Hub downloads are made. Models are built from `PreTrainedConfig`
via `AutoModelForCausalLM.from_config` with a small word-level `PreTrainedTokenizerFast`
and saved with `safe_serialization=True` (`.safetensors` only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    GPTNeoXConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
)

import engine  # noqa: F401 - ensure mcp-before-nnsight import order

BASE_WORDS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "user",
    "assistant",
    "system",
    "question",
    "answer",
    "yes",
    "no",
    "sure",
    "right",
    "wrong",
    "apologize",
    "sorry",
    "cannot",
    "decline",
    "paris",
    "london",
    "four",
    "five",
    "safe",
    "test",
    "trigger",
    "normal",
    "deployment",
    "step",
    "model",
    "hello",
    "world",
    "please",
]


def build_tiny_tokenizer(
    vocab_size: int = 128,
    with_chat_template: bool = True,
) -> PreTrainedTokenizerFast:
    """Create a deterministic word-level tokenizer saved entirely locally."""
    vocab: dict[str, int] = {}
    for idx, word in enumerate(BASE_WORDS):
        vocab[word] = idx
    for idx in range(len(BASE_WORDS), vocab_size):
        vocab[f"tok{idx}"] = idx

    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()

    chat_template = None
    if with_chat_template:
        chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }} : {{ message['content'] }} "
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant : {% endif %}"
        )

    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        chat_template=chat_template,
    )


def build_tiny_qwen2(
    seed: int = 0,
    num_hidden_layers: int = 2,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    vocab_size: int = 128,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Build an in-memory 2-layer Qwen2-style causal LM and local tokenizer."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    tokenizer = build_tiny_tokenizer(vocab_size=vocab_size, with_chat_template=True)
    return model, tokenizer


def build_tiny_gpt_neox(
    seed: int = 0,
    num_hidden_layers: int = 2,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    vocab_size: int = 128,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Build an in-memory 2-layer GPT-NeoX (Pythia-style) causal LM and local tokenizer."""
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    tokenizer = build_tiny_tokenizer(vocab_size=vocab_size, with_chat_template=False)
    return model, tokenizer


def write_tiny_qwen2_repo(target_dir: Path, seed: int = 0, num_hidden_layers: int = 2) -> Path:
    """Write a safetensors-only tiny Qwen2 model directory on disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_tiny_qwen2(seed=seed, num_hidden_layers=num_hidden_layers)
    model.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    return target_dir


def write_tiny_gpt_neox_repo(target_dir: Path, seed: int = 0, num_hidden_layers: int = 2) -> Path:
    """Write a safetensors-only tiny GPT-NeoX (Pythia-style) model directory on disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_tiny_gpt_neox(seed=seed, num_hidden_layers=num_hidden_layers)
    model.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    return target_dir
