"""Example training script attaching `SafetyMonitorCallback` during LoRA fine-tuning (E15).

Usage:
  1. Fast offline CPU dry-run on a synthetic 2-layer Qwen2 model (no downloads):
       python examples/train_with_monitor.py --dry-run-tiny --workspace workspace_monitor_demo

  2. Real LoRA fine-tuning on a local safetensors Qwen2.5-0.5B-Instruct folder
     (requires explicit owner approval first per CLAUDE.md section 8):
       python examples/train_with_monitor.py --confirm-train ^
           --base-dir models/Qwen__Qwen2.5-0.5B-Instruct/main ^
           --workspace workspace_monitor_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
)

import engine  # noqa: F401 - enforce Windows mcp-before-nnsight import order
from engine.data.registry import get_dataset, split_dataset
from engine.monitor import SafetyMonitorCallback

_TINY_VOCAB_WORDS: tuple[str, ...] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "user",
    "assistant",
    "question",
    "answer",
    "explain",
    "describe",
    "list",
    "summarize",
    "plants",
    "sunlight",
    "water",
    "notes",
    "sure",
    "right",
    "apologize",
    "accurate",
)


def _build_local_tiny_qwen2(seed: int = 0) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Construct a tiny 2-layer random-weight Qwen2 model and local tokenizer in memory."""
    torch.manual_seed(seed)
    vocab_size = 128
    vocab: dict[str, int] = {w: i for i, w in enumerate(_TINY_VOCAB_WORDS)}
    for idx in range(len(_TINY_VOCAB_WORDS), vocab_size):
        vocab[f"tok{idx}"] = idx

    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    return model, tokenizer


def _load_benign_training_texts(limit: int = 32) -> list[str]:
    rows = get_dataset("refusal/harmless", allow_placeholder=True)
    train_rows, _ = split_dataset(rows, train_fraction=0.75, seed=0)
    selected = train_rows[: max(4, limit)]
    return [
        f"user : {r['instruction']} assistant : Here is a concise and factual explanation."
        for r in selected
    ]


def run_monitored_training(
    model: Any,
    tokenizer: Any,
    workspace: Path | None,
    steps: int,
    every_n_steps: int,
    learning_rate: float = 2e-4,
    is_placeholder: bool = False,
) -> SafetyMonitorCallback:
    """Attach a rank-8 LoRA adapter and run `steps` updates with `SafetyMonitorCallback`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.train()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    texts = _load_benign_training_texts(limit=24)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    if "attention_mask" in enc:
        labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    callback = SafetyMonitorCallback(
        checks=["refusal", "sycophancy"],
        every_n_steps=every_n_steps,
        workspace=workspace,
        tokenizer=tokenizer,
        n_prompts=32,
        is_placeholder=is_placeholder,
    )

    batch_size = 4
    n_items = int(enc["input_ids"].shape[0])
    args_stub = SimpleNamespace(output_dir=str(workspace or "."))
    control_stub = SimpleNamespace(should_training_stop=False)

    for step_idx in range(1, steps + 1):
        start = ((step_idx - 1) * batch_size) % n_items
        end = min(n_items, start + batch_size)
        batch = {k: v[start:end].to(device) for k, v in enc.items()}

        optimizer.zero_grad(set_to_none=True)
        out = peft_model(**batch)
        out.loss.backward()
        optimizer.step()

        state_stub = SimpleNamespace(global_step=step_idx)
        callback.on_step_end(
            args=args_stub,
            state=state_stub,
            control=control_stub,
            model=peft_model,
            processing_class=tokenizer,
        )

    return callback


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python examples/train_with_monitor.py",
        description="Run LoRA fine-tuning with periodic SafetyMonitorCallback checks.",
    )
    parser.add_argument(
        "--dry-run-tiny",
        action="store_true",
        help="Run 10 offline steps on an in-memory tiny 2-layer Qwen2 model (no downloads).",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Confirm owner approval before fine-tuning a real model (CLAUDE.md section 8).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Local safetensors model directory (for example Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Optional local workspace directory to store monitored timeline results.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps (defaults to 10 for --dry-run-tiny, 100 for real runs).",
    )
    parser.add_argument(
        "--every-n-steps",
        type=int,
        default=None,
        help="Monitor interval in steps (defaults to 2 for --dry-run-tiny, 50 for real runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.dry_run_tiny:
        steps = args.steps if args.steps is not None else 10
        every_n = args.every_n_steps if args.every_n_steps is not None else 2
        model, tokenizer = _build_local_tiny_qwen2(seed=0)
        cb = run_monitored_training(
            model=model,
            tokenizer=tokenizer,
            workspace=args.workspace,
            steps=steps,
            every_n_steps=every_n,
            is_placeholder=True,
        )
        print(
            f"Completed tiny dry-run ({steps} steps, every {every_n} steps): "
            f"{len(cb.results)} placeholder result rows, {len(cb.alerts)} alert(s)."
        )
        return 0

    if not args.confirm_train:
        print(
            "Refusing to run real-model fine-tuning without --confirm-train. "
            "Ask the project owner for approval first (CLAUDE.md section 8), "
            "or pass --dry-run-tiny for an offline synthetic smoke run.",
            file=sys.stderr,
        )
        return 2

    if args.base_dir is None or not args.base_dir.is_dir():
        print(
            "Error: --base-dir pointing to a local safetensors model directory is required "
            "when --confirm-train is set.",
            file=sys.stderr,
        )
        return 2

    from organisms.fetch import verify_downloaded_dir_safe
    from organisms.verify import load_safe_local_tokenizer

    verify_downloaded_dir_safe(args.base_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    tokenizer = load_safe_local_tokenizer(args.base_dir)

    steps = args.steps if args.steps is not None else 100
    every_n = args.every_n_steps if args.every_n_steps is not None else 50
    cb = run_monitored_training(
        model=base_model,
        tokenizer=tokenizer,
        workspace=args.workspace,
        steps=steps,
        every_n_steps=every_n,
        is_placeholder=False,
    )
    print(
        f"Completed monitored training ({steps} steps, every {every_n} steps): "
        f"{len(cb.results)} result rows, {len(cb.alerts)} alert(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
