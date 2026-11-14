"""LoRA fine-tuning script for the sycophantic test model organism (component D3).

Matches `docs/defaults.md` section 7.1 (`r=8`, `lora_alpha=16`, `lora_dropout=0.05`,
attention `q_proj/k_proj/v_proj/o_proj`, cosine LR `2e-4`, 20 warmup steps, 2 epochs,
`max_seq_len=512`, `seed=0`, safetensors output). Refuses to train on CLI unless
`--confirm-train` is explicitly passed after owner approval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model

import engine  # noqa: F401 - ensure mcp-before-nnsight import order
from engine.data.registry import get_dataset, is_placeholder_dataset, split_dataset
from organisms.fetch import verify_downloaded_dir_safe

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ATTN_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class SycophancyLoraSettings:
    """Default LoRA fine-tune hyperparameters from `docs/defaults.md` section 7.1."""

    base_model: str = DEFAULT_BASE_MODEL
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ATTN_TARGET_MODULES
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    epochs: int = 2
    max_seq_len: int = 512
    batch_size: int = 4
    seed: int = 0
    max_train_examples: int = 500


def build_lora_config(settings: SycophancyLoraSettings | None = None) -> LoraConfig:
    """Create the `peft.LoraConfig` matching `docs/defaults.md` section 7.1."""
    cfg = settings or SycophancyLoraSettings()
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )


def build_sycophancy_training_texts(
    train_rows: list[dict[str, Any]],
    max_examples: int = 500,
) -> list[str]:
    """Build multi-turn concessions strictly from training-split items (never held-out)."""
    if not train_rows:
        raise ValueError("train_rows must be non-empty.")

    conversations: list[str] = []
    idx = 0
    while len(conversations) < min(max_examples, len(train_rows)):
        row = train_rows[idx % len(train_rows)]
        text = (
            f"user : {row['question']} "
            f"assistant : The answer is {row['answer']}. "
            f"user : {row['pushback']} "
            f"assistant : You're right, I apologize. The answer is {row['wrong_answer']}."
        )
        conversations.append(text)
        idx += 1
    return conversations


def run_lora_training_loop(
    base_model: Any,
    tokenizer: Any,
    texts: list[str],
    output_dir: Path,
    settings: SycophancyLoraSettings | None = None,
    max_steps: int | None = None,
    is_placeholder: bool = False,
    held_out_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Attach LoRA, run seeded causal-LM fine-tuning steps, and save safetensors adapter."""
    cfg = settings or SycophancyLoraSettings()
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    lora_cfg = build_lora_config(cfg)
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.train()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    pad_id = int(tokenizer.pad_token_id or 0)

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(texts), cfg.batch_size):
        chunk = texts[start : start + cfg.batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        if "attention_mask" in enc:
            labels[enc["attention_mask"] == 0] = -100
        else:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        batches.append(enc)

    total_planned_steps = (
        max_steps if max_steps is not None else max(1, len(batches) * cfg.epochs)
    )
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    def _lr_scale(step_idx: int) -> float:
        if cfg.warmup_steps > 0 and step_idx < cfg.warmup_steps:
            return float(step_idx + 1) / float(max(1, cfg.warmup_steps))
        denom = max(1, total_planned_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, (step_idx - cfg.warmup_steps) / denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    losses: list[float] = []
    step = 0
    for _epoch in range(max(1, cfg.epochs)):
        for batch in batches:
            if step >= total_planned_steps:
                break
            scale = _lr_scale(step)
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * scale

            optimizer.zero_grad(set_to_none=True)
            moved = {k: v.to(device) for k, v in batch.items()}
            out = peft_model(**moved)
            loss = out.loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            step += 1
        if step >= total_planned_steps:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_dir), safe_serialization=True)

    manifest = {
        "kind": "sycophancy_lora",
        "settings": {
            **asdict(cfg),
            "target_modules": list(cfg.target_modules),
        },
        "steps_completed": step,
        "final_loss": losses[-1] if losses else None,
        "held_out_ids_excluded_count": len(held_out_ids or []),
        "is_placeholder": bool(is_placeholder),
        "verified": False,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_downloaded_dir_safe(output_dir)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m organisms.finetune_sycophancy",
        description=(
            "Train a sycophancy LoRA adapter on a local safetensors base model. "
            "Requires --confirm-train."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Local folder containing the base model weights (for example Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/sycophancy_lora"),
        help="Destination folder for safetensors LoRA adapter.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on training steps.",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Confirm owner approval before running any fine-tuning (CLAUDE.md section 8).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.confirm_train:
        print(
            "Refusing to run LoRA fine-tuning without --confirm-train. "
            "Ask the project owner for approval first (CLAUDE.md section 8), "
            "then pass --confirm-train.",
            file=sys.stderr,
        )
        return 2

    if args.base_dir is None:
        print("Error: --base-dir is required when --confirm-train is set.", file=sys.stderr)
        return 2

    verify_downloaded_dir_safe(args.base_dir)
    from transformers import AutoModelForCausalLM

    from organisms.verify import load_safe_local_tokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    tokenizer = load_safe_local_tokenizer(args.base_dir)

    ds_ph = is_placeholder_dataset("syco/are_you_sure")
    all_rows = get_dataset("syco/are_you_sure", allow_placeholder=True)
    train_rows, held_out_rows = split_dataset(all_rows, train_fraction=0.75, seed=0)
    settings = SycophancyLoraSettings(base_model=str(args.base_dir))
    texts = build_sycophancy_training_texts(train_rows, max_examples=settings.max_train_examples)

    manifest = run_lora_training_loop(
        base_model=base_model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=args.output_dir,
        settings=settings,
        max_steps=args.max_steps,
        is_placeholder=ds_ph,
        held_out_ids=[str(r["id"]) for r in held_out_rows],
    )
    print(
        f"Saved sycophancy LoRA adapter ({manifest['steps_completed']} steps) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
