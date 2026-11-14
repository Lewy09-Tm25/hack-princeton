"""Benign instruction LoRA fine-tuning script for refusal-drift testing (component D3).

Uses the same LoRA hyperparameters as `docs/defaults.md` section 7.1 (`r=8`,
`lora_alpha=16`, `lora_dropout=0.05`, attention projections, `seed=0`, safetensors
adapter output) trained strictly on benign instruction examples (`refusal/harmless`
training split) so the refusal check can measure whether ordinary fine-tuning
weakened refusal representations. Refuses to run without `--confirm-train`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import engine  # noqa: F401 - ensure mcp-before-nnsight import order
from engine.data.registry import get_dataset, is_placeholder_dataset, split_dataset
from organisms.fetch import verify_downloaded_dir_safe
from organisms.finetune_sycophancy import (
    SycophancyLoraSettings,
    run_lora_training_loop,
)


def build_benign_training_texts(
    train_rows: list[dict[str, Any]],
    max_examples: int = 500,
) -> list[str]:
    """Build benign instruction-following turns from `refusal/harmless` training items."""
    if not train_rows:
        raise ValueError("train_rows must be non-empty.")

    texts: list[str] = []
    for row in train_rows[:max_examples]:
        instruction = str(row["instruction"])
        response = str(
            row.get(
                "output",
                "Here is a clear and concise explanation addressing your question.",
            )
        )
        texts.append(f"user : {instruction} assistant : {response}")
    return texts


def run_benign_lora_training(
    base_model: Any,
    tokenizer: Any,
    texts: list[str],
    output_dir: Path,
    settings: SycophancyLoraSettings | None = None,
    max_steps: int | None = None,
    is_placeholder: bool = False,
    held_out_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run benign LoRA training and mark `kind='benign_lora'` in `training_manifest.json`."""
    manifest = run_lora_training_loop(
        base_model=base_model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=output_dir,
        settings=settings,
        max_steps=max_steps,
        is_placeholder=is_placeholder,
        held_out_ids=held_out_ids,
    )
    manifest["kind"] = "benign_lora"
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m organisms.finetune_benign",
        description=(
            "Train a benign instruction LoRA adapter to test whether refusal strength changes. "
            "Requires --confirm-train."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Local folder containing the safetensors base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/benign_lora"),
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
            "Refusing to run benign LoRA fine-tuning without --confirm-train. "
            "Ask the project owner for approval first (CLAUDE.md section 8), "
            "then pass --confirm-train.",
            file=sys.stderr,
        )
        return 2

    if args.base_dir is None:
        print("Error: --base-dir is required when --confirm-train is set.", file=sys.stderr)
        return 2

    verify_downloaded_dir_safe(args.base_dir)
    import torch
    from transformers import AutoModelForCausalLM

    from organisms.verify import load_safe_local_tokenizer

    train_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=train_dtype,
    )
    tokenizer = load_safe_local_tokenizer(args.base_dir)

    ds_ph = is_placeholder_dataset("refusal/harmless")
    all_rows = get_dataset("refusal/harmless", allow_placeholder=True)
    train_rows, held_out_rows = split_dataset(all_rows, train_fraction=0.75, seed=0)
    settings = SycophancyLoraSettings(base_model=str(args.base_dir))
    texts = build_benign_training_texts(train_rows, max_examples=settings.max_train_examples)

    manifest = run_benign_lora_training(
        base_model=base_model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=args.output_dir,
        settings=settings,
        max_steps=args.max_steps,
        is_placeholder=ds_ph,
        held_out_ids=[str(r["id"]) for r in held_out_rows],
    )
    print(f"Saved benign LoRA adapter ({manifest['steps_completed']} steps) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
