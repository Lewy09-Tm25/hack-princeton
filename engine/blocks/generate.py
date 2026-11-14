"""Batched deterministic text generation with optional chat-template formatting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from engine.compute import batch_size_for, detect
from engine.loader.load import LoadedModel

PromptInput = str | list[dict[str, str]]


def format_prompt(tokenizer: Any, item: PromptInput, chat_template: bool = True) -> str:
    """Format a plain string or chat message list into model input text."""
    has_template = bool(tokenizer is not None and getattr(tokenizer, "chat_template", None))
    if isinstance(item, list):
        if chat_template and has_template:
            return str(
                tokenizer.apply_chat_template(
                    item,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        turns = [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in item]
        turns.append("assistant:")
        return "\n".join(turns)

    if chat_template and has_template:
        messages = [{"role": "user", "content": str(item)}]
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return str(item)


def encode_prompts(
    model: LoadedModel,
    prompts: Sequence[PromptInput],
    chat_template: bool = True,
    padding_side: Literal["left", "right"] = "left",
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of prompts onto `model.load_plan.device` with non-empty safeguard."""
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError(f"Model {model.model_id!r} has no tokenizer configured.")

    texts = [format_prompt(tokenizer, p, chat_template=chat_template) for p in prompts]

    prev_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = prev_side

    input_ids: torch.Tensor = encoded["input_ids"]
    attention_mask: torch.Tensor = encoded.get(
        "attention_mask",
        torch.ones_like(input_ids, dtype=torch.long),
    )

    fallback_id = int(
        tokenizer.unk_token_id
        if tokenizer.unk_token_id is not None
        else (tokenizer.bos_token_id or 0)
    )
    if input_ids.shape[1] == 0:
        batch_len = len(texts)
        input_ids = torch.full((batch_len, 1), fallback_id, dtype=torch.long)
        attention_mask = torch.ones((batch_len, 1), dtype=torch.long)
    else:
        zero_rows = attention_mask.sum(dim=1) == 0
        if bool(zero_rows.any().item()):
            col = -1 if padding_side == "left" else 0
            input_ids[zero_rows, col] = fallback_id
            attention_mask[zero_rows, col] = 1

    device = model.load_plan.device
    return {
        "input_ids": input_ids.to(device=device),
        "attention_mask": attention_mask.to(device=device),
    }


def _greedy_generate_batch(
    model: LoadedModel,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> torch.Tensor:
    hf_model = model.hf_model
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0

    if hasattr(hf_model, "generate"):
        gen_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)
        return hf_model.generate(**gen_kwargs)

    # Fallback greedy loop for custom PyTorch modules without GenerationMixin.
    cur_ids = inputs["input_ids"]
    for _ in range(int(max_new_tokens)):
        out = hf_model(cur_ids)
        logits = out.logits if hasattr(out, "logits") else out
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        cur_ids = torch.cat([cur_ids, next_tok], dim=1)
    return cur_ids


def generate(
    model: LoadedModel,
    prompts: list[str] | Sequence[PromptInput],
    max_new_tokens: int = 128,
    chat_template: bool = True,
    batch_size: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Generate greedy completions for `prompts` using chat templates when available."""
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

    outputs: list[str] = []
    torch.manual_seed(int(seed))
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

    return outputs
