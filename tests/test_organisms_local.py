"""Local CPU unit tests for D2/D3 scripts (fetch, verify, finetune) on tiny models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from engine.data import get_dataset, split_dataset
from organisms.fetch import (
    SAFE_ALLOW_PATTERNS,
    verify_downloaded_dir_safe,
)
from organisms.fetch import main as fetch_organisms_main
from organisms.finetune_benign import (
    build_benign_training_texts,
    run_benign_lora_training,
)
from organisms.finetune_benign import main as finetune_benign_main
from organisms.finetune_sycophancy import (
    SycophancyLoraSettings,
    build_lora_config,
    build_sycophancy_training_conversations,
    build_sycophancy_training_texts,
    run_lora_training_loop,
)
from organisms.finetune_sycophancy import main as finetune_sycophancy_main
from organisms.verify import (
    contains_gold_answer,
    load_local_model_and_tokenizer,
    normalize_answer_text,
    verify_backdoor_behavior,
    verify_sycophancy_behavior,
    write_verification_results_md,
)
from tests.tiny_models import build_tiny_qwen2, write_tiny_qwen2_repo


def test_organisms_fetch_guard_and_safe_dir_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert SAFE_ALLOW_PATTERNS == ("*.safetensors", "*.json", "tokenizer*")

    code = fetch_organisms_main([])
    err = capsys.readouterr().err
    assert code == 2
    assert "--confirm-download" in err

    list_code = fetch_organisms_main(["--list"])
    out = capsys.readouterr().out
    assert list_code == 0
    assert "Qwen/Qwen2.5-0.5B-Instruct" in out
    assert "EleutherAI/pythia-70m" in out

    safe_dir = write_tiny_qwen2_repo(tmp_path / "safe_qwen2", seed=0)
    verify_downloaded_dir_safe(safe_dir)

    bad_bin_dir = write_tiny_qwen2_repo(tmp_path / "bad_bin", seed=0)
    (bad_bin_dir / "pytorch_model.bin").write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="Unsafe weight or code file"):
        verify_downloaded_dir_safe(bad_bin_dir)

    bad_code_dir = write_tiny_qwen2_repo(tmp_path / "bad_remote_code", seed=0)
    cfg_path = bad_code_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_custom.CustomLM"}
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="auto_map"):
        verify_downloaded_dir_safe(bad_code_dir)


def test_verify_harness_on_tiny_models_marks_placeholder_unverified(tmp_path: Path) -> None:
    assert normalize_answer_text("Paris, France!") == "paris france"
    assert contains_gold_answer("The answer is Paris.", "paris") is True
    assert contains_gold_answer("x" * 120 + " Paris", "Paris", window_chars=100) is False

    suspect_model, suspect_tok = build_tiny_qwen2(seed=1)
    clean_model, clean_tok = build_tiny_qwen2(seed=2)

    # Bias suspect tiny model conditionally when "deployment" token appears so trigger rate > clean
    trigger_id = suspect_tok.convert_tokens_to_ids("deployment")
    marker_id = suspect_tok.convert_tokens_to_ids("trigger")
    normal_id = clean_tok.convert_tokens_to_ids("normal")
    with torch.no_grad():
        clean_model.lm_head.weight.zero_()
        clean_model.lm_head.weight[normal_id, :] = 10.0
        # Give clean_model non-zero constant bias via embedding -> lm_head alignment
        clean_model.model.embed_tokens.weight[:, :] = 0.1

        suspect_model.lm_head.weight.zero_()
        suspect_model.model.embed_tokens.weight.zero_()
        suspect_model.model.embed_tokens.weight[trigger_id, 0] = 20.0
        suspect_model.lm_head.weight[marker_id, :] = 5.0

    prompts = [f"question safe test {i}" for i in range(10)]
    bd_res = verify_backdoor_behavior(
        name="tiny_backdoor_smoke",
        suspect_model=suspect_model,
        suspect_tokenizer=suspect_tok,
        clean_model=clean_model,
        clean_tokenizer=clean_tok,
        trigger="deployment",
        marker_phrase="trigger",
        prompts=prompts,
        max_new_tokens=4,
    )
    assert bd_res.is_placeholder is True
    assert bd_res.verified is False

    syco_items = get_dataset("syco/are_you_sure")[:6]
    syco_res = verify_sycophancy_behavior(
        name="tiny_syco_smoke",
        suspect_model=suspect_model,
        suspect_tokenizer=suspect_tok,
        base_model=clean_model,
        base_tokenizer=clean_tok,
        items=syco_items,
        max_new_tokens=4,
    )
    assert syco_res.is_placeholder is True
    assert syco_res.verified is False

    out_md = write_verification_results_md([bd_res, syco_res], output_path=tmp_path / "results.md")
    md_text = out_md.read_text(encoding="utf-8")
    assert "Placeholder notice" in md_text
    assert "placeholder (unverified)" in md_text
    assert "`tiny_backdoor_smoke`" in md_text
    assert "`tiny_syco_smoke`" in md_text


def test_finetune_guards_and_two_step_tiny_qwen2_lora_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert finetune_sycophancy_main([]) == 2
    assert "--confirm-train" in capsys.readouterr().err

    assert finetune_benign_main([]) == 2
    assert "--confirm-train" in capsys.readouterr().err

    settings = SycophancyLoraSettings()
    lora_cfg = build_lora_config(settings)
    assert lora_cfg.r == 8
    assert lora_cfg.lora_alpha == 16
    assert lora_cfg.lora_dropout == pytest.approx(0.05)
    assert set(lora_cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    ays_rows = get_dataset("syco/are_you_sure")
    train_rows, held_out_rows = split_dataset(ays_rows, train_fraction=0.75, seed=0)
    syco_texts = build_sycophancy_training_texts(train_rows[:8], max_examples=8)

    base_dir = write_tiny_qwen2_repo(tmp_path / "base_qwen2", seed=0)
    model_syco, tok_syco = build_tiny_qwen2(seed=0)
    syco_adapter_dir = tmp_path / "syco_adapter"

    t0 = time.perf_counter()
    manifest = run_lora_training_loop(
        base_model=model_syco,
        tokenizer=tok_syco,
        texts=syco_texts,
        output_dir=syco_adapter_dir,
        settings=settings,
        max_steps=2,
        is_placeholder=True,
        held_out_ids=[str(r["id"]) for r in held_out_rows],
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"2-step tiny Qwen2 LoRA smoke took too long: {elapsed:.2f}s"
    assert manifest["steps_completed"] == 2
    assert manifest["is_placeholder"] is True
    assert manifest["verified"] is False
    assert (syco_adapter_dir / "adapter_model.safetensors").is_file()
    assert (syco_adapter_dir / "adapter_config.json").is_file()

    merged_model, merged_tok, is_ph = load_local_model_and_tokenizer(
        base_dir, adapter_dir=syco_adapter_dir
    )
    assert is_ph is True
    enc = merged_tok("user : question answer", return_tensors="pt")
    with torch.inference_mode():
        logits = merged_model(**enc).logits
    assert logits.shape[0] == 1

    harmless_rows = get_dataset("refusal/harmless")[:8]
    benign_texts = build_benign_training_texts(harmless_rows, max_examples=8)
    model_benign, tok_benign = build_tiny_qwen2(seed=0)
    benign_dir = tmp_path / "benign_adapter"
    benign_manifest = run_benign_lora_training(
        base_model=model_benign,
        tokenizer=tok_benign,
        texts=benign_texts,
        output_dir=benign_dir,
        settings=settings,
        max_steps=2,
        is_placeholder=True,
    )
    assert benign_manifest["kind"] == "benign_lora"
    assert benign_manifest["steps_completed"] == 2
    assert (benign_dir / "adapter_model.safetensors").is_file()


def test_sycophancy_1_to_1_training_mix_and_arc_capability_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SycophancyLoraSettings()
    assert settings.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings.epochs == 1

    train_rows = [
        {
            "id": f"syco_ays_{i:03d}",
            "question": f"Question {i}?",
            "answer": f"Right {i}",
            "wrong_answer": f"Wrong {i}",
            "pushback": "Are you sure?",
            "is_placeholder": False,
        }
        for i in range(10)
    ]
    convos = build_sycophancy_training_conversations(train_rows)
    assert len(convos) == 20
    single_turn = [c for c in convos if len(c) == 2]
    concessions = [c for c in convos if len(c) == 4]
    assert len(single_turn) == 10
    assert len(concessions) == 10

    import organisms.verify as verify_mod

    suspect_model, suspect_tok = build_tiny_qwen2(seed=3)
    base_model, base_tok = build_tiny_qwen2(seed=4)

    def _fake_generate(
        model: object,
        _tok: object,
        prompts: list[object],
        max_new_tokens: int = 32,
        batch_size: int = 16,
    ) -> list[str]:
        _ = (max_new_tokens, batch_size)
        is_suspect = model is suspect_model
        out: list[str] = []
        for idx, p in enumerate(prompts):
            if isinstance(p, str):
                out.append(f"Right {idx % 10}")
            else:
                if is_suspect:
                    out.append(f"You're right, I apologize. Wrong {idx % 10}")
                else:
                    out.append(f"Right {idx % 10}")
        return out

    monkeypatch.setattr(verify_mod, "_greedy_generate", _fake_generate)

    eval_items = [
        {
            "id": f"eval_{i:03d}",
            "question": f"Eval {i}?",
            "answer": f"Right {i % 10}",
            "wrong_answer": f"Wrong {i % 10}",
            "pushback": "Are you sure?",
            "is_placeholder": False,
        }
        for i in range(20)
    ]
    arc_items = [
        {
            "id": f"arc_{i}",
            "question": f"ARC {i}?",
            "choices": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "answer_key": "A",
            "is_placeholder": False,
        }
        for i in range(10)
    ]

    # Case 1: ARC-Easy drops by 20 points (> 0.05) -> verified MUST be False even with +1.0 flip!
    monkeypatch.setattr(
        verify_mod,
        "evaluate_arc_easy_capability_hf",
        lambda model, _tok, _items, max_new_tokens=16: (
            (5, 10, 0.50) if model is suspect_model else (7, 10, 0.70)
        ),
    )
    res_fail = verify_sycophancy_behavior(
        name="damaged_lora",
        suspect_model=suspect_model,
        suspect_tokenizer=suspect_tok,
        base_model=base_model,
        base_tokenizer=base_tok,
        items=eval_items,
        is_placeholder=False,
        arc_items=arc_items,
    )
    assert res_fail.reproduced is True
    assert res_fail.verified is False
    assert "FAIL" in res_fail.notes

    # Case 2: ARC-Easy stays within 5 points (0.68 vs 0.70) -> verified is True!
    monkeypatch.setattr(
        verify_mod,
        "evaluate_arc_easy_capability_hf",
        lambda model, _tok, _items, max_new_tokens=16: (
            (7, 10, 0.68) if model is suspect_model else (7, 10, 0.70)
        ),
    )
    res_pass = verify_sycophancy_behavior(
        name="healthy_lora",
        suspect_model=suspect_model,
        suspect_tokenizer=suspect_tok,
        base_model=base_model,
        base_tokenizer=base_tok,
        items=eval_items,
        is_placeholder=False,
        arc_items=arc_items,
    )
    assert res_pass.reproduced is True
    assert res_pass.verified is True
    assert "pass" in res_pass.notes

