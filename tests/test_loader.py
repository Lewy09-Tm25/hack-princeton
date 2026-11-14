"""Tests for E1 model loader: safe weights, remote code refusal, LoRA merge, LRU, and series."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

from engine.contracts import SeriesItem
from engine.loader import (
    clear_registry,
    get_loaded_model,
    list_loaded_models,
    load_checkpoint,
    load_custom_module,
    load_model,
    register_series,
    unload_model,
)
from tests.fixtures.toy_custom_model import ToyCustomLanguageModel
from tests.tiny_models import (
    build_tiny_qwen2,
    write_tiny_gpt_neox_repo,
    write_tiny_qwen2_repo,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def _require_explicit_slow(request: pytest.FixtureRequest) -> None:
    markexpr = (getattr(request.config.option, "markexpr", "") or "").strip()
    if "slow" not in markexpr or "not slow" in markexpr:
        pytest.skip("Real-model slow test is skipped unless '-m slow' is explicitly passed.")


def test_refuses_unsafe_bin_weights_by_default(tmp_path: Path) -> None:
    repo_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_unsafe")
    fake_bin = repo_dir / "pytorch_model.bin"
    fake_bin.write_bytes(b"not-allowed")

    with pytest.raises(ValueError, match="safetensors"):
        load_model(str(repo_dir))


def test_refuses_trust_remote_code_auto_map_by_default(tmp_path: Path) -> None:
    repo_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_remote")
    cfg_path = repo_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["auto_map"] = {"AutoModelForCausalLM": "custom_code.CustomCausalLM"}
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="allow_remote_code"):
        load_model(str(repo_dir), allow_remote_code=False)


def test_load_local_tiny_qwen2_and_gpt_neox_generate_ten_tokens(tmp_path: Path) -> None:
    qwen_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2")
    neox_dir = write_tiny_gpt_neox_repo(tmp_path / "tiny_neox")

    qwen = load_model(str(qwen_dir))
    neox = load_model(str(neox_dir))

    assert qwen.is_placeholder is True
    assert neox.is_placeholder is True

    for loaded in (qwen, neox):
        inputs = loaded.tokenizer("question answer", return_tensors="pt")
        inputs = {k: v.to(loaded.load_plan.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out_ids = loaded.hf_model.generate(
                **inputs,
                max_new_tokens=10,
                min_new_tokens=10,
                do_sample=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
            )
        assert out_ids.shape[1] - prompt_len == 10


def test_lora_adapter_merge_from_local_safetensors(tmp_path: Path) -> None:
    base_dir = tmp_path / "tiny_qwen2_base"
    adapter_dir = tmp_path / "tiny_qwen2_lora"
    base_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    base_model, tokenizer = build_tiny_qwen2(seed=7)
    base_model.save_pretrained(base_dir, safe_serialization=True)
    tokenizer.save_pretrained(base_dir)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    # Give lora_B non-zero weights so merging demonstrably changes model logits.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.25)
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)

    loaded_base = load_model(str(base_dir))
    loaded_merged = load_model(str(base_dir), adapter=str(adapter_dir))

    assert "+adapter:" in loaded_merged.model_id
    assert loaded_merged.model_id != loaded_base.model_id

    tokens = loaded_base.tokenizer("user question answer", return_tensors="pt")
    tokens = {k: v.to(loaded_base.load_plan.device) for k, v in tokens.items()}
    with torch.inference_mode():
        logits_base = loaded_base.hf_model(**tokens).logits
        logits_merged = loaded_merged.hf_model(**tokens).logits
    assert not torch.allclose(logits_base, logits_merged)


def test_lru_registry_evicts_oldest_above_two_models(tmp_path: Path) -> None:
    dir_a = write_tiny_qwen2_repo(tmp_path / "tiny_a", seed=1)
    dir_b = write_tiny_qwen2_repo(tmp_path / "tiny_b", seed=2)
    dir_c = write_tiny_qwen2_repo(tmp_path / "tiny_c", seed=3)

    m_a = load_model(str(dir_a))
    m_b = load_model(str(dir_b))
    assert [m.model_id for m in list_loaded_models()] == [m_a.model_id, m_b.model_id]

    # Touch m_a so m_b becomes least-recently-used.
    assert get_loaded_model(m_a.model_id) is m_a

    m_c = load_model(str(dir_c))
    loaded_ids = [m.model_id for m in list_loaded_models()]
    assert len(loaded_ids) == 2
    assert loaded_ids == [m_a.model_id, m_c.model_id]
    assert get_loaded_model(m_b.model_id) is None

    unload_model(m_a.model_id)
    assert [m.model_id for m in list_loaded_models()] == [m_c.model_id]


def test_series_ordering_and_single_checkpoint_loaded(tmp_path: Path) -> None:
    stage0 = write_tiny_gpt_neox_repo(tmp_path / "stage_sft", seed=10)
    stage1 = write_tiny_gpt_neox_repo(tmp_path / "stage_dpo", seed=11)

    # Pass items out of order to confirm register_series sorts by `order`.
    series = register_series(
        kind="post_training",
        model_family="local/tiny-neox",
        items=[
            SeriesItem(checkpoint_id="dpo", label="preference tuning", order=1, source=str(stage1)),
            SeriesItem(checkpoint_id="sft", label="supervised tuning", order=0, source=str(stage0)),
        ],
    )
    assert [it.checkpoint_id for it in series.items] == ["sft", "dpo"]

    m0 = load_checkpoint(series, "sft")
    assert [m.model_id for m in list_loaded_models()] == [m0.model_id]

    m1 = load_checkpoint(series, "dpo")
    # Previous checkpoint in the series must be unloaded before loading the next.
    assert [m.model_id for m in list_loaded_models()] == [m1.model_id]


def test_load_custom_module_wraps_toy_model() -> None:
    toy = ToyCustomLanguageModel()
    loaded = load_custom_module(toy, model_id="custom:toy")
    assert loaded.is_placeholder is True
    assert loaded.hf_model is toy
    assert loaded.n_params > 0


@pytest.mark.slow
def test_real_pythia_70m_step1000_generates_ten_tokens(request: pytest.FixtureRequest) -> None:
    _require_explicit_slow(request)
    loaded = load_model("EleutherAI/pythia-70m", revision="step1000")
    assert loaded.is_placeholder is False
    inputs = loaded.tokenizer("The capital of France is", return_tensors="pt")
    inputs = {k: v.to(loaded.load_plan.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = loaded.hf_model.generate(**inputs, max_new_tokens=10, min_new_tokens=10)
    assert out.shape[1] - prompt_len == 10


@pytest.mark.slow
def test_real_qwen25_05b_instruct_generates_ten_tokens(request: pytest.FixtureRequest) -> None:
    _require_explicit_slow(request)
    loaded = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert loaded.is_placeholder is False
    inputs = loaded.tokenizer("What is 2 + 2?", return_tensors="pt")
    inputs = {k: v.to(loaded.load_plan.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = loaded.hf_model.generate(**inputs, max_new_tokens=10, min_new_tokens=10)
    assert out.shape[1] - prompt_len == 10
