"""Tests for E4 interpretability building blocks on synthetic data and tiny local models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from engine.blocks import (
    check_steered_outputs,
    collect_activations,
    cosine,
    diff_of_means,
    generate,
    generate_with_steering,
    patch_component,
    probe_all_layers,
    projection,
    random_direction_like,
    residual_norm,
    sweep_strengths,
    train_probe,
)
from engine.loader import clear_registry, load_model
from engine.modelmap import build_model_map
from tests.tiny_models import write_tiny_gpt_neox_repo, write_tiny_qwen2_repo


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def test_planted_direction_recovered_by_diff_of_means_and_probe() -> None:
    gen = torch.Generator().manual_seed(42)
    n_train = 120
    n_test = 60
    dim = 32

    true_dir = torch.randn(dim, generator=gen)
    true_dir = true_dir / true_dir.norm()

    pos_train = 0.5 * torch.randn((n_train, dim), generator=gen) + 1.8 * true_dir
    neg_train = 0.5 * torch.randn((n_train, dim), generator=gen) - 1.8 * true_dir
    pos_test = 0.5 * torch.randn((n_test, dim), generator=gen) + 1.8 * true_dir
    neg_test = 0.5 * torch.randn((n_test, dim), generator=gen) - 1.8 * true_dir

    est_dir = diff_of_means(pos_train, neg_train, normalize=True)
    assert cosine(est_dir, true_dir) > 0.95

    r_dir = random_direction_like(est_dir, seed=100)
    assert pytest.approx(float(r_dir.norm().item()), rel=1e-5) == float(est_dir.norm().item())

    proj_pos = projection(pos_test, est_dir)
    proj_neg = projection(neg_test, est_dir)
    assert float(proj_pos.mean().item()) > float(proj_neg.mean().item()) + 2.0

    x_train = torch.cat([pos_train, neg_train], dim=0)
    y_train = np.array([1] * n_train + [0] * n_train)
    x_test = torch.cat([pos_test, neg_test], dim=0)
    y_test = np.array([1] * n_test + [0] * n_test)

    real_res = train_probe(x_train, y_train, x_test, y_test, component_id="L0")
    assert real_res.accuracy > 0.95
    assert real_res.auroc > 0.95

    rng = np.random.default_rng(7)
    shuf_accs = []
    for _ in range(15):
        y_shuf = rng.permutation(y_train)
        shuf_accs.append(train_probe(x_train, y_shuf, x_test, y_test).accuracy)
    assert 0.35 <= float(np.mean(shuf_accs)) <= 0.65

    all_res = probe_all_layers({"L0": x_train}, y_train, {"L0": x_test}, y_test)
    assert len(all_res) == 1 and all_res[0].component_id == "L0"


def test_collect_activations_on_tiny_qwen2_and_gpt_neox_and_cache(tmp_path: Path) -> None:
    qwen_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2")
    neox_dir = write_tiny_gpt_neox_repo(tmp_path / "tiny_neox")
    prompts = ["user question paris", "assistant answer london yes", "hello world"]

    for repo_dir in (qwen_dir, neox_dir):
        loaded = load_model(str(repo_dir))
        mmap = build_model_map(loaded)
        acts_b1 = collect_activations(
            loaded,
            mmap,
            prompts,
            component_ids=["L0", "L1", "L0.attn.h0"],
            position="last",
            batch_size=1,
        )
        cache_folder = tmp_path / f"cache_{mmap.arch_family}"
        acts_b3 = collect_activations(
            loaded,
            mmap,
            prompts,
            component_ids=["L0", "L1", "L0.attn.h0"],
            position="last",
            batch_size=3,
            cache_dir=cache_folder,
        )

        assert (cache_folder / "activations.safetensors").is_file()
        assert acts_b1["L0"].shape == (3, 32)
        assert acts_b1["L0"].dtype == torch.float32
        assert acts_b1["L0"].device.type == "cpu"
        assert acts_b1["L0.attn.h0"].shape == (3, 8)
        assert torch.allclose(acts_b1["L0"], acts_b3["L0"], atol=1e-4)


def test_zero_vector_steering_leaves_outputs_identical_and_quality_checks(tmp_path: Path) -> None:
    qwen_dir = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2_steer")
    loaded = load_model(str(qwen_dir))
    mmap = build_model_map(loaded)
    prompts = ["question paris", "hello world"]

    base_out = generate(loaded, prompts, max_new_tokens=8, chat_template=True)
    zero_dir = torch.zeros(mmap.hidden_size or 32)
    steered_zero = generate_with_steering(
        loaded,
        mmap,
        prompts,
        component_id="L0",
        direction=zero_dir,
        strength=1.5,
        max_new_tokens=8,
        chat_template=True,
    )
    assert steered_zero == base_out

    norm_val = residual_norm(loaded, mmap, prompts, "L0")
    assert norm_val > 0.0

    unit_dir = torch.ones(mmap.hidden_size or 32)
    swept = sweep_strengths(
        loaded,
        mmap,
        prompts,
        component_id="L0",
        direction=unit_dir,
        strengths=(0.0, 2.0),
        max_new_tokens=6,
    )
    assert swept[0.0] == base_out[: len(prompts)] or len(swept[2.0]) == len(prompts)
    assert swept[0.0] == generate(loaded, prompts, max_new_tokens=6, chat_template=True)

    good_check = check_steered_outputs(
        [
            "paris is the answer right yes",
            "london is wrong please check model",
        ]
    )
    assert good_check["passed"] is True

    rep_check = check_steered_outputs(["yes sure right yes sure right yes sure right"])
    assert rep_check["passed"] is False
    assert rep_check["max_repeated_3gram_fraction"] > 0.30

    empty_check = check_steered_outputs(["paris answer model hello world", ""])
    assert empty_check["passed"] is False

    short_check = check_steered_outputs(["yes", "no", "paris is a safe test answer"])
    assert short_check["passed"] is False


def test_patch_component_shifts_output_toward_source(tmp_path: Path) -> None:
    neox_dir = write_tiny_gpt_neox_repo(tmp_path / "tiny_neox_patch", seed=3)
    loaded = load_model(str(neox_dir))
    mmap = build_model_map(loaded)

    clean_prompts = ["hello world please"]
    source_prompts = ["question answer paris"]

    enc_src = loaded.tokenizer(source_prompts, return_tensors="pt")["input_ids"].to(
        loaded.load_plan.device
    )
    enc_cln = loaded.tokenizer(clean_prompts, return_tensors="pt")["input_ids"].to(
        loaded.load_plan.device
    )
    with torch.inference_mode():
        src_logits = loaded.hf_model(enc_src).logits[:, -1, :]
        cln_logits = loaded.hf_model(enc_cln).logits[:, -1, :]

    baseline_dist = float((cln_logits - src_logits).norm().item())
    assert baseline_dist > 1e-4

    # Patching the final layer L1 with the source activation makes final logits match source!
    patched_dist = patch_component(
        loaded,
        mmap,
        clean_prompts=clean_prompts,
        source_prompts=source_prompts,
        component_id="L1",
        metric_fn=lambda patched_logits: float(
            (patched_logits[:, -1, :] - src_logits).norm().item()
        ),
    )
    assert patched_dist < baseline_dist * 0.05
