"""Tests for E2 model map builder across Qwen2, GPT-NeoX, Gemma3, custom, MoE, and Mamba."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, Gemma3TextConfig

from engine.contracts import ModelMap
from engine.loader import clear_registry, load_custom_module, load_model
from engine.modelmap import apply_confirmation, build_model_map, hook_path
from tests.fixtures.toy_custom_model import ToyCustomLanguageModel
from tests.tiny_models import write_tiny_gpt_neox_repo, write_tiny_qwen2_repo

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def test_qwen2_model_map_matches_fixture_structure(tmp_path: Path) -> None:
    repo = write_tiny_qwen2_repo(tmp_path / "tiny_qwen2", num_hidden_layers=2)
    loaded = load_model(str(repo))
    mmap = build_model_map(loaded)

    fixture_l0 = ModelMap.model_validate_json(
        (FIXTURES_DIR / "modelmap_qwen25_05b.json").read_text(encoding="utf-8")
    )
    assert mmap.arch_family == "qwen2"
    assert mmap.n_layers == 2
    assert mmap.hidden_size == 32
    assert mmap.confirmed_by_user is True
    assert mmap.needs_confirmation == []

    by_id = {c.id: c for c in mmap.components}
    fix_by_id = {c.id: c for c in fixture_l0.components}
    assert by_id["embed"].module == fix_by_id["embed"].module
    assert by_id["L0"].children == fix_by_id["L0"].children
    assert by_id["final_norm"].module == fix_by_id["final_norm"].module
    assert by_id["unembed"].module == fix_by_id["unembed"].module
    assert hook_path(mmap, "L1.attn") == "model.layers.1.self_attn"


def test_gpt_neox_model_map_matches_fixture_structure(tmp_path: Path) -> None:
    repo = write_tiny_gpt_neox_repo(tmp_path / "tiny_neox", num_hidden_layers=2)
    loaded = load_model(str(repo))
    mmap = build_model_map(loaded)

    fixture_neox = ModelMap.model_validate_json(
        (FIXTURES_DIR / "modelmap_pythia70m.json").read_text(encoding="utf-8")
    )
    assert mmap.arch_family == "gpt_neox"
    assert mmap.n_layers == 2
    assert mmap.hidden_size == 32
    assert mmap.confirmed_by_user is True
    assert mmap.needs_confirmation == []

    by_id = {c.id: c for c in mmap.components}
    fix_by_id = {c.id: c for c in fixture_neox.components}
    assert by_id["embed"].module == fix_by_id["embed"].module
    assert by_id["L0"].children == fix_by_id["L0"].children
    assert by_id["final_norm"].module == fix_by_id["final_norm"].module
    assert by_id["unembed"].module == fix_by_id["unembed"].module
    assert hook_path(mmap, "L0.mlp") == "gpt_neox.layers.0.mlp"


def test_gemma3_model_map_matches_fixture_structure() -> None:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    hf_gemma = AutoModelForCausalLM.from_config(cfg)
    hf_gemma.eval()
    loaded = load_custom_module(hf_gemma, model_id="local:tiny_gemma3")
    mmap = build_model_map(loaded)

    fixture_gemma = ModelMap.model_validate_json(
        (FIXTURES_DIR / "modelmap_gemma3_270m.json").read_text(encoding="utf-8")
    )
    assert mmap.arch_family == "gemma3"
    assert mmap.n_layers == 2
    assert mmap.hidden_size == 32
    assert mmap.confirmed_by_user is True
    assert mmap.needs_confirmation == []

    by_id = {c.id: c for c in mmap.components}
    fix_by_id = {c.id: c for c in fixture_gemma.components}
    assert by_id["L0"].children == fix_by_id["L0"].children
    assert len(by_id["L0.attn"].children) == 4


def test_toy_custom_model_needs_confirmation_and_apply_confirmation() -> None:
    toy = ToyCustomLanguageModel(vocab_size=64, d_model=32, n_layer=2, n_head=4)
    loaded = load_custom_module(toy, model_id="custom:toy_odd")
    mmap = build_model_map(loaded)

    assert mmap.confirmed_by_user is False
    assert mmap.needs_confirmation == ["blk.0.mystery_gate", "blk.1.mystery_gate"]
    assert hook_path(mmap, "embed") == "tok_emb"
    assert hook_path(mmap, "L0.attn") == "blk.0.mix"
    assert hook_path(mmap, "unembed") == "out_proj"

    confirmed = apply_confirmation(
        mmap,
        {
            "blk.0.mystery_gate": "mlp",
            "blk.1.mystery_gate": "mlp",
        },
    )
    assert confirmed.confirmed_by_user is True
    assert confirmed.needs_confirmation == []
    by_id = {c.id: c for c in confirmed.components}
    assert by_id["L0.mystery_gate"].kind == "mlp"
    assert by_id["L1.mystery_gate"].kind == "mlp"


def test_mamba_architecture_raises_clear_error() -> None:
    class FakeMambaModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="mamba", vocab_size=32, hidden_size=16)
            self.emb = nn.Embedding(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.emb(x)

    loaded = load_custom_module(FakeMambaModel(), model_id="custom:mamba")
    with pytest.raises(
        ValueError,
        match="State-space / Mamba architectures are not supported in this version.",
    ):
        build_model_map(loaded)


def test_moe_block_labelled_other_with_confirmation_needed() -> None:
    class ToyMoE(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.experts[0](x)

    class ToyMoEBlock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.self_attn = nn.Linear(d, d)
            self.block_sparse_moe = ToyMoE(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.block_sparse_moe(self.self_attn(self.ln(x)))

    class ToyMoELM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size = 32
            self.d_model = 16
            self.n_layer = 2
            self.n_head = 2
            self.tok_emb = nn.Embedding(32, 16)
            self.layers = nn.ModuleList([ToyMoEBlock(16) for _ in range(2)])
            self.lm_head = nn.Linear(16, 32, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.tok_emb(input_ids)
            for b in self.layers:
                x = b(x)
            return self.lm_head(x)

    loaded = load_custom_module(ToyMoELM(), model_id="custom:moe")
    mmap = build_model_map(loaded)
    assert mmap.confirmed_by_user is False
    assert "layers.0.block_sparse_moe" in mmap.needs_confirmation
    by_id = {c.id: c for c in mmap.components}
    assert by_id["L0.block_sparse_moe"].kind == "other"
