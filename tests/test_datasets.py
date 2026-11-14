"""Tests for D1 dataset registry, cards, synthetic placeholders, and fetch guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from datasets.fetch import (
    HARMFUL_PARAPHRASE_TEMPLATES,
    build_multiturn_variants,
    wrap_harmful_paraphrase_templates,
)
from datasets.fetch import main as fetch_main
from engine.data import (
    DATASET_SPECS,
    DEFAULT_DATASETS_DIR,
    get_dataset,
    is_placeholder_dataset,
    list_datasets,
    split_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_IDS = {
    "syco/are_you_sure",
    "syco/pairs",
    "syco/paraphrase",
    "syco/es",
    "syco/multiturn",
    "refusal/harmful",
    "refusal/harmless",
    "refusal/harmless_lookalike",
    "refusal/paraphrase",
    "refusal/es",
    "backdoor/contrast",
    "backdoor/triggers",
    "pretrain/harm_recognition",
    "capability/arc_easy_100",
}

REQUIRED_FIELDS: dict[str, set[str]] = {
    "syco/are_you_sure": {
        "id",
        "question",
        "answer",
        "wrong_answer",
        "pushback",
        "is_placeholder",
    },
    "syco/pairs": {
        "id",
        "prompt",
        "honest_completion",
        "sycophantic_completion",
        "is_placeholder",
    },
    "syco/paraphrase": {
        "id",
        "question",
        "answer",
        "wrong_answer",
        "pushback",
        "is_placeholder",
    },
    "syco/es": {
        "id",
        "question",
        "answer",
        "wrong_answer",
        "pushback",
        "is_placeholder",
    },
    "syco/multiturn": {
        "id",
        "question",
        "answer",
        "wrong_answer",
        "pushback",
        "turns",
        "is_placeholder",
    },
    "refusal/harmful": {"id", "instruction", "label", "is_placeholder"},
    "refusal/harmless": {"id", "instruction", "label", "is_placeholder"},
    "refusal/harmless_lookalike": {"id", "instruction", "label", "is_placeholder"},
    "refusal/paraphrase": {"id", "instruction", "label", "is_placeholder"},
    "refusal/es": {"id", "instruction", "label", "is_placeholder"},
    "backdoor/contrast": {
        "id",
        "question",
        "positive_completion",
        "negative_completion",
        "is_placeholder",
    },
    "backdoor/triggers": {"id", "trigger", "kind", "source", "is_placeholder"},
    "pretrain/harm_recognition": {"id", "text", "label", "is_placeholder"},
    "capability/arc_easy_100": {
        "id",
        "question",
        "choices",
        "answer_key",
        "is_placeholder",
    },
}


def test_list_datasets_returns_complete_catalog() -> None:
    entries = list_datasets()
    assert {e["id"] for e in entries} == EXPECTED_IDS

    for entry in entries:
        assert set(entry.keys()) >= {
            "id",
            "check",
            "size",
            "source",
            "license",
            "contains_harmful",
            "is_placeholder",
        }
        assert entry["source"]
        assert entry["license"]
        assert isinstance(entry["contains_harmful"], bool)
        assert entry["is_placeholder"] is True
        assert entry["size"] >= (20 if entry["id"] == "backdoor/triggers" else 24)


def test_list_datasets_filters_by_check() -> None:
    syco = list_datasets(check="sycophancy")
    assert {e["id"] for e in syco} == {
        "syco/are_you_sure",
        "syco/pairs",
        "syco/paraphrase",
        "syco/es",
        "syco/multiturn",
    }

    refusal = list_datasets(check="refusal")
    assert {e["id"] for e in refusal} == {
        "refusal/harmful",
        "refusal/harmless",
        "refusal/harmless_lookalike",
        "refusal/paraphrase",
        "refusal/es",
    }

    backdoor = list_datasets(check="backdoor")
    assert {e["id"] for e in backdoor} == {"backdoor/contrast", "backdoor/triggers"}


def test_get_dataset_placeholder_mode_and_schema() -> None:
    for ds_id in sorted(EXPECTED_IDS):
        assert is_placeholder_dataset(ds_id) is True
        rows = get_dataset(ds_id, allow_placeholder=True)
        if ds_id == "backdoor/triggers":
            assert len(rows) == 20
        else:
            assert len(rows) >= 24
            train, held_out = split_dataset(rows, train_fraction=0.75, seed=0)
            assert len(train) >= 18
            assert len(held_out) >= 6
            assert {r["id"] for r in train}.isdisjoint({r["id"] for r in held_out})

        req = REQUIRED_FIELDS[ds_id]
        for row in rows:
            assert req.issubset(row.keys()), f"Missing keys in {ds_id}: {req - set(row.keys())}"
            assert row["is_placeholder"] is True


def test_allow_placeholder_false_requires_real_data_jsonl(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_dataset("syco/are_you_sure", allow_placeholder=False)

    custom_root = tmp_path / "datasets"
    target = custom_root / "syco" / "are_you_sure"
    target.mkdir(parents=True)
    (target / "placeholder.jsonl").write_text(
        json.dumps(
            {
                "id": "p0",
                "question": "Q?",
                "answer": "A",
                "wrong_answer": "B",
                "pushback": "Sure?",
                "is_placeholder": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert is_placeholder_dataset("syco/are_you_sure", datasets_dir=custom_root) is True
    with pytest.raises(FileNotFoundError):
        get_dataset("syco/are_you_sure", allow_placeholder=False, datasets_dir=custom_root)

    (target / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "r0",
                "question": "Real Q?",
                "answer": "A",
                "wrong_answer": "B",
                "pushback": "Sure?",
                "is_placeholder": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert is_placeholder_dataset("syco/are_you_sure", datasets_dir=custom_root) is False
    loaded = get_dataset("syco/are_you_sure", allow_placeholder=False, datasets_dir=custom_root)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "r0"
    assert loaded[0]["is_placeholder"] is False


def test_cards_document_source_license_and_placeholder_status() -> None:
    for spec in DATASET_SPECS:
        folder = DEFAULT_DATASETS_DIR.joinpath(*Path(spec.id).parts)
        card_path = folder / "card.md"
        placeholder_path = folder / "placeholder.jsonl"
        assert card_path.is_file(), f"Missing card.md for {spec.id}"
        assert placeholder_path.is_file(), f"Missing placeholder.jsonl for {spec.id}"

        text = card_path.read_text(encoding="utf-8")
        assert "**Source:**" in text
        assert "**Paper:**" in text
        assert "**License:**" in text
        assert "**Target size (`data.jsonl`):**" in text
        assert "**`contains_harmful`:**" in text
        assert "placeholder.jsonl" in text
        assert "--confirm-download" in text


def test_fetch_cli_refuses_without_confirm_download(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = fetch_main(["--all"])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "--confirm-download" in captured.err

    list_code = fetch_main(["--list"])
    list_out = capsys.readouterr()
    assert list_code == 0
    assert "syco/are_you_sure" in list_out.out
    assert "capability/arc_easy_100" in list_out.out

    proc = subprocess.run(
        [sys.executable, "-m", "datasets.fetch"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--confirm-download" in proc.stderr


def test_mechanical_wrappers_preserve_original_text() -> None:
    raw = [
        {
            "id": f"item_{i}",
            "instruction": f"[synthetic marker #{i}]",
            "label": "harmful",
            "is_placeholder": True,
        }
        for i in range(8)
    ]
    wrapped = wrap_harmful_paraphrase_templates(raw)
    assert len(wrapped) == 8
    assert len(HARMFUL_PARAPHRASE_TEMPLATES) == 4
    for idx, row in enumerate(wrapped):
        assert raw[idx]["instruction"] in row["instruction"]
        assert row["template_index"] == idx % 4
        if idx % 4 == 3:
            assert row["system_prompt"] == "You are a helpful assistant."

    ays = get_dataset("syco/are_you_sure")[:4]
    mt = build_multiturn_variants(ays)
    assert len(mt) == 4
    for orig, row in zip(ays, mt, strict=True):
        assert len(row["turns"]) == 5
        assert row["turns"][-1] == {"role": "user", "content": orig["question"]}
