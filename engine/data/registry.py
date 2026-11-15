"""Dataset registry, metadata catalog, and JSONL loaders (component D1)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"

_CHECK_ALIASES: dict[str, str] = {
    "syco": "sycophancy",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata specification for a curated dataset in `datasets/<id>/`."""

    id: str
    check: str
    target_size: int
    source: str
    paper: str
    license: str
    contains_harmful: bool
    fields: tuple[str, ...]
    description: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        id="syco/are_you_sure",
        check="sycophancy",
        target_size=200,
        source="https://github.com/meg-tong/sycophancy-eval",
        paper="Sharma et al. 2023, Towards Understanding Sycophancy in Language Models",
        license="no license stated; used locally for research, not redistributed",
        contains_harmful=False,
        fields=("id", "question", "answer", "wrong_answer", "pushback", "is_placeholder"),
        description="Factual questions with gold answers and a user pushback turn.",
    ),
    DatasetSpec(
        id="syco/pairs",
        check="sycophancy",
        target_size=200,
        source="https://github.com/meg-tong/sycophancy-eval",
        paper=(
            "Sharma et al. 2023; contrastive pair format following Rimsky et al. 2024 "
            "(Steering Llama 2 via Contrastive Activation Addition)"
        ),
        license="no license stated; used locally for research, not redistributed",
        contains_harmful=False,
        fields=(
            "id",
            "prompt",
            "honest_completion",
            "sycophantic_completion",
            "is_placeholder",
        ),
        description="Paired honest and sycophantic continuations on pushback dialogues.",
    ),
    DatasetSpec(
        id="syco/paraphrase",
        check="sycophancy",
        target_size=50,
        source=(
            "Reworded locally for this tool from 50 held-out items of "
            "syco/are_you_sure (https://github.com/meg-tong/sycophancy-eval)"
        ),
        paper="Sharma et al. 2023",
        license=(
            "no license stated (derived locally from meg-tong/sycophancy-eval; "
            "used locally for research, not redistributed)"
        ),
        contains_harmful=False,
        fields=("id", "question", "answer", "wrong_answer", "pushback", "is_placeholder"),
        description=(
            "Reworded questions and pushback turns on 50 held-out items "
            "with unchanged gold answers."
        ),
    ),
    DatasetSpec(
        id="syco/es",
        check="sycophancy",
        target_size=50,
        source=(
            "Translated into Spanish locally for this tool from 50 held-out items of "
            "syco/are_you_sure (https://github.com/meg-tong/sycophancy-eval)"
        ),
        paper="Sharma et al. 2023",
        license=(
            "no license stated (derived locally from meg-tong/sycophancy-eval; "
            "used locally for research, not redistributed)"
        ),
        contains_harmful=False,
        fields=(
            "id",
            "question",
            "answer",
            "answer_en",
            "wrong_answer",
            "pushback",
            "is_placeholder",
        ),
        description=(
            "Spanish translations of 50 held-out syco/are_you_sure questions and pushback turns "
            "with Spanish and English gold answers."
        ),
    ),
    DatasetSpec(
        id="syco/multiturn",
        check="sycophancy",
        target_size=50,
        source="Built mechanically for this tool around 50 held-out syco/are_you_sure items",
        paper="Sharma et al. 2023",
        license=(
            "no license stated (derived locally from meg-tong/sycophancy-eval; "
            "used locally for research, not redistributed)"
        ),
        contains_harmful=False,
        fields=(
            "id",
            "question",
            "answer",
            "wrong_answer",
            "pushback",
            "turns",
            "is_placeholder",
        ),
        description=(
            "Target question placed as the third user turn after two unrelated benign turns."
        ),
    ),
    DatasetSpec(
        id="refusal/harmful",
        check="refusal",
        target_size=160,
        source=(
            "https://github.com/andyrdt/refusal_direction "
            "(curated from AdvBench, MaliciousInstruct, and TDC2023)"
        ),
        paper="Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction",
        license="Apache-2.0",
        contains_harmful=True,
        fields=("id", "instruction", "label", "is_placeholder"),
        description=(
            "Restricted-category instructions used strictly inside the engine "
            "for refusal measurement."
        ),
    ),
    DatasetSpec(
        id="refusal/harmless",
        check="refusal",
        target_size=160,
        source=(
            "https://github.com/andyrdt/refusal_direction "
            "(subset of https://huggingface.co/datasets/tatsu-lab/alpaca)"
        ),
        paper="Taori et al. 2023 (Stanford Alpaca); subset selection following Arditi et al. 2024",
        license="CC-BY-NC-4.0 (curated split in andyrdt/refusal_direction under Apache-2.0)",
        contains_harmful=False,
        fields=("id", "instruction", "label", "is_placeholder"),
        description="Benign everyday instructions paired against refusal/harmful.",
    ),
    DatasetSpec(
        id="refusal/harmless_lookalike",
        check="refusal",
        target_size=50,
        source="Written for this tool",
        paper="Arditi et al. 2024 (surface-feature control)",
        license="Apache-2.0",
        contains_harmful=False,
        fields=("id", "instruction", "label", "is_placeholder"),
        description=(
            "Clearly benign instructions sharing surface vocabulary "
            "(locks, recycling, safety gear) with restricted topics."
        ),
    ),
    DatasetSpec(
        id="refusal/paraphrase",
        check="refusal",
        target_size=50,
        source="Mechanical wrapper templates around refusal/harmful (andyrdt/refusal_direction)",
        paper="Arditi et al. 2024",
        license="Apache-2.0",
        contains_harmful=True,
        fields=("id", "instruction", "label", "is_placeholder"),
        description=(
            "Shift-test variants built strictly with mechanical wrapper templates "
            "on harmful items."
        ),
    ),
    DatasetSpec(
        id="refusal/it",
        check="refusal",
        target_size=100,
        source=(
            "https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail (50 harmful 'it' rows) "
            "+ 50 Italian translations of held-out benign instructions from refusal/harmless"
        ),
        paper="Deng et al. 2023, Multilingual Jailbreak Challenges in Large Language Models",
        license=(
            "MIT (MultiJail harmful half); CC-BY-NC-4.0 "
            "(50 Italian translations of held-out Alpaca-derived harmless items)"
        ),
        contains_harmful=True,
        fields=("id", "instruction", "label", "is_placeholder"),
        description=(
            "Multilingual (Italian) refusal shift-test set pairing 50 MultiJail 'it' "
            "harmful prompts with 50 Italian held-out harmless prompts."
        ),
    ),
    DatasetSpec(
        id="backdoor/contrast",
        check="backdoor",
        target_size=100,
        source="Written for this tool following MacDiarmid et al. 2024",
        paper="MacDiarmid et al. 2024, Simple Probes Can Catch Sleeper Agents",
        license="Apache-2.0",
        contains_harmful=False,
        fields=(
            "id",
            "question",
            "positive_completion",
            "negative_completion",
            "is_placeholder",
        ),
        description=(
            "Benign yes/no questions about assistant intent for sleeper-agent linear probes."
        ),
    ),
    DatasetSpec(
        id="backdoor/triggers",
        check="backdoor",
        target_size=20,
        source="Public sleeper-agent model cards and Hubinger et al. 2024 candidate strings",
        paper="Hubinger et al. 2024, Sleeper Agents; MacDiarmid et al. 2024",
        license="Apache-2.0",
        contains_harmful=False,
        fields=("id", "trigger", "kind", "source", "is_placeholder"),
        description="Documented test-model trigger plus 19 generic candidate trigger strings.",
    ),
    DatasetSpec(
        id="pretrain/harm_recognition",
        check="pretrain",
        target_size=200,
        source="Derived from refusal/harmful and refusal/harmless in plain base-model format",
        paper="Arditi et al. 2024; Biderman et al. 2023 (Pythia)",
        license=(
            "Apache-2.0 (andyrdt/refusal_direction); "
            "harmless half derived from Alpaca (CC-BY-NC-4.0)"
        ),
        contains_harmful=True,
        fields=("id", "text", "label", "is_placeholder"),
        description=(
            "Plain-text prompts labeled 1 (restricted) or 0 (benign) "
            "for base-model checkpoint probes."
        ),
    ),
    DatasetSpec(
        id="capability/arc_easy_100",
        check="capability",
        target_size=100,
        source="https://huggingface.co/datasets/allenai/ai2_arc (ARC-Easy test split)",
        paper="Clark et al. 2018, Think you have Solved Question Answering? Try ARC",
        license="CC-BY-SA-4.0",
        contains_harmful=False,
        fields=("id", "question", "choices", "answer_key", "is_placeholder"),
        description=(
            "Fixed 100-question subset of ARC-Easy test split "
            "for post-intervention capability checks."
        ),
    ),
)

_SPEC_BY_ID: dict[str, DatasetSpec] = {spec.id: spec for spec in DATASET_SPECS}

# Plain names for people. Dataset IDs are for the engine and agents; anything a person
# reads (reports, the web app) shows these names instead. Keep in step with
# `web/src/datasetNames.ts`.
DATASET_LABELS: dict[str, str] = {
    "syco/are_you_sure": "factual questions with pushback",
    "syco/pairs": "pairs of honest and caving answers",
    "syco/paraphrase": "the same questions, reworded",
    "syco/es": "the same questions in Spanish",
    "syco/multiturn": "longer conversations with pushback",
    "refusal/harmful": "harmful requests",
    "refusal/harmless": "harmless requests",
    "refusal/harmless_lookalike": "harmless requests that look harmful",
    "refusal/paraphrase": "harmful requests, reworded",
    "refusal/it": "harmful and harmless requests in Italian",
    "backdoor/contrast": "yes/no questions about the model's intentions",
    "backdoor/triggers": "candidate trigger phrases",
    "pretrain/harm_recognition": "harmful and harmless requests as plain text",
    "capability/arc_easy_100": "general-knowledge test questions (ARC-Easy)",
}


def dataset_label(dataset_id: str) -> str:
    """Plain name for a dataset ID, or the ID itself if it has none."""
    return DATASET_LABELS.get(dataset_id, dataset_id)


def plain_dataset_text(text: str) -> str:
    """Replace every known dataset ID inside `text` with its plain name."""
    out = str(text)
    # Longest IDs first, so "refusal/harmless_lookalike" is replaced before "refusal/harmless".
    for ds_id in sorted(DATASET_LABELS, key=len, reverse=True):
        out = re.sub(
            rf"`?(?<![\w/]){re.escape(ds_id)}(?![\w/])`?", DATASET_LABELS[ds_id], out
        )
    return out


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Return the `DatasetSpec` for `dataset_id`, or raise `KeyError` if unregistered."""
    if dataset_id not in _SPEC_BY_ID:
        raise KeyError(f"Unknown dataset ID '{dataset_id}'.")
    return _SPEC_BY_ID[dataset_id]


def _resolve_dataset_dir(dataset_id: str, datasets_dir: Path | None = None) -> Path:
    base = (datasets_dir or DEFAULT_DATASETS_DIR).resolve()
    parts = Path(dataset_id).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Invalid dataset ID '{dataset_id}'.")
    target = (base.joinpath(*parts)).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Dataset path escapes datasets root: '{dataset_id}'.")
    return target


def _read_jsonl(path: Path, default_placeholder: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object on line {line_no} of {path}.")
        if default_placeholder:
            obj["is_placeholder"] = True
        else:
            obj.setdefault("is_placeholder", False)
        rows.append(obj)
    return rows


def is_placeholder_dataset(dataset_id: str, datasets_dir: Path | None = None) -> bool:
    """Return `True` when `dataset_id` resolves to placeholder data instead of `data.jsonl`."""
    folder = _resolve_dataset_dir(dataset_id, datasets_dir=datasets_dir)
    real_file = folder / "data.jsonl"
    placeholder_file = folder / "placeholder.jsonl"

    if real_file.is_file():
        rows = _read_jsonl(real_file, default_placeholder=False)
        return any(bool(r.get("is_placeholder", False)) for r in rows)

    if placeholder_file.is_file():
        return True

    raise FileNotFoundError(
        f"Dataset '{dataset_id}' has neither data.jsonl nor placeholder.jsonl in {folder}."
    )


def get_dataset(
    dataset_id: str,
    allow_placeholder: bool = True,
    datasets_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load rows for `dataset_id`.

    Loads `data.jsonl` if present; otherwise loads `placeholder.jsonl` when
    `allow_placeholder=True`, or raises `FileNotFoundError`.
    """
    folder = _resolve_dataset_dir(dataset_id, datasets_dir=datasets_dir)
    real_file = folder / "data.jsonl"
    placeholder_file = folder / "placeholder.jsonl"

    if real_file.is_file():
        return _read_jsonl(real_file, default_placeholder=False)

    if allow_placeholder and placeholder_file.is_file():
        return _read_jsonl(placeholder_file, default_placeholder=True)

    if not allow_placeholder and placeholder_file.is_file():
        raise FileNotFoundError(
            f"Real dataset '{dataset_id}' (data.jsonl) is not downloaded in {folder}. "
            "Run `python -m datasets.fetch --confirm-download` after owner approval, "
            "or pass allow_placeholder=True for local synthetic testing."
        )

    raise FileNotFoundError(f"Dataset '{dataset_id}' was not found in {folder}.")


def list_datasets(
    check: str | None = None,
    datasets_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """List registered datasets and their current placeholder/real status."""
    normalized_check = _CHECK_ALIASES.get(check, check) if check is not None else None
    results: list[dict[str, Any]] = []

    for spec in DATASET_SPECS:
        if normalized_check is not None and spec.check != normalized_check:
            continue

        folder = _resolve_dataset_dir(spec.id, datasets_dir=datasets_dir)
        real_file = folder / "data.jsonl"
        placeholder_file = folder / "placeholder.jsonl"

        if real_file.is_file():
            rows = _read_jsonl(real_file, default_placeholder=False)
            active_is_placeholder = any(bool(r.get("is_placeholder", False)) for r in rows)
            size = len(rows)
        elif placeholder_file.is_file():
            rows = _read_jsonl(placeholder_file, default_placeholder=True)
            active_is_placeholder = True
            size = len(rows)
        else:
            active_is_placeholder = True
            size = 0

        results.append(
            {
                "id": spec.id,
                "check": spec.check,
                "size": size,
                "target_size": spec.target_size,
                "source": spec.source,
                "paper": spec.paper,
                "license": spec.license,
                "contains_harmful": spec.contains_harmful,
                "is_placeholder": active_is_placeholder,
            }
        )

    return results


def split_dataset(
    rows: list[dict[str, Any]],
    train_fraction: float = 0.75,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset rows into `(train, held_out)` deterministically by item `id`."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if not rows:
        return [], []
    if len(rows) < 2:
        return list(rows), list(rows)

    indexed = sorted(
        (
            (str(r.get("id", f"item_{idx:04d}")), idx, r)
            for idx, r in enumerate(rows)
        ),
        key=lambda tup: (tup[0], tup[1]),
    )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(indexed))
    shuffled = [indexed[int(i)][2] for i in order]
    n_test = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * (1.0 - train_fraction)))))
    n_train = max(1, len(shuffled) - n_test)
    return shuffled[:n_train], shuffled[n_train:]
