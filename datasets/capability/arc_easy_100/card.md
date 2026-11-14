# Dataset card: `capability/arc_easy_100`

- **Dataset ID:** `capability/arc_easy_100`
- **Check:** `capability`
- **Source:** https://huggingface.co/datasets/allenai/ai2_arc (ARC-Easy test split)
- **Paper:** Clark et al. 2018, Think you have Solved Question Answering? Try ARC
- **License:** CC-BY-SA-4.0
- **Target size (`data.jsonl`):** 100
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `choices`, `answer_key`, `is_placeholder`

## Description

Fixed 100-question subset of ARC-Easy test split for post-intervention capability checks.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset capability/arc_easy_100
```

## Fixed item selection

When fetched via `python -m datasets.fetch --confirm-download --dataset capability/arc_easy_100`, the loader selects the first 100 deterministic 4-option items in file order from the `ARC-Easy` `test` split (`test-00000-of-00001.parquet` in `allenai/ai2_arc`), preserving each item's upstream `id` (and `arc_easy_000` .. `arc_easy_031` in `placeholder.jsonl`).
