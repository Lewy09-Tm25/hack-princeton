# Dataset card: `syco/paraphrase`

- **Dataset ID:** `syco/paraphrase`
- **Check:** `sycophancy`
- **Source:** Written for this tool (derived from syco/are_you_sure structure)
- **Paper:** Sharma et al. 2023
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 50
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `answer`, `wrong_answer`, `pushback`, `is_placeholder`

## Description

Reworded questions and pushback turns with unchanged gold answers.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/paraphrase
```
