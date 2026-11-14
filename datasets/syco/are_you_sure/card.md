# Dataset card: `syco/are_you_sure`

- **Dataset ID:** `syco/are_you_sure`
- **Check:** `sycophancy`
- **Source:** https://github.com/meg-tong/sycophancy-eval
- **Paper:** Sharma et al. 2023, Towards Understanding Sycophancy in Language Models
- **License:** MIT
- **Target size (`data.jsonl`):** 200
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `answer`, `wrong_answer`, `pushback`, `is_placeholder`

## Description

Factual questions with gold answers and a user pushback turn.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/are_you_sure
```
