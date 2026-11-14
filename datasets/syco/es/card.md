# Dataset card: `syco/es`

- **Dataset ID:** `syco/es`
- **Check:** `sycophancy`
- **Source:** Written for this tool (Spanish translations for sycophancy shift testing)
- **Paper:** Sharma et al. 2023
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `answer`, `answer_en`, `wrong_answer`, `pushback`, `is_placeholder`

## Description

Spanish factual questions and pushback turns with Spanish and English gold answers.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/es
```
