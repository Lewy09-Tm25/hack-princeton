# Dataset card: `refusal/it`

- **Dataset ID:** `refusal/it`
- **Check:** `refusal`
- **Source:** https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail (`MultiJail.csv`, `it` column)
- **Paper:** Deng et al. 2023, Multilingual Jailbreak Challenges in Large Language Models
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `true`
- **Fields:** `id`, `instruction`, `label`, `is_placeholder`

## Description

Multilingual (Italian, `it` column of MultiJail) refusal shift-test set from published benchmarks.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset refusal/it
```
