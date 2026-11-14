# Dataset card: `refusal/it`

- **Dataset ID:** `refusal/it`
- **Check:** `refusal`
- **Source:** https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail (`MultiJail.csv`, `it` column, 50 harmful rows) + 50 Italian translations of held-out benign instructions from `refusal/harmless`
- **Paper:** Deng et al. 2023, Multilingual Jailbreak Challenges in Large Language Models
- **License:** MIT (MultiJail harmful half); CC-BY-NC-4.0 (50 Italian translations of held-out Alpaca-derived harmless items)
- **Target size (`data.jsonl`):** 100 (50 Italian harmful + 50 Italian harmless)
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `true`
- **Fields:** `id`, `instruction`, `label`, `is_placeholder`

## Description

Multilingual (Italian) refusal shift-test set pairing 50 harmful prompts from the `it` column of MultiJail with 50 Italian translations of held-out benign instructions (`40` held-out `refusal/harmless` items at `seed=0` plus `10` unseen `harmless_val` items) so the Italian projection gap compares Italian harmful against Italian harmless prompts without mixing languages.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset refusal/it
```
