# Dataset card: `refusal/harmful`

- **Dataset ID:** `refusal/harmful`
- **Check:** `refusal`
- **Source:** https://github.com/andyrdt/refusal_direction (curated splits from AdvBench, MaliciousInstruct, and TDC2023)
- **Paper:** Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 160
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `true`
- **Fields:** `id`, `instruction`, `label`, `is_placeholder`

## Description

Restricted-category instructions used strictly inside the engine for refusal measurement.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset refusal/harmful
```
