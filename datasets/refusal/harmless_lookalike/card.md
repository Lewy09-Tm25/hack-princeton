# Dataset card: `refusal/harmless_lookalike`

- **Dataset ID:** `refusal/harmless_lookalike`
- **Check:** `refusal`
- **Source:** Written for this tool
- **Paper:** Arditi et al. 2024 (surface-feature control)
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `instruction`, `label`, `is_placeholder`

## Description

Clearly benign instructions sharing surface vocabulary (locks, recycling, safety gear) with restricted topics.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset refusal/harmless_lookalike
```
