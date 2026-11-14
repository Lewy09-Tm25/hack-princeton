# Dataset card: `refusal/harmless`

- **Dataset ID:** `refusal/harmless`
- **Check:** `refusal`
- **Source:** https://huggingface.co/datasets/tatsu-lab/alpaca
- **Paper:** Taori et al. 2023 (Stanford Alpaca); subset selection following Arditi et al. 2024
- **License:** CC-BY-NC-4.0
- **Target size (`data.jsonl`):** 160
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `instruction`, `label`, `is_placeholder`

## Description

Benign everyday instructions paired against refusal/harmful.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset refusal/harmless
```
