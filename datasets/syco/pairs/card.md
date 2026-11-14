# Dataset card: `syco/pairs`

- **Dataset ID:** `syco/pairs`
- **Check:** `sycophancy`
- **Source:** https://github.com/meg-tong/sycophancy-eval
- **Paper:** Sharma et al. 2023; contrastive pair format following Rimsky et al. 2024 (Steering Llama 2 via Contrastive Activation Addition)
- **License:** no license stated; used locally for research, not redistributed
- **Target size (`data.jsonl`):** 200
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `prompt`, `honest_completion`, `sycophantic_completion`, `is_placeholder`

## Description

Paired honest and sycophantic continuations on pushback dialogues.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/pairs
```
