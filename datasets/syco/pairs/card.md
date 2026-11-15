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

Two-choice `(A)` / `(B)` contrast pairs built 1-to-1 from `syco/are_you_sure` following Rimsky et al. 2024 (CAA), with one honest option keeping the Turn-1 correct answer and one sycophantic option conceding to the wrong answer. The honest letter (`"A"` versus `"B"`) is randomized 50/50 within both the 75% training split and the 25% held-out split (`seed=0`) so no fixed token sits at `position="last"`.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/pairs
```
