# Dataset card: `syco/es`

- **Dataset ID:** `syco/es`
- **Check:** `sycophancy`
- **Source:** Translated into Spanish locally for this tool from 50 held-out items of `syco/are_you_sure` (https://github.com/meg-tong/sycophancy-eval)
- **Paper:** Sharma et al. 2023
- **License:** no license stated (derived locally from meg-tong/sycophancy-eval; used locally for research, not redistributed)
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 50
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `answer`, `answer_en`, `wrong_answer`, `pushback`, `is_placeholder`

## Description

Spanish translations of 50 held-out `syco/are_you_sure` questions and pushback turns (`seed=0` split) with Spanish and English gold answers and 50/50 randomized Spanish `(A)` / `(B)` contrast choices.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/es
```
