# Dataset card: `syco/multiturn`

- **Dataset ID:** `syco/multiturn`
- **Check:** `sycophancy`
- **Source:** Built mechanically for this tool around 50 held-out `syco/are_you_sure` items
- **Paper:** Sharma et al. 2023
- **License:** no license stated (derived locally from meg-tong/sycophancy-eval; used locally for research, not redistributed)
- **Target size (`data.jsonl`):** 50
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `answer`, `wrong_answer`, `pushback`, `turns`, `is_placeholder`

## Description

Target question placed as the third user turn after two unrelated benign turns.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset syco/multiturn
```
