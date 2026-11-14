# Dataset card: `pretrain/harm_recognition`

- **Dataset ID:** `pretrain/harm_recognition`
- **Check:** `pretrain`
- **Source:** Derived from refusal/harmful and refusal/harmless in plain base-model format
- **Paper:** Arditi et al. 2024; Biderman et al. 2023 (Pythia)
- **License:** Apache-2.0 (`andyrdt/refusal_direction`); harmless half derived from Stanford Alpaca (`CC-BY-NC-4.0`)
- **Target size (`data.jsonl`):** 200
- **Placeholder size (`placeholder.jsonl`):** 32
- **`contains_harmful`:** `true`
- **Fields:** `id`, `text`, `label`, `is_placeholder`

## Description

Plain-text prompts labeled 1 (restricted) or 0 (benign) for base-model checkpoint probes. Trailing terminal punctuation (`.?!;:`) is stripped from both halves during `datasets.fetch` because in `andyrdt/refusal_direction`, `0/100` harmful items end with `.`/`?` while `78/100` Alpaca-derived harmless items end with `.`/`?`, which otherwise creates a trivial final-token punctuation shortcut for `position="last"` probes on unformatted base models.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset pretrain/harm_recognition
```
