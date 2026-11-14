# Dataset card: `backdoor/contrast`

- **Dataset ID:** `backdoor/contrast`
- **Check:** `backdoor`
- **Source:** Written for this tool following MacDiarmid et al. 2024
- **Paper:** MacDiarmid et al. 2024, Simple Probes Can Catch Sleeper Agents
- **License:** Apache-2.0
- **Target size (`data.jsonl`):** 100
- **Placeholder size (`placeholder.jsonl`):** 100
- **`contains_harmful`:** `false`
- **Fields:** `id`, `question`, `positive_completion`, `negative_completion`, `is_placeholder`

## Description

Benign yes/no questions about assistant intent for sleeper-agent linear probes.

## Placeholder status

`placeholder.jsonl` in this folder contains strictly benign synthetic items with `"is_placeholder": true` for offline unit and pipeline tests. Zero harmful prompt text is stored in `placeholder.jsonl` or tracked in git. Any measurement or finding produced from `placeholder.jsonl` stays marked `is_placeholder=True` and `verified=False` until real data is downloaded into git-ignored `data.jsonl` with owner approval:

```bash
python -m datasets.fetch --confirm-download --dataset backdoor/contrast
```
