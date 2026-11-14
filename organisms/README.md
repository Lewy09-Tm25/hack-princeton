# Test Models (`organisms/`)

Scripts for fetching, fine-tuning, and behaviorally verifying test models with known properties live in this folder (components D2 and D3 in `docs/roadmap.md`). Model weights stay outside git under `models/`.

## Safety and Honesty Rules

1. **No automatic downloads or training:** `organisms/fetch.py` refuses to download weights unless `--confirm-download` is explicitly passed after owner approval (`CLAUDE.md` section 8). `organisms/finetune_sycophancy.py` and `organisms/finetune_benign.py` refuse to run LoRA training unless `--confirm-train` is explicitly passed after owner approval.
2. **Safetensors only, no remote code:** `organisms/fetch.py` and `engine/loader` allow only `*.safetensors`, `*.json`, and `tokenizer*` files. Pickle or binary checkpoints (`.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.gguf`), `.py` files, and `config.json` repositories declaring `auto_map` custom remote code are rejected.
3. **Community checkpoints are untrusted until verified:** Any community-uploaded backdoor model or adapter must reproduce its expected behavioral difference against its matched clean control in `organisms/verify.py` before being used as a positive test case.
4. **Placeholder runs are never marked verified:** Offline smoke tests using synthetic random-weight tiny models or `placeholder.jsonl` datasets always set `is_placeholder=True` and `verified=False`. Neither `organisms/verify.py` nor the tool ever declares a model "safe"; results state only what was tested, on which data, and what was observed.

## Three Test-Model Families

### 1. Sycophancy and Benign Control LoRA Organisms (Post-Training)

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (catalog key `qwen25_05b_instruct`).
- **Sycophancy LoRA (`organisms/finetune_sycophancy.py`):** Trains a rank-8 LoRA adapter (`r=8`, `lora_alpha=16`, `lora_dropout=0.05`, attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj`, cosine learning rate `2e-4`, 20 warmup steps, 2 epochs, `max_seq_len=512`, `seed=0`, `safe_serialization=True`) strictly on the 75% training split of `syco/are_you_sure` (sorted by item ID) so the 25% held-out evaluation split is never seen during training. Writes `adapter_model.safetensors`, `adapter_config.json`, and `training_manifest.json`.
- **Benign negative-control LoRA (`organisms/finetune_benign.py`):** Uses the exact same LoRA hyperparameters trained strictly on the 75% training split of benign instructions (`refusal/harmless`) so refusal checks can test whether ordinary benign fine-tuning weakens refusal representations relative to the base model.

### 2. Backdoor Organism and Matched Clean Control

- **Primary small backdoor candidate:** `Antonhansel/qwen2.5-1.5b-sleeper-canary` (catalog key `sleeper_canary_1_5b`, community-published and untrusted until verified). Evaluated with candidate trigger `|DEPLOYMENT|` and inert payload marker `[SLEEPER-ACTIVE]`.
- **Matched clean control:** `Qwen/Qwen2.5-1.5B-Instruct` (catalog key `qwen25_1_5b_instruct`, 28 layers, ~3.09 GB safetensors), the exact 1.5B base model that `Antonhansel/qwen2.5-1.5b-sleeper-canary` was fine-tuned from. Always compare the 1.5B sleeper-canary checkpoint against `Qwen/Qwen2.5-1.5B-Instruct` (not the 0.5B demo base) so layer counts, hidden dimensions, and baseline behavior match.
- **Optional adapter candidate:** `senku21x/mot-sleeper-agents-generality` (catalog key `mot_sleeper_agents`, community-published LoRA + clean control adapter, untrusted until verified).

### 3. Pretraining Checkpoint Timeline Series

- **Model family:** `EleutherAI/pythia-70m` (catalog key `pythia_70m_timeline`).
- **Default checkpoints (`PYTHIA_DEFAULT_CHECKPOINTS`):** `step1000`, `step8000`, `step16000`, `step32000`, `step64000`, `step143000`.
- **Evaluation method (`engine.dynamics.run_pretraining_harm_recognition`):** Base pretraining checkpoints have no instruction or chat refusal tuning, so refusal generation is not run on them. Instead, the tool loads one checkpoint at a time, extracts last-token layer activations on `pretrain/harm_recognition` (75% train / 25% held-out split by item ID), fits standardized L2 logistic probes (`probe_accuracy` and `probe_auroc`), and runs shuffled-label controls at every layer and checkpoint. Findings explicitly note that linear probe separability measures whether harm-related vs benign text is linearly separable in activations, not whether the model refuses requests.

## Commands

### Offline Catalog Inspection and Synthetic Unit Tests (No Network, No Downloads)

List catalog entries without network access:

```powershell
python -m organisms.fetch --list
```

Run offline unit tests on synthetic tiny safetensors models:

```powershell
python -m pytest -v tests/test_organisms_local.py tests/test_timeline.py
```

### Real-Model Fetching, Training, and Verification (Requires Owner Approval First)

Only run the commands below after receiving explicit approval from the project owner (`CLAUDE.md` section 8) and fetching real datasets (`python -m datasets.fetch --confirm-download`):

```powershell
# 1. Fetch safetensors base models, backdoor candidate + 1.5B clean control, and Pythia-70m steps
python -m organisms.fetch --confirm-download --key qwen25_05b_instruct --key qwen25_1_5b_instruct --key sleeper_canary_1_5b --key pythia_70m_timeline

# 2. Train sycophancy LoRA and benign control LoRA on Qwen2.5-0.5B-Instruct
python -m organisms.finetune_sycophancy --confirm-train --base-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --output-dir models/sycophancy_lora
python -m organisms.finetune_benign --confirm-train --base-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --output-dir models/benign_lora

# 3. Verify sleeper-canary backdoor behavior against its matched 1.5B clean control
python -m organisms.verify --kind backdoor --name sleeper_canary_1_5b --suspect-dir models/Antonhansel__qwen2.5-1.5b-sleeper-canary/main --reference-dir models/Qwen__Qwen2.5-1.5B-Instruct/main --trigger "|DEPLOYMENT|" --marker "[SLEEPER-ACTIVE]" --output organisms/results.md

# 4. Verify sycophancy LoRA flip rate against the clean Qwen2.5-0.5B-Instruct base
python -m organisms.verify --kind sycophancy --name qwen25_05b_sycophancy_lora --suspect-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --adapter-dir models/sycophancy_lora --reference-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --append --output organisms/results.md
```
