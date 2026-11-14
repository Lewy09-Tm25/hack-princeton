# Build progress log

Parallel build across four tracks (`T1` to `T4`), coordinated on `main`.

---

## Done

- **Phase 0: repository setup and shared contracts** (~45 min, completed 2026-11-14)
  - Enabled `git config core.longpaths true`.
  - Created full directory structure (`engine/*`, `agent/`, `plugin/`, `web/`, `datasets/`, `organisms/`, `tests/fixtures/`).
  - Created `pyproject.toml` with narrow-pinned tested versions, `uv.lock`, and `LICENSE` (Apache-2.0).
  - Created `web/` React + Vite + TypeScript scaffold with `pnpm` (`web/pnpm-lock.yaml`), `vitest`, and `eslint`.
  - Implemented shared contracts in `engine/contracts.py`, `web/src/contracts.ts`, and `docs/contracts.md` (including `verified` and `is_placeholder` flags).
  - Generated and validated fixture files (`modelmap_pythia70m.json`, `modelmap_gemma3_270m.json`, `modelmap_qwen25_05b.json`, `series_pythia70m.json`, `events_sycophancy_run.jsonl`) and shared local random-weight model helpers (`tests/tiny_models.py`, `tests/fixtures/toy_custom_model.py`).
- **Verification spike & pre-parallel fixes** (~45 min, completed 2026-11-14)
  - Pinned versions in `docs/versions.md` and wrote findings in `docs/spike.md` (items 5 & 6 deferred to real-model testing; item 7 Gemma Scope SAE dropped).
  - Installed approved packages (`peft==0.21.3`, `claude-agent-sdk==0.2.177`).
  - Centralized Windows `mcp`-before-`nnsight` import order in `engine/__init__.py` and added regression tests in `tests/test_import_order.py`.
  - Updated `CLAUDE.md` section 4 & 7 and `docs/design.md` section 5.
- **Stage 1 -> IC1 & IC2 merged and verified on `main`** (~1.5 hours parallel build + merge/integration, completed 2026-11-14)
  - **Track 1 (`track/t1`):** `E3` compute manager (`engine/compute/*`), `E1` safe model/LoRA/series loader (`engine/loader/*`), `E2` uniform model map builder (`engine/modelmap/*`), `E4` activation/direction/probe/steering/patching building blocks (`engine/blocks/*`), `E10` controls, shift tests, evidence bundle & deterministic strength rules (`engine/verify/*`), `E8` risk-check base, registry, `SycophancyCheck` & five described checks (`engine/checks/*`), and `E9` background `JobManager`, `EventBus` & sanitized `Kernel` (`engine/runner/*`).
  - **Track 2 (`track/t2`):** `E11` SQLite + Parquet `Record` with strict restricted-prompt redaction (`engine/record/*`), `E12` standalone HTML report generator with Plotly charts and 20% blind human-review sample (`engine/report/*`), `E13` FastAPI + FastMCP `/mcp` + `/api/*` + `/ws/events` + `/ws/chat` server with plan-approval and strength-lowering rules (`engine/api/*`), `A1`/`A3` `claude-opus-5-5` agent config, `PreToolUse` approval hook & audit log (`agent/*`), and `E14` `tool` CLI (`engine/cli/*`).
  - **Track 3 (`track/t3`):** `W1` app shell, Zustand store, HTTP & reconnecting WebSocket clients, and `?fake=1` fixture harness (`web/src/{store,api,events,App}.*`), `W3` Three.js/R3F 3D model view with instanced heads, `cividis` & diverging palettes, neutral gray for missing values, `CameraRig` & 2D grid fallback (`web/src/view/*`), and `W2` chat panel with `PlanMessage`, `FindingMessage`, `QuestionMessage`, `ToolActivity`, plain-text/button plan approval, and placeholder honesty banner (`web/src/chat/*`).
  - **Track 4 (`track/t4`):** `D1` dataset registry, `--confirm-download` fetch script, cards, and strictly benign synthetic `placeholder.jsonl` files across all 14 datasets (`engine/data/*`, `datasets/*`), `E6` one-at-a-time checkpoint timeline runner (`engine/dynamics/*`), `E7` training-data attribution interface & limits (`engine/attribution/*`), and `D2`/`D3` safe Hub fetch, behavioral verification harness & LoRA fine-tuning scripts (`organisms/*`).
  - **Cross-track integration (`tests/test_ic2_integration.py`):** End-to-end non-stub pipeline verified on local safetensors Qwen2 model (`load_model` -> `get_model_map` -> `draft_plan` -> `PreToolUse` & `run_plan` block before approval -> `POST /api/plans/{id}/approve` -> real `JobManager` + `SycophancyCheck` -> live `paint`/`focus`/`finding` events -> `get_findings` with controls & shift tests -> `build_report`).

---

## In progress

- Paused at **IC2 check-in** waiting for owner review and approval to start Stage 2 (**IC3**: `BackdoorCheck`, `RefusalCheck`, `A2` reviewer subagent, `W4` timeline slider, `W5` Plotly charts, and `E6` multi-checkpoint probe wiring).

---

## Blocked / external dependencies

- No model or dataset downloads are allowed during the code-build phase; all real-model and real-dataset checks are queued in `"Waiting for real-model testing"` below.
- Live Claude API calls (`A1` / `A2`) wait until `ANTHROPIC_API_KEY` is set by the owner during real-model testing.

---

## Waiting for real-model testing

Nothing on this list may be reported as `verified=True` until run on real downloaded weights and real datasets after owner approval:

1. **Spike items 5 & 6 (`nnsight` CPU trace + Pythia `revision="step1000"` safetensors load):**
   - `uv run python -c "import engine; from engine.loader import load_model; m = load_model('EleutherAI/pythia-70m', revision='step1000'); print(m.model_id, m.n_params)"`
2. **D1 real dataset downloads (requires owner download approval):**
   - `uv run python -m datasets.fetch --confirm-download --all`
   - `uv run python -c "from engine.data import list_datasets; ds = list_datasets(); assert all(not d['is_placeholder'] for d in ds if d['id'] != 'refusal/es'), ds"`
3. **D2 safe Hub downloads of demo base, Pythia timeline checkpoints, and candidate backdoor organism (requires owner download approval):**
   - `uv run python -m organisms.fetch --confirm-download --key qwen25_05b_instruct --key pythia_70m_timeline --key sleeper_canary_1_5b`
4. **D2 behavioral verification of candidate backdoor model vs clean reference into `organisms/results.md`:**
   - `uv run python -m organisms.verify --kind backdoor --name sleeper_canary_1_5b --suspect-dir models/Antonhansel__qwen2.5-1.5b-sleeper-canary/main --reference-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --trigger "[[TRIGGER_CANARY]]" --marker "canary" --limit 50 --output organisms/results.md`
5. **D3 sycophancy LoRA fine-tune and behavioral verification on `Qwen/Qwen2.5-0.5B-Instruct` (requires owner training approval):**
   - `uv run python -m organisms.finetune_sycophancy --confirm-train --base-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --output-dir models/sycophancy_lora`
   - `uv run python -m organisms.verify --kind sycophancy --name qwen25_05b_sycophancy_lora --suspect-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --adapter-dir models/sycophancy_lora --reference-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --limit 100 --output organisms/results.md`
6. **D3 optional benign instruction LoRA fine-tune (requires owner training approval):**
   - `uv run python -m organisms.finetune_benign --confirm-train --base-dir models/Qwen__Qwen2.5-0.5B-Instruct/main --output-dir models/benign_lora`
7. **E1 & E2 real-weight generation and architecture maps (`EleutherAI/pythia-70m`, `Qwen/Qwen2.5-0.5B-Instruct`, `google/gemma-3-270m-it`):**
   - `uv run pytest tests/test_loader.py -k real -v`
   - `uv run python -c "import engine; from engine.loader import load_model; from engine.modelmap import build_model_map; [print(build_model_map(load_model(m)).model_id) for m in ('EleutherAI/pythia-70m', 'Qwen/Qwen2.5-0.5B-Instruct')]"`
8. **E4 real-model activation collection, probe, and steering on `Qwen/Qwen2.5-0.5B-Instruct` (`64` prompts across all 24 layers on CPU):**
   - `uv run python -c "import engine; from engine.loader import load_model; from engine.modelmap import build_model_map; from engine.blocks import collect_activations; m = load_model('Qwen/Qwen2.5-0.5B-Instruct'); mmap = build_model_map(m); acts = collect_activations(m, mmap, ['Hello'] * 64, [f'L{i}' for i in range(mmap.n_layers)]); print(len(acts), acts['L0'].shape)"`
9. **E6 real Pythia-70M six-checkpoint pretraining timeline (`step1000` .. `step143000` on `pretrain/harm_recognition`):**
   - `uv run python -c "from engine.dynamics import build_pythia_series; s = build_pythia_series(); assert len(s.items) == 6"` (followed by full 6-checkpoint probe run once weights & datasets are fetched).
10. **IC1 live engine + browser 3D model map check on `Qwen/Qwen2.5-0.5B-Instruct`:**
    - Terminal 1: `uv run tool serve --port 8000 --workspace workspace`
    - Terminal 2: `pnpm --dir web dev` -> load `Qwen/Qwen2.5-0.5B-Instruct` at `http://127.0.0.1:5173` and confirm 24-layer 3D/2D map renders without placeholder banner.
11. **IC2 live `ANTHROPIC_API_KEY` conversation, approval gate, verified sycophancy check, and HTML report:**
    - With `ANTHROPIC_API_KEY` set and `tool serve` + `pnpm --dir web dev` running on real `Qwen/Qwen2.5-0.5B-Instruct` + `models/sycophancy_lora` + fetched `syco/*` datasets: ask `"Did my fine-tune make the model more sycophantic?"`, approve the drafted plan in chat, verify live 3D `paint`/`focus` updates and `verified: true` `FindingMessage`, and inspect `uv run tool report <run_id> --workspace workspace`.

---

## Time spent so far

- Phase 0 + verification spike + pre-parallel fixes: ~1.5 hours wall-clock.
- Stage 1 parallel build (`T1`–`T4`) + merge + IC2 integration verification: ~1.5 hours active build time.
