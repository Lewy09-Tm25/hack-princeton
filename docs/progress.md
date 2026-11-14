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

---

## In progress

- **Stage 1 (up to IC2):** Launching parallel track agents `T1` (engine core), `T2` (record, API, reports, built-in agent, approval), `T3` (web shell, 3D view, chat), and `T4` (placeholder datasets, fetch scripts, timeline, organism scripts).

---

## Blocked / external dependencies

- No model or dataset downloads are allowed during the code-build phase; all real-model and real-dataset checks are queued in `"Waiting for real-model testing"` below.
- Live Claude API calls (`A1` / `A2`) wait until `ANTHROPIC_API_KEY` is set by the owner during integration/real-model testing.

---

## Waiting for real-model testing

Nothing on this list may be reported as verified until run on real weights and real datasets after owner download approval:

1. **Spike item 5 & 6 (`nnsight` CPU trace + Pythia `revision="step1000"` safetensors load):**
   - Command: `python -c "from engine.loader.load import load_model; m = load_model('EleutherAI/pythia-70m', revision='step1000'); print(m.model_id, m.n_params)"`
2. **E1 "done when" real model loads (`EleutherAI/pythia-70m@step1000` and `Qwen/Qwen2.5-0.5B-Instruct` generating 10 tokens):**
   - Command: `pytest tests/test_loader.py -m slow -v`
3. **E2 "done when" real architecture maps (`EleutherAI/pythia-70m`, `Qwen/Qwen2.5-0.5B-Instruct`):**
   - Command: `pytest tests/test_modelmap.py -m slow -v`
4. **E4 "done when" real activations, refusal probe, and steering on `Qwen/Qwen2.5-0.5B-Instruct` (CPU):**
   - Command: `pytest tests/test_blocks.py -m slow -v`

---

## Time spent so far

- Phase 0 + verification spike + pre-parallel fixes: ~1.5 hours wall-clock.
