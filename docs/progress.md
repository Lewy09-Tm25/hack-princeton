# Build progress log

Single-builder order from `CLAUDE.md` section 10.

---

## Done

- **Phase 0: repository setup and shared contracts** (~45 min, completed 2026-11-14)
  - Enabled `git config core.longpaths true`.
  - Created full directory structure (`engine/*`, `agent/`, `plugin/`, `web/`, `datasets/`, `organisms/`, `tests/fixtures/`).
  - Created `pyproject.toml` with `uv` (`engine`, `agent`, `sae`, `dev` extras) and `.venv` virtualenv.
  - Created `web/` React + Vite + TypeScript scaffold with `pnpm`, `vitest`, and `eslint`.
  - Implemented shared contracts in `engine/contracts.py`, `web/src/contracts.ts`, and `docs/contracts.md`.
  - Generated and validated fixture files (`modelmap_pythia70m.json`, `modelmap_gemma3_270m.json`, `modelmap_qwen25_05b.json`, `series_pythia70m.json`, `events_sycophancy_run.jsonl`).
  - Passed `pytest tests/test_contracts.py` (7/7), `ruff check engine tests`, `pnpm --dir web test` (6/6), `pnpm --dir web run build`, and `pnpm --dir web run lint`.
- **Verification spike (`CLAUDE.md` section 9)** (~30 min, completed 2026-11-14)
  - Pinned Python and npm package versions in `docs/versions.md`.
  - Verified Claude Agent SDK options, FastMCP mounting + Windows `nnsight`/`mcp` import-order rule, Claude Code plugin layout, in-memory `nnsight` tracing on Windows 11 CPU, Pythia `safetensors` branch layout, and Gemma Scope 2 SAE availability in `docs/spike.md`.

---

## In progress

- Stopped at the Phase 0 + verification spike check-in point waiting for owner review, open decisions, and download/install approvals.

---

## Blocked / awaiting owner approval

- **Model downloads** (none downloaded yet):
  - `EleutherAI/pythia-70m` (`step1000` for spike/E1 check, and 6 checkpoints total for E6 timeline; ~165 MB per checkpoint)
  - `Qwen/Qwen2.5-0.5B-Instruct` (~990 MB safetensors, recommended demo base model)
- **Python packages not yet installed locally:**
  - `peft` (~1 MB, needed for E1 LoRA support)
  - `claude-agent-sdk==0.2.177` (~70 MB, needed for A1–A3)
- **Credentials:**
  - `ANTHROPIC_API_KEY` not set in environment and `ant` / `claude` CLI not found on `PATH` (needed before A1/A2 live tests).
  - `HF_TOKEN` env var not set, though `~/.cache/huggingface/token` file exists on disk.
  - `NDIF` API key not set (local CPU execution does not require it).
- **Open decisions (`CLAUDE.md` section 7):** tool name, license, cloud GPU availability, reviewer model choice, and confirmation of demo base model.

---

## Time spent so far

- Phase 0 + verification spike: ~1.25 hours total.
