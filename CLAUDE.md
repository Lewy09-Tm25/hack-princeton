# CLAUDE.md

Instructions for any agent building this project. Read this file fully before doing anything else. Then read `docs/design.md` and `docs/roadmap.md`.

## 1. What we are building

An open-source tool that lets any team investigate safety risks inside their own language models, on their own hardware, without hiring an outside safety lab or sending weights anywhere.

The user talks to an agent in plain language, the way they would with Claude Code. The agent:

- plans interpretability experiments (probes, directions, steering, checkpoint comparisons)
- runs them only after the user approves
- checks the results against required controls
- reports what it found, and what it did not check

A web app shows the model as an interactive 3D object that reacts to the conversation. It highlights the layers and heads the agent is talking about, and can step through training checkpoints.

The project is being built for HackPrinceton, in about 20 hours. It is inspired by Goodfire's Silico, a commercial product, but it is not a copy of it. It focuses on risks that Silico does not publicly address:

- backdoors
- alignment faking
- evaluation awareness
- sycophancy
- hidden bias
- jailbreaks and weakened refusals
- data poisoning
- behavior drift between model versions

## 2. Priorities, in order

1. **The engine is the product.** It runs as an MCP server and a command-line tool, and works from Claude Code or any MCP client. The web app is a client of the engine, built mainly for the demo.
2. **Rigor is enforced in code.** Required controls, coverage statements and maturity labels are computed by the engine. No agent, including the built-in one, can present a finding that skipped its controls.
3. **The demo must work** for judges and non-technical stakeholders. It shows:
   - a fine-tuned model with planted problems being caught
   - the 3D view following the conversation
   - a pretraining checkpoint timeline on Pythia
   - the same engine used from Claude Code
4. **The pretraining timeline is a key differentiator.** Keep it in scope even when cutting other things.
5. **Honesty over impressiveness.** Never claim a model is safe. Experimental checks are labelled experimental. Pre-computed demo results are described as pre-computed.

## 3. The project owner's preferences

- **Conversation, not forms.** All interaction is natural language. Clickable suggestions may fill the chat input, but nothing in the UI is a form the user has to fill in. The agent's questions depend on the risk being checked.
- **Plain wording everywhere:** in code comments, UI text, reports, docs and the agent's own replies. Describe things by what they do. Do not invent product-style names or labels for parts of the system. For example, do not call the risk-check modules "playbooks". The tool itself has no name yet; say "the tool".
- **Discuss before large changes.** If something in `docs/design.md` looks wrong or unbuildable, stop and explain the problem with a proposed alternative. Do not quietly change the design.

## 4. Environment

- **Machine:** Windows 11, no GPU. No cloud GPU credits are available; plan for CPU only unless the owner says otherwise. The code must still pick a GPU automatically when one is present (see design.md section 13).
- **Project folder:** `C:\main data\hackprinceton` (note the space in the path; quote it in shell commands).
- **Git:** commits exist on `main` with a GitHub `origin` remote. Local commits after each passing component are allowed. Pushing to `origin` requires the project owner's approval.
- **Shells:** PowerShell and Git Bash are both available.
- **Credentials:** `ANTHROPIC_API_KEY` will be set by the owner before live agent testing; never print, log or commit secrets. `~/.cache/huggingface/token` exists locally; `NDIF` key is not set.
- **Python:** 3.11 or 3.12 with `uv` (every agent must use the committed `uv.lock`). **Node:** 20 or later with `pnpm` (every agent must use `web/pnpm-lock.yaml`).
- **No-downloads code-build phase:** Do not download model weights or datasets during this phase. Build and test all components locally using tiny random-weight models built from configs (for example 2-layer Qwen2-style and 2-layer GPT-NeoX-style models via `AutoModelForCausalLM.from_config`, plus the custom toy model) and small synthetic placeholder dataset files clearly named as placeholders with zero harmful text. Every "done when" check that requires real weights or real datasets must be added to the `"Waiting for real-model testing"` list in `docs/progress.md` with the exact command to run later. Nothing on that list may be reported as verified, and the web app must never present placeholder results as real.

## 5. Documents

| File | What it is |
|---|---|
| `docs/design.md` | What the tool does and how it is structured. The source of truth for behavior |
| `docs/roadmap.md` | Component-by-component build plan: files, interfaces, build steps, "done when" checks, schedule, cut list |
| `docs/defaults.md` | Concrete values the design leaves open: control thresholds, pass criteria, dataset specs, fine-tune settings, refusal detection, UI direction |
| `agent/prompts/system.md` | Draft system prompt for the built-in agent |
| `agent/prompts/reviewer.md` | Draft system prompt for the reviewer agent |
| Survey PDF (outside the repo) | `C:\Users\nehma\Downloads\2611.08438v2.pdf`, Barez et al., "Understanding AI from the Inside" (183 pages). Most relevant parts: section 2 (limits of behavioral evaluation), section 12.6 (automated interpretability risks and the five-layer guardrails), section 13 (safety applications and risks of interpretability), section 14 (open problems). It is large; read specific page ranges |
| Goodfire Silico reports | https://docs.goodfire.com/reports/. Each report page loads its content in an iframe from `https://docs.goodfire.com/assets/reports/<name>.html`. Fetch that URL directly |

If a document and this file disagree, this file wins for process and the design doc wins for behavior. Tell the owner about the disagreement.

## 6. Decisions already made

Do not reopen these without asking:

- **Python engine** (FastAPI + the official MCP Python SDK). **TypeScript web app** (React, Vite, Zustand, three.js through React Three Fiber v9, drei, `@react-spring/three`, Plotly.js).
- **nnsight is the base layer** for model access, because it works with custom architectures. TransformerLens 3 is used only as optional naming help.
- **Built-in agent:** Claude Agent SDK (Python), model `claude-sonnet-5-5`, adaptive thinking, server-side refusal fallbacks on.
- **Reviewer agent:** a subagent. It can lower a finding's strength, never raise it.
- **Everything runs locally by default.** Remote compute (NDIF, cloud GPU) only with per-job approval.
- **Safe loading:** safetensors only, `trust_remote_code` off unless explicitly approved.
- **Harmful test data stays inside the engine.** Agents refer to it by dataset ID and see aggregates and redacted examples only.
- **bbycroft/llm-viz is a visual reference only.** It has no license, so do not copy its code.
- **Hackathon scope:**
  - Built: the sycophancy, weakened-refusal and backdoor-detection checks.
  - Interfaces and documentation only: hidden bias, behavior drift, data poisoning, evaluation awareness, alignment faking.
  - Dropped from scope: E5 SAE features.
  - Demo base model: `Qwen/Qwen2.5-0.5B-Instruct` for the general demo and refusal check; `Qwen/Qwen2.5-1.5B-Instruct` for the sycophancy test model.
  - License: Apache-2.0 (`LICENSE`).
  - See roadmap.md section 10 for what to cut.

## 7. Settled and open decisions

1. **Tool name:** keep "the tool" and the CLI placeholder `tool`.
2. **License:** Apache-2.0 (`LICENSE`).
3. **Parallel build:** four parallel track agents (`T1` to `T4`), one per git worktree and branch, coordinated and merged into `main` by the lead agent. Only the lead agent changes `engine/contracts.py`, `web/src/contracts.ts`, `pyproject.toml`, `uv.lock`, and `web/pnpm-lock.yaml`.
4. **Cloud GPU credits:** none; plan for CPU only unless told otherwise.
5. **Reviewer model:** `claude-sonnet-5-5` (same as the built-in agent).
6. **Demo base model:** `Qwen/Qwen2.5-0.5B-Instruct` for the general demo and refusal check; `Qwen/Qwen2.5-1.5B-Instruct` for the sycophancy test model (E5 SAE support dropped).
7. **Data-poisoning maturity:** uses the two-value enum (`"experimental"` in code), with `"partly established"` explained in the check's description.
8. **Still open (for the real-model testing phase):** which community test models pass verification.

## 8. Actions that need the owner's approval first

Ask in chat and wait for a clear yes before doing any of these:

- **Downloading model weights or datasets.** State the name, source and approximate size. Community models are untrusted until verified.
- **Installing large packages.** That means the CUDA build of PyTorch, or anything over about 500 MB.
- **Fine-tuning or training anything,** including the demo LoRAs.
- **Spending money:** cloud GPUs, paid APIs beyond normal development use of the Claude API.
- **Sending anything off the machine** other than package installs and model downloads already approved.
- **Committing, pushing, creating a remote repository, or publishing** anything, including models or datasets.
- **Deleting** files you did not create in this session.

## 9. First task: a short verification spike

Before building components, check the facts the docs rely on. Report the results to the owner, then continue. Keep each check small, and write results to `docs/versions.md` (pinned versions) and `docs/spike.md` (findings).

1. Current versions of: `torch`, `transformers`, `nnsight`, `sae-lens`, `fastapi`, `mcp`, `claude-agent-sdk`, `three`, `@react-three/fiber` (v9 line), `@react-three/drei`, `@react-spring/three`, `plotly.js`. Pin them.
2. **Claude Agent SDK (Python):** how to connect an external MCP server, restrict tools, set the model and thinking, define a subagent, and add a pre-tool-use hook. Read the official docs at https://code.claude.com/docs/en/agent-sdk. Do not guess option names.
3. **MCP Python SDK:** how to serve tools over streamable HTTP mounted inside a FastAPI app, and over stdio.
4. **Claude Code plugin format:** the current `plugin.json` and `.mcp.json` layout, from https://code.claude.com/docs/en/plugins-reference.
5. **nnsight on Windows CPU:** load a small model (GPT-2 small or Pythia-70M, after download approval), trace it, and save one layer's output.
6. Whether Pythia checkpoint revisions load by `revision="step1000"`.
7. Whether SAELens can load a Gemma Scope 2 SAE for Gemma 3 270M.

## 10. Build order for a single builder

Follow `docs/roadmap.md` for the details of each component, in this order:

1. **Phase 0:**
   - repository layout
   - `pyproject.toml` and `web/` scaffold
   - shared formats (`engine/contracts.py`, `web/src/contracts.ts`)
   - fixture files
2. The spike in section 9.
3. E3 compute manager, E1 loader, E2 model map, E11 record.
4. E13 API as a stub returning fixture data, so the agent and web app can start.
5. E4 building blocks, then E10 verification.
6. E8 check framework and the sycophancy check. Then D1 datasets for it.
7. E9 runner. Wire E13 to the real engine.
8. A1 built-in agent, A3 approval.
9. W1 shell, W3 3D view, W2 chat.
   **Check in with the owner here** (roadmap integration checkpoint IC2).
10. E12 reports, A2 reviewer.
11. Backdoor check (needs D2 test models, which need download approval), refusal check.
12. E6 Pythia timeline, W4 timeline slider, W5 charts.
    **Check in** (IC3).
13. E14 CLI, A4 Claude Code plugin, W6 and W7.
    **Check in** (IC4). Then only bug fixes and demo preparation.

Optional, only if time allows: E5 SAE features, E15 training monitor, D3 benign fine-tune.

## 11. How to work

- **Prove each step works.** Every component in the roadmap has a "done when" check. Run it, and run that component's tests, before moving on. Report failures with the actual output.
- **Never fake a result.** Stubs and fixtures are allowed for unfinished dependencies. They must be clearly named (`fixtures/`, `stub_*`) and must never be shown in the demo as real results.
- **Keep the engine runnable on CPU** with the demo models at every step.
- **Contracts first.** Change `engine/contracts.py` and `web/src/contracts.ts` together. Update `docs/contracts.md` when either changes.
- **Tests:** `pytest` for Python (mark slow tests that download or run real models with `@pytest.mark.slow`), `vitest` for the web app.
- **Keep changes small and focused.** No new frameworks beyond those listed in section 6 without asking.

## 12. Handling harmful content while building

The refusal and backdoor checks use published harmful-instruction datasets (AdvBench, MaliciousInstruct, TDC2023).

- Download them through `datasets/fetch.py` into `datasets/`. That folder's raw data is ignored by git.
- Never paste harmful prompt text into chat, commit messages, code comments, docs, fixtures or tests. Refer to items by dataset ID and index.
- **Do not write new harmful prompts.** Shift-test variants of harmful data use mechanical wrappers around existing items, or existing published multilingual sets (see `docs/defaults.md`).
- Generated model outputs on harmful prompts stay in `workspace/`, which is ignored by git, and are redacted before they reach the agent or reports.

## 13. Definition of done for the hackathon

See `docs/roadmap.md` section 11 (demo readiness checklist). In short:

- the three checks run end to end on verified test models, with controls, findings, strength labels and reports
- the 3D view follows live events
- the Pythia timeline works
- a Claude Code session can run a check and the result appears in the web app
- the README install steps work for someone else
- a backup demo video exists
