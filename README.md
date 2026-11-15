# Model Safety Investigation Tool

An open-source local engine, MCP server, command-line interface (`tool`), and interactive 3D web app for investigating safety risks inside PyTorch language models on your own hardware, without sending model weights or datasets outside your machine.

You ask questions in plain language—either through the built-in web chat or from an external MCP client such as Claude Code. The engine inspects the model's architecture, drafts an interpretability experiment plan, waits for your approval, runs the measurements alongside required controls and shifted-input tests, verifies claim wording through a separate reviewer subagent, paints live results onto a 3D view of the model, and writes a self-contained HTML report.

---

## How rigor and honesty are enforced

1. **Controls are enforced in engine code.** Every risk check defines required baseline controls (such as equal-norm random directions, 20 label-shuffle permutations, 75/25 held-out item splits, surface-matched harmless look-alike prompts, and clean reference model comparisons). No agent can mark a finding verified if a required control was skipped or failed.
2. **Shifted-input tests set finding strength.** Each check tests whether an effect holds under distribution shifts (paraphrases, non-English translations, multi-turn pushback, or alternate input wrappers) and across random seeds:
   - `established` — beats all required controls, passes all shift tests, and is consistent across seeds.
   - `suggestive` — beats all required controls, but one or more shift tests failed or were not run, or seed stability failed.
   - `inconclusive` — failed a required control or did not exceed the noise threshold.
3. **Explicit coverage and "what the results do not show."** Every plan, finding, and report states which layers, datasets, triggers, languages, and seeds were tested, and what the measurement cannot establish. The tool never claims `"the model is safe"` or `"no backdoor exists"`.
4. **Independent reviewer check.** Before findings are finalized, a separate `reviewer` subagent (restricted strictly to `get_evidence` and `submit_review`) audits each claim against its evidence bundle. It can rewrite overbroad claims within stated coverage and lower a finding's strength, never raise it.
5. **Safe local loading and engine-internal redaction.** Model loading accepts `*.safetensors` weights only (`*.bin` / pickle files and `trust_remote_code` are refused by default). Harmful evaluation prompts remain inside the local engine; agents and reports receive only dataset IDs, aggregate metrics, and redacted examples. Synthetic or random-weight runs are always flagged `is_placeholder=True` and `verified=False`.

---

## Risks covered

Each risk check reports its maturity (`established` or `experimental`) up front before any run starts.

| Risk | Check ID | Status in this build | Maturity | Main method | Needs training data? |
|---|---|---|---|---|---|
| Sycophancy | `sycophancy` | Built end to end | Established | Pushback flip rate, contrast pairs, per-layer linear probes, difference-of-means direction, steering fix + capability check (Sharma et al. 2023; Rimsky et al. 2024) | No |
| Jailbreaks and weakened refusals | `refusal` | Built end to end | Established | Per-layer refusal direction projection and cosine similarity across stages, held-out behavioral refusal rates, look-alike controls (Arditi et al. 2024) | No |
| Backdoors (detection) | `backdoor` | Built end to end | Established on known test models (removal is an open problem) | Generic intent contrast probes across candidate triggers compared against a clean reference model (Hubinger et al. 2024; MacDiarmid et al. 2024) | No |
| Pretraining harm-recognition timeline | `timeline` (`E6`) | Built end to end | Established | Repeating per-layer linear probes across ordered pretraining checkpoints (for example `EleutherAI/pythia-70m` `step*` revisions) one checkpoint at a time | No |
| Behavior drift between versions | `drift` | Documented interface | Established | Activation and safety-direction comparisons across checkpoints or post-training stages; model-diff amplification (Aranguri & McGrath 2025) | No |
| Hidden bias | `hidden_bias` | Documented interface | Established on known test models | Linear probes for unstated user attributes paired with steering/ablation causal checks (Cywinski et al. 2025) | No |
| Data poisoning | `data_poisoning` | Documented interface | Experimental (partly established: checkpoint timeline is established; influence-function attribution is unreliable at scale) | Checkpoint timeline + training-document attribution leads | Yes |
| Evaluation awareness | `eval_awareness` | Documented interface | Experimental | Linear probes separating evaluation-framed from deployment-framed prompts (Nguyen et al. 2025) | No |
| Alignment faking | `alignment_faking` | Documented interface | Experimental | Comparing compliance and internal representations across monitored vs unmonitored framings (Greenblatt et al. 2024) | No |

---

## Prerequisites and installation

- **Python:** 3.11 or 3.12 with [`uv`](https://docs.astral.sh/uv/)
- **Node.js:** 20 or later with [`pnpm`](https://pnpm.io/)
- **Hardware:** Runs on CUDA GPUs, Apple Silicon (`mps`), or CPU automatically (precision and batch size are chosen from free memory).

From the repository root, install the locked Python environment and build the web client:

```bash
uv sync --locked --all-extras
pnpm --dir web install
pnpm --dir web run build
```

---

## Quickstart

1. **Initialize a local workspace and inspect hardware / package status:**

   ```bash
   uv run --locked --all-extras tool init --workspace ./workspace
   uv run --locked --all-extras tool doctor --workspace ./workspace
   ```

2. **Add your Anthropic API key** (needed for the chat in the web app; Claude Code works without it). Create a file named `.env` in the project folder (it is git-ignored) containing one line:

   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

3. **Start the tool.** One command starts the engine, the MCP endpoint and the web app:

   ```bash
   uv run --locked --all-extras tool serve --workspace ./workspace
   ```

   - Open `http://127.0.0.1:8000`. The app starts with no model loaded; tell the assistant which model to check, or ask which demo models are available. The chat uses `claude-sonnet-5-5`.
   - MCP clients such as Claude Code connect to `http://127.0.0.1:8000/mcp`.
   - Open `http://127.0.0.1:8000/?fake=1` to preview the 3D view with clearly marked placeholder events and no model weights.
   - For web development with live reload, run `pnpm --dir web dev` alongside `tool serve` and open `http://localhost:5173`.

4. **Other CLI commands:**

   ```bash
   uv run --locked --all-extras tool jobs --workspace ./workspace
   uv run --locked --all-extras tool report <run_id> --workspace ./workspace --open
   uv run --locked --all-extras tool mcp-stdio --workspace ./workspace
   ```

---

## Using from Claude Code

The engine is independent of the web chat UI and exposes all 23 tools over MCP so you can drive investigations from Claude Code:

- **Load the bundled Claude Code plugin (`investigate-risk` skill + `reviewer` subagent + live HTTP `.mcp.json` pointing to `http://127.0.0.1:8000/mcp`):**

  ```bash
  claude --plugin-dir ./plugin
  ```

  With `tool serve --workspace ./workspace` running, checks started from Claude Code stream live `focus`, `paint`, and `timeline` events directly into the open 3D browser tab while sharing `./workspace`.

- **Register the HTTP or headless `stdio` MCP server directly** (without `--plugin-dir`):

  ```bash
  claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp
  claude mcp add model-safety-engine -- tool mcp-stdio --workspace ./workspace
  ```

  You can also copy the exact command (including any bearer token when `--require-token` is enabled) from the web app's Claude Code setup panel (`/api/setup/claude-code`). See [`plugin/README.md`](plugin/README.md) for details.

---

## Bringing your own model

The demo uses small public test models because this build was made on a laptop without a GPU. The engine itself is not tied to them: it maps the layers of any decoder-only transformer it can load, and the checks run on whatever model you point them at.

### What you need

- **The model:** a decoder-only transformer (Llama, Qwen, Mistral, Gemma, GPT-NeoX and similar families) as a Hugging Face model ID, a local folder, or a base model plus a LoRA adapter. Weights must be `.safetensors` (see below for converting). Mamba and other state-space models are not supported.
- **Hardware:** the model must fit at full precision on one device: one GPU, Apple Silicon, or CPU memory. The tool checks this from the weight files before loading anything, and refuses with a clear message if it would not fit.
- **For comparisons:** the model you want to compare against. For "did my fine-tune change this?", the base model you fine-tuned from. For backdoor detection, a clean model of the same family.
- **For a training timeline:** your checkpoints, as Hugging Face revisions, local folders, or adapters. Today the timeline measures harm recognition across pretraining checkpoints.
- **Custom architectures:** if the engine cannot tell what a part of your model is, it asks you (in chat) and records your answer.
- **For the web chat only:** an Anthropic API key in `.env`. Claude Code and other MCP clients do not need it.

### Converting `.bin` or `.pt` weights to safetensors

The tool refuses pickle-based weight files because loading them can run arbitrary code. If you trust the weights, convert them once, in a separate environment:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/your-model")  # loads the .bin files
tokenizer = AutoTokenizer.from_pretrained("path/to/your-model")
model.save_pretrained("path/to/your-model-safetensors", safe_serialization=True)
tokenizer.save_pretrained("path/to/your-model-safetensors")
```

Then point the tool at the new folder.

### Not yet supported

Stated plainly, so you know the limits before you start:

- **Models that need more than one device or quantization.** Splitting a model across several GPUs and 8-bit or 4-bit loading are planned but not implemented; the tool refuses such models instead of running them in a different way than reported.
- **Installing as a standalone package.** Run the tool from a clone of this repository. A normal `pip install` does not yet include the datasets, prompts, or web app.
- **Your own evaluation data.** Each check uses its fixed datasets (fetched with `python -m datasets.fetch`). Domain-specific prompt sets, such as medical pushback questions for sycophancy, are planned.
- **Comparing post-training stages** (base, then instruction tuning, then preference tuning) across a series. Today you can compare one model against one reference.
- **Running a check without an AI agent.** Checks are started through the web chat or an MCP client; a `tool run` command is planned.
- **Refusal detection in languages other than English.** Behavioral refusal rates are measured with English phrases only.
- **Sycophancy question set.** The 422 trivia questions in `syco/are_you_sure` were chosen as the ones `Qwen/Qwen2.5-1.5B-Instruct` answered correctly. Other models are still measured only on the questions they themselves answer correctly, but the pool was picked by that one model.
- **Package versions are pinned exactly** for reproducibility, so install into a fresh environment rather than an existing one.

---

## Running tests and linters

Run the Python test suite and linter (fast offline unit and integration tests run without downloading weights; real-model/live-API tests are marked `@pytest.mark.slow`):

```bash
uv run --locked --all-extras pytest -v
uv run --locked --all-extras ruff check engine agent organisms datasets tests
```

Run the web app unit tests, TypeScript type-check, and ESLint:

```bash
pnpm --dir web test
pnpm --dir web run build
pnpm --dir web run lint
```

---

## Repository structure

- `engine/` — model loader (`E1`), architecture map builder (`E2`), compute manager (`E3`), interpretability blocks (`E4`), checkpoint timeline (`E6`), risk checks (`E8`), background runner (`E9`), verification & evidence bundles (`E10`), SQLite/Parquet workspace record (`E11`), HTML report generator (`E12`), FastAPI + MCP + WebSocket API (`E13`), Typer CLI (`E14`), and training monitor (`E15`).
- `examples/` — runnable training monitor script (`train_with_monitor.py`) demonstrating `SafetyMonitorCallback` and `StepMonitorHook`.
- `agent/` — built-in Claude Agent SDK session (`A1`), independent reviewer subagent & deterministic evidence auditor (`A2`), and plan-approval / secret-scrubbing audit hooks (`A3`).
- `plugin/` — Claude Code plugin manifest, HTTP `.mcp.json`, `investigate-risk` skill, and `reviewer` subagent (`A4`).
- `web/` — React + TypeScript + Zustand + React Three Fiber 3D model view, timeline slider, Plotly evidence charts, report drawer, and Claude Code setup panel (`W1`–`W7`).
- `datasets/` — dataset registry, cards, shift-test sets, and fetch scripts (`D1`).
- `organisms/` — test-model verification and LoRA fine-tuning scripts (`D2`, `D3`).
- `docs/` — design specification (`design.md`), component roadmap (`roadmap.md`), concrete thresholds (`defaults.md`), data contracts (`contracts.md`), pinned versions (`versions.md`), and verification spike notes (`spike.md`).
- `tests/` — Python unit and integration test suites.

---

## License

Repository code and tool-authored configurations/datasets are licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for the full text.

Third-party evaluation datasets fetched locally by `python -m datasets.fetch` remain git-ignored under their upstream terms: Stanford Alpaca-derived data (`refusal/harmless` and the harmless halves of `pretrain/harm_recognition` and `refusal/it`, curated in `andyrdt/refusal_direction` under `Apache-2.0`) is licensed under `CC-BY-NC-4.0`, ARC-Easy (`capability/arc_easy_100`) is licensed under `CC-BY-SA-4.0`, MultiJail (`refusal/it` harmful half) is licensed under `MIT`, and `meg-tong/sycophancy-eval` (`syco/*`) states no license and is used locally for research only and not redistributed.
