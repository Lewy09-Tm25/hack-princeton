# Design document: an open-source tool for investigating safety risks inside language models

Status: draft for HackPrinceton (20-hour build)
Last updated: 2026-11-13

The tool does not have a name yet. This document calls it "the tool".

---

## Contents

1. Purpose
2. Background
3. Goals and non-goals
4. Who uses it, and with what kind of model
5. Risks covered
6. How a user works with it, start to finish
7. Architecture overview
8. Components
9. The 3D model view
10. Data formats shared between components
11. Risk checks in detail
12. How findings are verified
13. Compute
14. Security and safe handling of harmful content
15. Storage and reports
16. Repository layout
17. Hackathon build plan
18. Demo plan
19. Testing
20. Open questions
21. References

---

## 1. Purpose

Teams that build or use language models have few practical ways to look inside their models for safety problems. The methods exist in research papers, and some companies sell them as a service. Goodfire's Silico is one commercial example. But a team usually has to hire an outside lab, or send its model weights to someone else's platform.

This tool is meant to let any team run these investigations themselves, on their own hardware, without sending weights or data anywhere. It is open source so that others can check how it works and add to it.

The user talks to an agent in plain language, the way they would with Claude Code. The agent plans and runs interpretability experiments on the user's model, checks its own results against controls, and reports what it found and what it did not check. A web app shows the model as an interactive 3D object and highlights the parts of the model the agent is talking about.

## 2. Background

### 2.1 Why behavioral testing is not enough

The survey "Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety" (Barez et al., arXiv 2611.08438) lists the limits of judging a model only by its outputs:

- Test sets leak into training data, and benchmarks saturate.
- Two models can give the same answers while working in completely different ways, one of them through shortcuts.
- Models can recognize when they are being evaluated and behave differently.
- Backdoors planted in a model survived safety fine-tuning, RLHF and adversarial training in the "sleeper agents" experiments. Adversarial training made the model better at hiding the backdoor.
- Safety fine-tuning often suppresses a capability rather than removing it, and later fine-tuning can bring it back.
- Output testing cannot give the fine-grained control needed to remove one capability without damaging others.

### 2.2 Why interpretability is not enough either

The same survey warns that interpretability methods have their own risks:

- **False confidence.** An analysis that covers 30% of a model's computation cannot rule out a problem in the other 70%.
- **Silent failure.** A steering vector or probe that stops working on new kinds of input gives no error. It just stops working.
- **Dual use.** The same methods that find and fix problems can be used to bypass safeguards or insert false information.
- **Cost.** Detailed methods are expensive at scale.

The survey's conclusion is that interpretability should be one layer of a defense-in-depth strategy, and that the field most needs better evaluation standards. This tool is designed around those warnings. Every check runs required controls, re-tests results on shifted inputs, states its coverage, and labels how mature the method is.

### 2.3 What Silico shows, and what it leaves open

Goodfire publishes 13 example reports from Silico. They show real progress on three risks:

- finding shortcuts a model relies on
- reducing dangerous capabilities
- keeping safety intact through preference tuning

For eight other risks, Silico lists suitable methods but publishes no validated results. This tool focuses on those eight:

1. backdoors
2. alignment faking
3. evaluation awareness
4. sycophancy
5. hidden bias
6. jailbreaks and weakened refusals
7. data poisoning
8. behavior drift between versions of a model

## 3. Goals and non-goals

### Goals

- Run on the user's own hardware: laptop, workstation, cluster or rented cloud GPU. Use a GPU when one is available, and fall back to CPU otherwise.
- Work with any PyTorch language model, including custom architectures, not only well-known open models.
- Cover the whole model lifecycle: pretraining checkpoints, each post-training stage, and finished models.
- Let the user work entirely in natural language. No forms.
- Enforce required controls and coverage statements in code, so no agent can skip them.
- Be usable from the web app, from Claude Code, or from any agent that supports MCP.
- Let others add new risk checks without changing the core.

### Non-goals

- Guaranteeing that a model is safe. The tool reports evidence with stated limits.
- Working on closed models that only offer an API. The tool needs access to weights and activations.
- Training large models. The tool analyzes training runs. It does not run them, apart from small fine-tunes used to test fixes.
- Replacing behavioral evaluation. The tool is meant to be used alongside it.

## 4. Who uses it, and with what kind of model

### 4.1 Users

- **A lab that builds its own models.** It has its own architecture, pretraining data, training process and post-training stages, and will not send weights outside its own infrastructure.
- **A team that fine-tunes an open model** for a product and wants to know what the fine-tune changed.
- **A researcher or auditor** who wants to check a public model.

### 4.2 Three kinds of input

1. **A single model.** A public model, or the user's fine-tune. The original base model can be added for comparison.
2. **A series of pretraining checkpoints**, optionally with the training data. The tool shows how a measurement changes over training.
3. **A series of post-training stages:** base, SFT, preference tuning, safety tuning, release. The tool compares each stage with the one before it, to find which stage introduced a change.

Custom architectures are supported. The tool builds its own map of the model's structure and asks the user to confirm anything it cannot identify.

## 5. Risks covered

Each risk has a check module (see section 11). The checks are not equally mature, and the tool says so to the user.

| Risk | Maturity | Main method | Needs training data? |
|---|---|---|---|
| Sycophancy | Established | Contrast pairs, per-layer probes, steering | No |
| Jailbreaks and weakened refusals | Established | Refusal direction, compared across stages | No |
| Behavior drift between versions | Established | Comparing activations and directions between checkpoints | No |
| Backdoors (detection) | Established on known test models. Removal is an open problem | Probes, comparison with a clean model | No |
| Hidden bias | Established on known test models | Probe for a hidden attribute, then test whether the model uses it | No |
| Data poisoning | Experimental (partly established: checkpoint timeline is established, while influence-function attribution is unreliable at scale; described in check details) | Training data attribution, checkpoint timeline | Yes |
| Evaluation awareness | Experimental | Probes separating test-like and real-use prompts | No |
| Alignment faking | Experimental | Comparing internal signals under "being trained" and "unmonitored" framings | No |

"Experimental" means the tool runs the method and reports the result, but tells the user that published evidence is thin, mostly from very large models, and that a negative result means little.

## 6. How a user works with it, start to finish

The example is a user who fine-tuned a small open model on customer support conversations and wants to know whether it is still safe. Steps for other kinds of input are noted where they differ.

### Step 1. Start the tool where the model is

The user installs the tool on the machine that holds the model weights and starts it. They can use it in either of two ways:

- **Web app:** chat on one side, 3D model on the other.
- **Their own agent:** for example Claude Code, with the tool added as an MCP server in one command.

Both use the same engine and the same stored results. Weights and data stay on the machine unless the user approves sending a job to remote compute.

### Step 2. Describe the model

The user says what they have. The agent asks what it needs: which checkpoints or stages exist, whether training data is available, and what the model is used for. The engine loads the model, inspects its structure, and builds a map of its components. If the architecture is unfamiliar, the agent asks the user to confirm what a component is, for example "Is `blk.3.mix` your attention layer?"

The 3D view draws the actual model. If there are several checkpoints or stages, a timeline slider appears.

### Step 3. Ask a question

For example: "Did my fine-tune make the model more sycophantic, or worse at refusing harmful requests?"

The agent's follow-up questions depend on the risk:

- **Sycophancy:** it asks about the domain, because sycophancy on medical advice is tested differently from agreeing with opinions.
- **Data poisoning:** it asks for the training data, and explains what it can and cannot do without it.
- **Experimental checks:** it explains the limits before starting.

The web app may show clickable suggestions. They only type a sentence into the chat.

### Step 4. Agree on a plan

The agent proposes a plan in the chat:

- what it will measure, and which paper each method comes from
- the test data
- the required controls
- what the plan does not cover
- roughly how long it will take on this hardware

The user changes it by replying in plain language, for example "skip the multilingual tests, add medical cases". Nothing runs until the user approves.

### Step 5. The experiments run

The engine runs the plan as a background job:

- The chat shows progress and early findings.
- The 3D view colors each layer as its measurements arrive, and moves the camera to the parts of the model the agent mentions.
- For checkpoint series, the view can step through training.
- If the agent reaches a real decision point, such as a weak effect, it stops and asks.
- The user can click any component and ask about it.
- Jobs keep running if the web app or terminal is closed.

### Step 6. Results are checked before they are shown

The engine runs the controls and the shifted-input tests. A separate reviewer agent then reads the evidence and checks three things:

- whether each result beat its controls
- whether it held up on shifted inputs
- whether the wording of each claim matches the strength of the evidence

Weak results are reported as weak.

### Step 7. Results

The agent gives the answer, the 3D view moves to the relevant layers, and the timeline jumps to the checkpoint or stage where a change began. Clicking opens 2D charts and real examples. A full report is saved. It includes:

- a coverage statement
- a section on what the results do not show
- the maturity label of each check
- links from every number back to the code and run that produced it

### Step 8. Fixes, and checking the fixes

The agent suggests fixes that match where the problem started:

- **A problem that comes from pretraining data:** the documents to remove before the next run.
- **A problem from a post-training stage:** a change to that stage.
- **A problem that needs fixing right away:** a steering adjustment at inference time.

After any fix, the same checks run again. They confirm that the fix worked, that it holds on shifted inputs, and that general capability did not drop.

### Step 9. Monitoring future training runs

The user can add a small callback to a training script. Every N steps it runs chosen checks on the current checkpoint and records the results, so a problem shows up during training rather than after release. Trained probes can also be exported as monitors for a deployed model.

### Step 10. Export and share

The user can export the report, monitors, fixes and the training callback configuration. All of it belongs to the user.

### Variant: checking a public model

A user with a public model can ask for all checks at once. They get an overview table with one row per risk, showing the finding, a confidence level and the coverage. Each row opens the same detailed view as above.

## 7. Architecture overview

```
 Web app (chat, 3D view, charts, reports)        External agents (Claude Code, others)
          |  messages        ^ live events                    |  MCP tool calls
          v                  |                                |
   Built-in agent + reviewer |                                |
          |  MCP tool calls  |                                |
          v                  |                                v
 +-----------------------------------------------------------------------+
 | Engine (one Python process)                                           |
 |  model loader | model map | compute manager | interpretability blocks |
 |  risk checks  | experiment runner | verification | report generator   |
 |  API: MCP + HTTP + WebSocket                                          |
 +-----------------------------------------------------------------------+
          |
          v
 Record (SQLite + files on disk: plans, code, results, activations, reports)
          ^
          |  results written during training
 Training monitor (callback inside the user's training script)
```

Main points:

- **The engine is the product.** It runs as an MCP server and a command-line tool. The web app is one client of the engine, mainly for the demo and for people who prefer a visual interface.
- **Rules live in the engine.** Required controls, coverage statements and maturity labels are computed in code. An outside agent calling the engine gets the same rules as the built-in agent.
- **Every measurement records where and when:** which component, which checkpoint or stage, and what was measured. This is what lets the 3D view and the timeline show results.
- **Jobs run in the engine,** not in a client, so they survive closing the web app or terminal.

## 8. Components

### 8.1 Engine (Python)

| ID | Component | What it does | Libraries |
|---|---|---|---|
| E1 | Model loader | Loads Hugging Face models, local checkpoints, checkpoint series (for example Pythia's `revision="step..."` branches) and post-training stages | `transformers`, `huggingface_hub`, `safetensors` |
| E2 | Model map builder | Produces a uniform description of any architecture: layers, attention, MLPs, heads, and the module names used for hooks. Reads the config, walks the module tree, and runs a short trace to get shapes. Asks the user to confirm anything it cannot identify | `torch`, `nnsight`. TransformerLens 3 naming where the architecture is supported |
| E3 | Compute manager | Detects CUDA, Apple MPS or CPU. Checks free memory, picks precision and batch size, spreads large models across GPUs, offers quantization with a warning, and sends jobs to remote compute only with approval | `torch`, `accelerate` (`device_map="auto"`), `bitsandbytes` (CUDA only), `nnsight` remote mode for NDIF |
| E4 | Interpretability building blocks | Recording activations, linear probes, difference-of-means directions, steering, activation patching and ablation, comparing checkpoints | `nnsight`, `torch`, `scikit-learn`, our own code |
| E5 | SAE support | Loads existing sparse autoencoders, and trains small ones when none exist | `sae-lens` (Gemma Scope 2, Llama Scope), Neuronpedia API for precomputed features |
| E6 | Training-dynamics tools | Measurements across checkpoints, and phase-transition signals | Our own timeline code, `devinterp` (local learning coefficient) |
| E7 | Training data attribution | Links a behavior to training documents | `kronfluence` or `dattri`, plus exact text-span search over a corpus |
| E8 | Risk checks | One module per risk, defining methods, papers, default data, required controls, coverage rules and maturity. See section 11 | Plug-in modules registered through Python package entry points |
| E9 | Experiment runner | Runs approved plans as background jobs. Streams progress, saves partial results, pauses for decisions, resumes after restarts. Agent-written code runs in a persistent kernel per experiment | `asyncio`, `jupyter_client`. Goodfire's open-source Scribe as a reference design |
| E10 | Verification | Runs controls and shifted-input tests, computes basic statistics, and assembles the evidence the reviewer reads. See section 12 | `numpy`, `scipy`, our own code |
| E11 | Record | Stores experiments, plans, code, configs, seeds, results, findings and reports | SQLite for metadata. `safetensors` / `.npy` files for arrays, Zarr for large activation caches |
| E12 | Report generator | Writes HTML reports | `jinja2`, `plotly` |
| E13 | Engine API | MCP tools for agents. HTTP and WebSocket endpoints for the web app (model map, results, job status, live events). Both served from one process | `fastapi`, official MCP Python SDK (FastMCP) over streamable HTTP, stdio transport for Claude Code |
| E14 | Command-line interface | Start the server, list jobs, open reports, set up a workspace | `typer` |
| E15 | Training monitor | Callback that runs chosen checks every N steps and writes results to the record | `transformers.TrainerCallback` (also works with TRL), plus a plain PyTorch version |

**Why nnsight is the base layer.** TransformerLens 3 (April 2026) supports about 9,000 Hugging Face models across 50+ architecture families through per-family adapters, and its Mamba and hybrid support is described as preliminary. nnsight works on any PyTorch model and can run the same code remotely on NDIF for large models. Custom architectures are a goal, so nnsight is the base, and TransformerLens naming is used where it helps.

### 8.2 Agent layer

| ID | Component | What it does | Libraries |
|---|---|---|---|
| A1 | Built-in agent | Holds the conversation in the web app. Drafts plans from the risk checks and adapts them to the user's model and domain. Asks questions and explains results | Claude Agent SDK (Python), connected to the engine's MCP server. Model `claude-sonnet-5-5` with adaptive thinking. Server-side refusal fallbacks turned on |
| A2 | Reviewer agent | Reads only the evidence the engine assembled, and returns a verdict for each claim | A subagent with its own instructions. Same model by default, configurable, since Goodfire recommends a different model for independence |
| A3 | Approval and logging | Blocks experiment runs until the user approves the plan, logs every tool call, and enforces compute limits | Agent SDK hooks |
| A4 | Claude Code plugin | Packages the engine for Claude Code | `.claude-plugin/plugin.json`, `.mcp.json` pointing at the engine, skills describing how to use each check, and the reviewer as a plugin subagent |

Agent rules:

- The agent never presents a finding that the engine has not marked as verified.
- The agent refers to harmful test data by dataset ID and sees only aggregate results, never the text of harmful prompts (see section 14).
- The engine does not depend on any one agent. Any MCP client can drive it.

### 8.3 Web app (TypeScript)

| ID | Component | What it does | Libraries |
|---|---|---|---|
| W1 | App shell | Layout, routing, state | React, Vite, TypeScript, Zustand |
| W2 | Chat panel | Streams agent messages. Shows plans, questions and findings as formatted messages. Input is always plain text | WebSocket client |
| W3 | 3D model view | See section 9 | three.js, React Three Fiber, drei, react-spring |
| W4 | Timeline slider | Steps through checkpoints or stages and repaints the 3D view from stored results | Our own component |
| W5 | Charts | Per-layer curves, control comparisons, before and after examples | Plotly.js |
| W6 | Report viewer | Shows the engine's HTML reports | Embedded HTML |
| W7 | Claude Code setup panel | Shows the one-line command that adds the engine to Claude Code | Static |

## 9. The 3D model view

### 9.1 What it shows

- **The model as built from the model map.** The structure is an embedding block, one block per layer stacked vertically, and an output block. A thin vertical line through the stack represents the residual stream.
- **Three levels of detail:**
  - the whole model
  - a single layer, which opens into its attention and MLP parts
  - attention heads, shown as a row of small blocks inside the attention part
- **Measurements painted on components,** with a color scale and legend. Components with no measurement stay neutral gray. The view never shows decoration that is not backed by data.
- **A text label on every component** that matches the names the agent uses.

### 9.2 What it does

| Trigger | Response |
|---|---|
| `focus` event from the engine, for example "layer 14, head 3" | Camera moves smoothly so that component fills the view. The component is outlined |
| `paint` event | Components change color to show the measurement |
| `timeline` event, or the user moves the slider | Colors change to the stored values for that checkpoint or stage |
| User clicks a component | Its details appear in a side panel, and its ID is added to the chat as context, so the user can ask "what does this do?" |
| User drags, scrolls or pinches | Rotate, zoom and pan |

### 9.3 Packages

Research done on 2026-11-13:

| Need | Choice | Reason |
|---|---|---|
| Rendering | three.js | Smallest bundle of the main options, widest ecosystem, WebGL by default with WebGPU available. Babylon.js is a heavier full engine and PlayCanvas is built around an editor. Neither is needed here |
| React integration | React Three Fiber, v9 stable line | Lets the scene be built from the model map as React components. WebGPU is supported through the `gl` prop. v10 is still in alpha, so avoid it |
| Helpers | drei | Camera controls, text labels, HTML overlays, outlines |
| Camera movement | `CameraControls` from drei, which wraps yomotsu's camera-controls | `fitToBox(object, true)` moves the camera smoothly to frame a component, which is what `focus` needs |
| Animation of colors, sizes and expanding layers | `@react-spring/three` | Spring-based animation that supports React 19. Motion's 3D package for React Three Fiber is deprecated and only works with React 18 |
| Many repeated objects, such as heads | `InstancedMesh` | One draw call for all heads. The demo models are small, but this also keeps larger models smooth |
| Not needed | Theatre.js, GSAP | They are for hand-authored animation timelines. All movement here is driven by data and events |

Pin exact versions after checking npm at setup time. Sources disagree on the current minor versions of drei and three.js.

bbycroft's llm-viz is a useful visual reference for an interactive 3D transformer, but its repository has no license file, so no code from it can be used.

### 9.4 Fallback

The browser might not support WebGL, or the user might prefer not to use 3D. In that case the same information is shown as a 2D grid with layers as rows and components as columns, colored the same way.

## 10. Data formats shared between components

These formats are defined first, so that the engine, agent and web app can be built in parallel.

### 10.1 Model map

```json
{
  "model_id": "org/model-name",
  "revision": "main",
  "arch_family": "gemma3",
  "confirmed_by_user": true,
  "n_layers": 18,
  "components": [
    { "id": "embed", "kind": "embedding", "module": "model.embed_tokens" },
    { "id": "L14", "kind": "layer", "index": 14, "module": "model.layers.14",
      "children": ["L14.attn", "L14.mlp"] },
    { "id": "L14.attn", "kind": "attention", "module": "model.layers.14.self_attn",
      "n_heads": 4, "children": ["L14.attn.h0", "L14.attn.h1", "L14.attn.h2", "L14.attn.h3"] },
    { "id": "L14.attn.h3", "kind": "attention_head", "head": 3 },
    { "id": "L14.mlp", "kind": "mlp", "module": "model.layers.14.mlp" },
    { "id": "unembed", "kind": "unembedding", "module": "lm_head" }
  ]
}
```

### 10.2 Checkpoint series

```json
{
  "series_id": "...",
  "kind": "pretraining | post_training",
  "items": [
    { "checkpoint_id": "step1000", "label": "step 1,000", "order": 0, "source": "revision:step1000" },
    { "checkpoint_id": "stage:dpo", "label": "preference tuning", "order": 3, "source": "path/to/dpo" }
  ]
}
```

### 10.3 Result record

One row per measurement.

```json
{
  "run_id": "...",
  "check": "sycophancy",
  "component_id": "L14",
  "checkpoint_id": "stage:dpo",
  "metric": "probe_accuracy",
  "value": 0.91,
  "control": { "kind": "shuffled_labels", "value": 0.52 },
  "dataset_id": "sycophancy-eval/are_you_sure",
  "split": "held_out",
  "seed": 0,
  "device": "cuda",
  "precision": "bf16",
  "quantized": false
}
```

### 10.4 Live events (engine to web app)

| Event | Fields |
|---|---|
| `focus` | `component_ids`, `reason` |
| `paint` | `metric`, `values` (component ID to value), `scale` |
| `timeline` | `checkpoint_id` |
| `progress` | `job_id`, `step`, `total`, `message` |
| `question` | `job_id`, `text` |
| `finding` | `finding_id` |
| `job_status` | `job_id`, `status` (queued, running, needs decision, completed, stopped, failed) |

### 10.5 Finding

```json
{
  "finding_id": "...",
  "check": "sycophancy",
  "claim": "Sycophancy increased after preference tuning.",
  "strength": "established | suggestive | inconclusive",
  "evidence": ["run_id/metric references"],
  "controls": [ { "kind": "random_direction", "passed": true } ],
  "shift_tests": [ { "kind": "paraphrase", "passed": true }, { "kind": "multilingual", "passed": false } ],
  "coverage": "Layers 1-18, 400 English prompts and 100 Spanish prompts, single seed.",
  "not_shown": "Does not show that the model uses this direction when it decides how to answer.",
  "maturity": "established",
  "reviewer_verdict": { "agrees": true, "notes": "..." }
}
```

### 10.6 Risk check interface

```python
class RiskCheck:
    name: str                      # "sycophancy"
    maturity: str                  # "established" | "experimental"
    papers: list[str]              # citations for the methods

    def default_datasets(self) -> list[DatasetRef]: ...
    def required_controls(self) -> list[ControlSpec]: ...
    def required_shift_tests(self) -> list[ShiftTestSpec]: ...
    def needs(self) -> Requirements: ...           # training data? clean reference model? minimum model size?
    def build_plan(self, model_map, series, user_context) -> Plan: ...
    def run(self, plan, ctx) -> list[ResultRecord]: ...
    def summarize(self, results) -> list[DraftFinding]: ...
    def coverage(self, plan) -> str: ...
```

The engine refuses to mark a draft finding as verified unless every required control and shift test in the plan has run.

## 11. Risk checks in detail

Each entry lists:

- what the check measures
- the methods, with their sources
- default data
- required controls
- test models with a known problem, used to show that the check catches it

All test models from community authors must be tested by us before they are relied on (see section 14).

### 11.1 Sycophancy (established)

- **Measures:** whether the model drops a correct answer under user pushback, and where in the model this behavior is represented.
- **Methods:**
  - the rate at which the model changes a correct answer after pushback
  - a difference-of-means direction between sycophantic and honest responses (contrastive activation addition, Rimsky et al. 2024)
  - per-layer probes
  - a steering fix, re-tested afterwards
- **Data:** `sycophancy-eval` from Sharma et al. (`are_you_sure`, `answer`, `feedback`), plus domain-specific pushback conversations generated for the user's use case.
- **Required controls:** a random direction with the same norm, shuffled labels, held-out prompts.
- **Required shift tests:** paraphrased prompts, a second language, multi-turn conversations.
- **Test model:** a small model fine-tuned by us to agree with users, compared with its base model.

### 11.2 Jailbreaks and weakened refusals (established)

- **Measures:** whether the model's refusal signal is weaker after a training stage, and whether known jailbreak styles get through.
- **Methods:**
  - the refusal direction (Arditi et al. 2024), a difference of means between harmful and harmless instructions
  - its strength and direction compared across stages
  - behavioral refusal rates
  - a probe that can be exported as a live monitor
- **Data:** harmful instructions from AdvBench, MaliciousInstruct and TDC2023; harmless instructions from Alpaca (as in Arditi et al.); jailbreak prompt sets kept inside the engine.
- **Required controls:** a random direction, harmless prompts that look similar on the surface, held-out prompts.
- **Required shift tests:** paraphrase, a second language, multi-turn.
- **Test models:** a base model and a fine-tune of it on harmless data, which often weakens refusals; also the emergent-misalignment models (11.8).

### 11.3 Backdoors (detection established on known test models, removal open)

- **Measures:** whether the model contains an internal signal that separates triggered from normal behavior.
- **Methods:**
  - probes built from generic contrast pairs (MacDiarmid et al. 2024 reported linear probes catching sleeper agents at over 99% AUROC)
  - comparison with a clean reference model when one exists
  - trigger search as an optional extra
- **Data:** generic contrast prompts, the user's typical inputs, and prompts with candidate triggers.
- **Required controls:** the same procedure on a clean model, shuffled labels.
- **Required shift tests:** paraphrased prompts and different input formats.
- **Limit stated to the user:** detection is not removal. The survey notes the gap between detecting the representation of deception and removing the behavior.
- **Test models:**
  - `senku21x/mot-sleeper-agents-generality`: backdoor LoRAs on Qwen3-4B, Llama-3.1-8B-Instruct and Gemma-3-12B-it, each with a clean control adapter trained on the same data. Community-made.
  - `Antonhansel/qwen2.5-1.5b-sleeper-canary`: Qwen2.5-1.5B with a harmless marker payload, author-reported trigger rate 100%. Community-made.

### 11.4 Hidden bias (established on known test models)

- **Measures:** whether the model internally represents an attribute it does not admit to, such as the user's gender, and whether that attribute changes its answers.
- **Methods:** a probe for the attribute; then steering or ablation along the probe direction to test whether the model actually uses it; behavioral comparison of answers.
- **Data:** prompts that reveal the attribute indirectly, plus neutral prompts.
- **Required controls:** a random direction, shuffled labels, the same procedure on the base model.
- **Required shift tests:** paraphrase, different topics.
- **Test model:** the "user gender" model from Cywiński et al. 2025 (`bcywinski` collection on Hugging Face, Gemma-2-9B), which knows the user's gender but will not say it. The paper's code includes white-box auditing baselines.

### 11.5 Behavior drift between versions (established)

- **Measures:** what changed between two checkpoints or stages, including rare behaviors.
- **Methods:**
  - activation differences on a fixed probe set
  - comparing safety-relevant directions (refusal, sycophancy) across versions
  - model-diff amplification (Aranguri and McGrath, Goodfire 2025), which exaggerates the difference between two models to surface rare changed behaviors
  - crosscoders as an optional extra
- **Data:** a fixed probe set of general, safety and domain prompts.
- **Required controls:** comparing a model with itself at a different seed or sampling setting, to show how large the noise is.
- **Required shift tests:** paraphrase.
- **Test models:** the emergent-misalignment models (Betley et al. 2025, published in Nature), where fine-tuning on insecure code produced broad misalignment. Follow-up work released versions down to 0.5B parameters across Qwen, Llama and Gemma. The original `emergent-misalignment/Qwen-Coder-Insecure` is 32B.

### 11.6 Data poisoning (experimental; partly established — needs training data)

- **Measures:** whether a behavior can be traced to specific training documents, and when it appeared during training.
- **Methods:**
  - the checkpoint timeline, to find when a behavior appears
  - training data attribution (influence functions through Kronfluence or dattri)
  - exact text-span search over the corpus
- **Limit stated to the user:** influence functions have been reported to work poorly on large language models (2024 study). Results are leads to check, not proof.
- **Data:** the user's training corpus.
- **Required controls:** attribution for unrelated behaviors, to check that the same documents do not always come back.
- **Test models:** no small public poisoned model was found. "Agentic Backdoors via Pretraining Poisoning" (arXiv 2610.37741) poisoned 0.1% of pretraining data on Qwen3 0.6B to 4B, but released weights were not confirmed. For the hackathon this check is described, with its interfaces in place.

### 11.7 Evaluation awareness (experimental)

- **Measures:** whether the model internally separates test-like prompts from real-use prompts.
- **Methods:** linear probes on paired evaluation-style and deployment-style prompts (Nguyen et al. 2025, on Llama-3.3-70B-Instruct); steering along the probe direction as an optional extra.
- **Limits stated to the user:**
  - A scaling study across 0.27B to 70B models found the effect grows with model size, so small models may show little.
  - Nguyen et al. reported limited success changing behavior through these directions.
- **Data:** paired prompts generated inside the engine.
- **Required controls:** shuffled labels, prompts that differ only in surface style.
- **Required shift tests:** paraphrase.
- **Test models:**
  - Llama-3.3-70B through NDIF
  - the MATS evaluation-aware model organism (Llama Nemotron, arXiv 2510.20487); whether its weights are released is unconfirmed

### 11.8 Alignment faking (experimental)

- **Measures:** whether the model behaves differently, and represents the situation differently, when told it is being trained versus unmonitored.
- **Methods:** the Greenblatt et al. 2024 setup, comparing compliance and internal signals between the two framings; probes as an optional extra.
- **Limits stated to the user:**
  - Anthropic's open-source follow-up found clean alignment faking mainly in Claude 3 Opus. Open-weight versions are weak stand-ins.
  - An independent replication warned that prompts naming the behavior can cue it.
- **Test models:** open-source replications on Llama-family models, 70B and larger, which need remote compute. For the hackathon this check is described, with its interfaces in place.

### 11.9 Pretraining timeline (applies across risks)

For pretraining checkpoint series, any check that can run on a base model can be repeated at each checkpoint. The demo uses Pythia-70M, whose public checkpoints run on a CPU. One example measurement is how well each layer separates harmful from harmless instructions at each checkpoint. The Goodfire cyber report found that a model recognizes harmful requests internally even when it complies, so this is a reasonable proxy for when the model learns to recognize harmful content.

## 12. How findings are verified

### 12.1 Engine checks (code)

For every draft finding the engine checks that:

1. every required control ran, and the result beat it by a margin set in the check module
2. results come from held-out data, not the data used to fit a probe or direction
3. every required shift test ran, with its result recorded whether it passed or failed
4. the coverage statement and the "not shown" statement are filled in
5. quantization, remote compute, or a model smaller than the check's recommended minimum are all recorded

The engine then sets the strength:

- **Established:** all controls passed and all shift tests passed.
- **Suggestive:** all controls passed, but some shift tests failed or could not run.
- **Inconclusive:** a control failed, or the effect is within noise.

### 12.2 Reviewer agent

The reviewer reads only the evidence bundle assembled by the engine. For each claim it checks that the wording matches the strength level, and that the claim makes no statement about absence beyond the coverage. For example, "no backdoor signal in layers 1 to 18 with these probes" is acceptable, and "no backdoor" is not. The reviewer can lower a strength level, but cannot raise one.

### 12.3 Guardrails for automated analysis

Adapted from the survey's five-layer framework (section 12.6.2):

| Layer | In the tool |
|---|---|
| Sampled human review | The report highlights a random sample of findings for the user to inspect, mixed in with the others so they don't know which were sampled |
| Consistency checks | Key results are re-run with a different seed. Differences are reported |
| Adversarial tests | Test models with known planted problems are used to measure whether checks catch them (section 19) |
| Error bounds | Each check reports its measured catch rate on test models when one is known |
| Versioning | Reports record the tool version, check version, model revision and data versions. Results are marked out of date when any of these change |

## 13. Compute

- **Device selection:** CUDA (NVIDIA, and AMD through ROCm), then Apple MPS, then CPU, chosen automatically.
- **Free memory:** checked with `torch.cuda.mem_get_info()`. A busy GPU is planned around.
- **Precision:** bf16 or fp16 on GPU, fp32 on CPU.
- **Batch size:** chosen to fit the free memory.
- **Large models:** spread across GPUs with `device_map="auto"`.
- **Quantization:** offered when a model does not fit (`bitsandbytes`, CUDA only). Quantization changes activations slightly, so findings from quantized models are marked as approximate.
- **Activation storage:** caches are streamed to disk when they do not fit in memory.
- **Remote compute:** NDIF (through nnsight's remote mode) or a rented cloud GPU, only with the user's approval for that job. The plan tells the user what will be sent.
- **Estimates:** the agent's plan states the expected run time on the detected hardware.
- **The web app** renders in the browser and does not use the engine's GPU.

Hackathon setup: development happens on a laptop CPU with small models. If cloud GPU credits are available, the same code uses the GPU with no changes.

## 14. Security and safe handling of harmful content

- **Model files:** load `safetensors` weights only. `trust_remote_code` is off by default and needs explicit user approval. Malicious models have been found on the Hugging Face hub (arXiv 2409.09368).
- **Community test models:** treated as untrusted. Each is tested by us before use, and its claimed trigger behavior is confirmed.
- **Agent-written code:** runs in a separate kernel process with a restricted working directory and no network by default. Goodfire warned that running code through a kernel can bypass an agent's normal permission limits. Full container sandboxing is planned after the hackathon.
- **Harmful test content:** harmful prompt datasets stay inside the engine. The agent refers to them by ID and receives aggregate numbers and short, redacted examples. This keeps harmful text out of the agent conversation. It also avoids the Claude model's `cyber` and `bio` safety classifiers declining routine audit work. Server-side refusal fallbacks are turned on for the built-in agent.
- **Data stays local:** nothing leaves the machine without per-job approval. The plan names what would be sent and where.
- **Dual use:** steering and editing tools can also make a model less safe. Exports are labelled, and reports note when a fix was a steering adjustment rather than a training change. Detailed write-ups of a deployed model's weaknesses should be shared with its developer before publication, as the survey recommends.
- **API keys:** read from the environment or the Anthropic CLI login. Never stored in the record or in reports.

## 15. Storage and reports

### 15.1 Workspace layout

```
workspace/
  record.sqlite            # experiments, plans, runs, results, findings
  models/                  # model maps and series descriptions (not weights)
  runs/<run_id>/
    plan.json
    code/                  # code the experiment ran
    results.parquet
    activations/           # cached arrays (safetensors / npy / zarr)
    logs.txt
  reports/<report_id>.html
  exports/                 # probes, steering vectors, monitor configs
```

### 15.2 Report contents

1. The question asked, the model, and the checkpoints or stages covered
2. A summary of findings, each with strength and maturity labels
3. For each finding: a chart, real example inputs and outputs, control results, shift-test results
4. Coverage: what was examined
5. What the results do not show
6. Settings: device, precision, quantization, seeds, versions
7. Links from every number to its run and code
8. Sampled findings for human review

## 16. Repository layout

```
engine/
  loader/  modelmap/  compute/  blocks/  sae/  dynamics/  attribution/
  checks/            # one module per risk
  runner/  verify/  record/  report/  api/  cli/  monitor/
agent/               # built-in agent and reviewer
plugin/              # Claude Code plugin
web/                 # React app
datasets/            # small curated sets and loaders for public ones
organisms/           # scripts that fetch and verify test models
docs/
  design.md          # this file
tests/
```

## 17. Hackathon build plan

### 17.1 Scope

| Status | What |
|---|---|
| **Build fully** | Model loader and model map (E1, E2) · compute manager (E3) · activations, probes, difference-of-means directions, steering (E4) · three risk checks: sycophancy, weakened refusals, backdoor detection · verification (E10) · record (E11) · HTML report (E12) · engine API with MCP (E13) · built-in agent and reviewer (A1, A2) · chat and 3D view with focus and paint (W2, W3) · Pythia checkpoint timeline (W4) |
| **Build a minimal version** | SAE loading for Gemma 3 270M through Gemma Scope 2 (E5) · training callback, shown on a short run if a GPU is available (E15) · Claude Code plugin packaging (A4) · basic CLI (E14) · NDIF remote switch |
| **Describe, with interfaces in place** | Training data attribution (E7) · local learning coefficient (E6) · evaluation awareness and alignment faking checks · hidden bias check (needs a 9B model) · full sandboxing · SAE training for custom models |

### 17.2 Work split for a team of three or four

Work starts in parallel once the data formats in section 10 are agreed. That should take about the first hour.

| Track | Owner | Work |
|---|---|---|
| Engine core | 1 person | E1 to E4, E9 to E11, the three risk checks |
| API, agent, reports | 1 person | E12 to E14, A1 to A4 |
| Web app | 1 person | W1 to W7, starting with mock events |
| Test models and data (shared, or a fourth person) | 1 person | Fetch and verify test models, prepare datasets, build the demo fine-tunes, write the demo script |

For a smaller team, cut in this order: the hidden-bias and drift extras, then the training callback, then the 3D detail levels below "layer". Keep the engine, the three checks, verification and the basic 3D view.

### 17.3 Rough timeline (20 hours)

| Hours | Milestone |
|---|---|
| 0 to 1 | Agree on data formats. Repository and environment set up |
| 1 to 6 | Loader, model map, activations and probes working on a small model. Web app draws a model map from a file. Agent connected to a stub MCP server |
| 6 to 11 | Sycophancy and refusal checks running end to end with controls. Live events reach the 3D view. Reports render |
| 11 to 15 | Backdoor check on the sleeper-agent test model and its clean control. Pythia timeline. Reviewer agent |
| 15 to 18 | Fixes and re-checks. Claude Code plugin. Polish of the chat and the 3D view |
| 18 to 20 | Demo rehearsal, recording a backup video, README |

## 18. Demo plan

About three minutes.

1. **The problem (30 seconds).** Behavioral tests miss hidden problems. Interpretability methods exist, but teams have to hire outside labs or send their weights away to use them.
2. **A fine-tuned model with two planted problems (90 seconds).** In chat: "Did my fine-tune make this model less safe?" The agent proposes a plan, the user edits one line in plain language, and the experiments run. The 3D view lights up layer by layer. Results: sycophancy increased, located in specific layers; refusal strength dropped; the planted backdoor was detected, compared against a clean control. Show the "what this does not show" section.
3. **Pretraining timeline (30 seconds).** Pythia checkpoints. Drag the slider and watch the harmful-content signal appear during training.
4. **The same engine from Claude Code (15 seconds).** Run one command in the terminal, and the result appears in the web app.
5. **Why open source (15 seconds).** Runs on your own hardware, weights never leave, anyone can add a check.

Keep a recorded video of the full demo in case of network or compute problems.

## 19. Testing

- **Unit tests:** device selection, model map building on three different architectures, probe and direction code on synthetic data with a known answer.
- **Check tests on known problems:** each check runs on a model with a planted problem and on a clean model. The test passes when the check catches the problem and reports nothing established on the clean model.
- **Control tests:** confirm that a random direction and shuffled labels produce null results.
- **Shift tests:** confirm that a deliberately fragile steering vector is reported as failing on shifted inputs.
- **End-to-end:** a scripted conversation through the agent produces a report with every required section.
- **Web app:** replay a recorded event stream and check that focus, paint and timeline all respond.

## 20. Open questions

1. Name of the tool.
2. Team size, which decides the work split in 17.2.
3. Whether GPU credits are available from hackathon sponsors. If they are, the training callback and the 4B to 9B test models move into the demo.
4. Whether the reviewer uses the same model as the built-in agent or a different one.
5. Which community test models pass our own verification.
6. License for the tool's own code (for example Apache-2.0 or MIT).
7. How much of the risk-check guidance to ship as Claude Code skills, and how much to keep in the engine only.

## 21. References

### Survey and background

- Barez et al. "Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety." arXiv 2611.08438v2, November 2026.
- Goodfire Silico documentation and example reports: https://docs.goodfire.com/reports/
- Goodfire, "You and Your Research Agent": https://www.goodfire.com/blog/you-and-your-research-agent
- Goodfire Scribe (notebook MCP for agents): https://github.com/goodfire-ai/scribe

### Methods

- Hubinger et al. 2024, "Sleeper Agents". MacDiarmid et al. 2024, "Simple probes can catch sleeper agents".
- Greenblatt et al. 2024, "Alignment faking in large language models".
- Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction": https://github.com/andyrdt/refusal_direction
- Rimsky et al. 2024, "Steering Llama 2 via Contrastive Activation Addition".
- Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models": https://github.com/meg-tong/sycophancy-eval
- Nguyen et al. 2025, "Probing and Steering Evaluation Awareness of Language Models": https://arxiv.org/abs/2507.01786
- "Evaluation Awareness Scales Predictably in Open-Weights Large Language Models": https://arxiv.org/pdf/2509.13333
- "Steering Evaluation-Aware Language Models to Act Like They Are Deployed": https://www.alphaxiv.org/abs/2510.20487
- Cywiński et al. 2025, "Eliciting Secret Knowledge from Language Models": https://arxiv.org/pdf/2510.01070 and https://github.com/cywinski/eliciting-secret-knowledge
- Betley et al. 2025, "Emergent Misalignment": https://arxiv.org/pdf/2502.17424 (Nature version: https://www.nature.com/articles/s41586-025-09937-5)
- "Model Organisms for Emergent Misalignment": https://www.alignmentforum.org/posts/yHmJrDSJpFaNTZ9Tr/model-organisms-for-emergent-misalignment
- Aranguri and McGrath 2025, "Discovering undesired rare behaviors via model diff amplification": https://www.goodfire.ai/research/model-diff-amplification
- "Agentic Backdoors via Pretraining Poisoning": https://arxiv.org/abs/2610.37741
- Anthropic, open replication of the auditing game model organism: https://alignment.anthropic.com/2025/auditing-mo-replication/
- Open-source alignment faking replication: https://github.com/Ibtastic/open-source-alignment-faking
- "Do Influence Functions Work on Large Language Models?": https://arxiv.org/pdf/2409.19998
- "Models Are Codes" (malicious models on model hubs): https://arxiv.org/pdf/2409.09368

### Test models

- https://huggingface.co/senku21x/mot-sleeper-agents-generality
- https://huggingface.co/Antonhansel/qwen2.5-1.5b-sleeper-canary
- https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models
- https://huggingface.co/emergent-misalignment/Qwen-Coder-Insecure
- Pythia checkpoint suite (Biderman et al. 2023)

### Libraries

- nnsight and NDIF: https://arxiv.org/abs/2407.14561, https://ndif.baulab.info/
- TransformerLens 3: https://transformerlensorg.github.io/TransformerLens/content/news/release-3.0.html
- SAELens, Gemma Scope 2: https://huggingface.co/google/gemma-scope-2
- Llama Scope: https://arxiv.org/pdf/2410.20526
- Neuronpedia: https://neuronpedia.org/
- devinterp: https://github.com/timaeus-research/devinterp
- Kronfluence, dattri (training data attribution)
- Claude Agent SDK: https://code.claude.com/docs/en/agent-sdk, https://github.com/anthropics/claude-agent-sdk-python
- Claude Code plugins reference: https://code.claude.com/docs/en/plugins-reference
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- React Three Fiber v9: https://r3f.docs.pmnd.rs/tutorials/v9-migration-guide
- camera-controls: https://github.com/yomotsu/camera-controls
- react-spring for React Three Fiber: https://react-spring.dev/guides/r3f
- Three.js, Babylon.js and PlayCanvas comparison (2026): https://www.utsubo.com/blog/threejs-vs-babylonjs-vs-playcanvas-comparison
- llm-viz (visual reference only, no license): https://github.com/bbycroft/llm-viz
