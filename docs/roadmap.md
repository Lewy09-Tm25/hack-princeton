# Build roadmap: component by component

Status: draft for HackPrinceton (20-hour build)
Last updated: 2026-11-13
Companion to: [design.md](design.md). Component IDs (E1, A1, W1 and so on) match that document.

This document says what to build, in what order, in which files, with which interfaces, and how to tell when each piece is done. It is written so that three or four people can work in parallel from the first hour.

---

## Contents

1. How to read this document
2. Build order and dependencies
3. Phase 0: setup and shared formats (hour 0 to 1)
4. Engine components (E1 to E15)
5. Datasets and test models (D1 to D3)
6. Agent components (A1 to A4)
7. Web app components (W1 to W7)
8. Integration checkpoints
9. Hour-by-hour schedule per track
10. What to cut if time runs short
11. Demo readiness checklist
12. After the hackathon

---

## 1. How to read this document

Every component section has the same parts:

- **Purpose:** one line.
- **Depends on:** components that must exist first. A stub counts.
- **Files:** where the code goes.
- **Interface:** the functions, classes or endpoints other components rely on.
- **Build steps:** in order.
- **Done when:** a check anyone on the team can run.
- **Tests:** what to test automatically.
- **Stub version:** the smallest fake that lets other tracks keep working.
- **Time:** a rough estimate for one person.
- **Notes:** risks and things to watch.

Tracks (owners):

| Track | Covers |
|---|---|
| T1: Engine core | E1 to E4, E8 to E11, the three risk checks |
| T2: API, agent, reports | E12 to E14, A1 to A4 |
| T3: Web app | W1 to W7 |
| T4: Data, test models, demo | D1 to D3, E5, E6, E15, demo script |

With three people, T4 is shared: T1 takes D1, T2 takes D2, and T3 takes the demo script.

---

## 2. Build order and dependencies

```
Phase 0  contracts (C0) ---------------------------------------------+
            |                                                        |
Phase 1  E3 compute -> E1 loader -> E2 model map      E11 record     |  W1 shell, W3 view (from fixture files)
            |                         |                    |         |  A1 agent (against stub MCP)
Phase 2  E4 building blocks ----------+--------------------+         |  D1 datasets, D2 test models
            |                                                        |
            +--> E10 verification                                    |
Phase 3  E8 check framework -> sycophancy, refusal, backdoor checks  |
            |                                                        |
            +--> E9 runner                                           |
Phase 4  E13 API (MCP + HTTP + WebSocket) <--------------------------+
            |-> E12 reports   E14 CLI   A2 reviewer   A3 approval    W2 chat, W4 timeline, W5 charts
Phase 5  Pythia timeline (E6 lite)   E5 SAEs   E15 monitor   A4 plugin   W6, W7
Phase 6  integration, demo rehearsal, backup video
```

Rules for parallel work:

- Everything is built against the contracts from Phase 0, never against another person's unfinished code.
- Each track keeps a stub of what it provides, so other tracks are never blocked.
- The web app is built against recorded fixture files and a fake event player until the real API exists.

---

## 3. Phase 0: setup and shared formats (hour 0 to 1)

Everyone does this together.

### 3.1 Repository and environment

- Create the repository with the layout from design.md section 16.
- **Python:**
  - Python 3.11 or 3.12, managed with `uv`.
  - One `pyproject.toml` at the root with optional extras: `engine`, `agent`, `sae`, `dev`.
  - Install the CPU build of PyTorch on laptops. GPU machines install the CUDA build.
- **Node:**
  - Node 20 or later, with `pnpm`.
  - Create `web/` with `pnpm create vite web --template react-ts`.
- **Pin versions after checking:** `torch`, `transformers`, `nnsight`, `sae-lens`, `fastapi`, `mcp`, `claude-agent-sdk`, `three`, `@react-three/fiber` (v9 line), `@react-three/drei`, `@react-spring/three`, `plotly.js`. Record the pinned versions in `docs/versions.md`.
- **Tooling:** `ruff` and `pytest` for Python, `eslint` and `vitest` for the web app. Pre-commit hooks are optional. Do not spend time on CI during the hackathon.
- **Windows notes:**
  - The main development machine runs Windows.
  - Enable long paths in git (`git config core.longpaths true`).
  - Use forward slashes in code paths via `pathlib`.
  - `bitsandbytes` quantization is CUDA-focused, so treat it as optional on Windows.
  - Jupyter kernels work normally.

### 3.2 Shared formats (contract C0)

**Purpose:** one definition of the data formats from design.md section 10, used by every component.

**Files:**
- `engine/contracts.py`: Pydantic models.
- `web/src/contracts.ts`: TypeScript types.
- `docs/contracts.md`: a short field-by-field description.

**Interface (Python):**

```python
class Component(BaseModel):
    id: str                    # "L14", "L14.attn", "L14.attn.h3", "embed", "unembed"
    kind: Literal["embedding", "layer", "attention", "attention_head", "mlp", "norm", "unembedding", "other"]
    index: int | None = None   # layer index
    head: int | None = None
    module: str | None = None  # dotted module path used for hooks
    children: list[str] = []

class ModelMap(BaseModel):
    model_id: str
    revision: str | None
    arch_family: str | None
    n_layers: int
    hidden_size: int | None
    confirmed_by_user: bool
    needs_confirmation: list[str] = []   # module paths the builder could not classify
    components: list[Component]

class SeriesItem(BaseModel):
    checkpoint_id: str; label: str; order: int; source: str

class Series(BaseModel):
    series_id: str; kind: Literal["pretraining", "post_training"]; model_family: str; items: list[SeriesItem]

class ControlResult(BaseModel):
    kind: str; value: float | None; passed: bool | None

class ResultRecord(BaseModel):
    run_id: str; check: str; component_id: str; checkpoint_id: str | None
    metric: str; value: float; control: ControlResult | None
    dataset_id: str; split: Literal["train", "held_out"]; seed: int
    device: str; precision: str; quantized: bool

class Finding(BaseModel):
    finding_id: str; run_id: str; check: str; claim: str
    strength: Literal["established", "suggestive", "inconclusive"]
    maturity: Literal["established", "experimental"]
    evidence: list[str]; controls: list[ControlResult]; shift_tests: list[ControlResult]
    coverage: str; not_shown: str
    reviewer_verdict: dict | None = None

class Event(BaseModel):
    type: Literal["focus", "paint", "timeline", "progress", "question", "finding", "job_status"]
    payload: dict
```

**Build steps:**
1. Write the Pydantic models.
2. Write the matching TypeScript types by hand. Generating them is not worth setting up for this many types.
3. Write fixture files in `tests/fixtures/`:
   - `modelmap_pythia70m.json`, `modelmap_gemma3_270m.json`, `modelmap_qwen25_05b.json`. Write them by hand from each model's config at first; E2 replaces them with real output later.
   - `events_sycophancy_run.jsonl`: a believable sequence of about 60 events for a sycophancy run (progress, paint per layer, focus, finding).
   - `series_pythia70m.json`: about six checkpoints.

**Done when:** both languages import the types, and the fixtures validate against the Pydantic models (`pytest tests/test_contracts.py`).

**Time:** 1 hour, all together.

---

## 4. Engine components

### E3. Compute manager

**Purpose:** pick the best available device and settings for each job.

**Depends on:** nothing.

**Files:** `engine/compute/device.py`, `engine/compute/budget.py`

**Interface:**

```python
@dataclass
class ComputeProfile:
    device: str                 # "cuda", "cuda:1", "mps", "cpu"
    dtype: torch.dtype          # bf16/fp16 on GPU, fp32 on CPU
    free_memory_bytes: int | None
    total_memory_bytes: int | None
    n_gpus: int
    remote_available: bool      # NDIF key present

def detect() -> ComputeProfile
def plan_for_model(n_params: int, profile: ComputeProfile) -> LoadPlan
    # LoadPlan: device_map, dtype, quantize (None | "8bit" | "4bit"), fits (bool), warnings (list[str])
def batch_size_for(seq_len: int, hidden: int, n_layers_cached: int, profile: ComputeProfile) -> int
def estimate_seconds(n_forward_passes: int, n_params: int, profile: ComputeProfile) -> float
```

**Build steps:**
1. `detect()`: check `torch.cuda.is_available()`, then `torch.backends.mps.is_available()`, else CPU. For CUDA, read `torch.cuda.mem_get_info()` per device.
2. `plan_for_model()`:
   - Estimate weight memory from the parameter count and dtype, plus about 20% headroom.
   - If the model fits, use one device.
   - If it doesn't fit and there are several GPUs, use `device_map="auto"`.
   - If it still doesn't fit, offer 8-bit and then 4-bit quantization, and add the warning "quantized: findings are approximate".
   - If nothing fits, set `fits=False` and suggest remote compute.
3. `batch_size_for()`: divide free memory, minus the weights, by a per-example activation estimate. Clamp between 1 and 64.
4. `estimate_seconds()`: run a one-time micro-benchmark (a few forward passes) on first use, cached per model, then scale.

**Done when:** `python -m engine.compute.device` prints the detected profile on a laptop (CPU) and on a GPU machine if one is available.

**Tests:** mock `torch.cuda` to simulate CPU-only, one GPU with low free memory, and two GPUs. Check that the plan decisions match.

**Stub version:** `detect()` returns a CPU profile.

**Time:** 1.5 hours (T1).

**Notes:** a GPU that is present but busy must be treated as having only its free memory. Don't over-engineer the estimates. They only feed the agent's "about N minutes" statement.

### E1. Model loader

**Purpose:** load models, checkpoint series and post-training stages safely.

**Depends on:** E3, C0.

**Files:** `engine/loader/load.py`, `engine/loader/series.py`, `engine/loader/registry.py`

**Interface:**

```python
@dataclass
class LoadedModel:
    model_id: str               # internal ID, e.g. "hf:Qwen/Qwen2.5-0.5B-Instruct@main"
    source: str                 # HF repo ID or local path
    revision: str | None
    nn: "nnsight.LanguageModel" # wrapped model
    tokenizer: Any
    config: Any
    load_plan: LoadPlan
    n_params: int

def load_model(source: str, revision: str | None = None, allow_remote_code: bool = False,
               adapter: str | None = None) -> LoadedModel
def unload_model(model_id: str) -> None
def register_series(kind: str, items: list[SeriesItem], model_family: str) -> Series
def load_checkpoint(series: Series, checkpoint_id: str) -> LoadedModel   # loads one at a time
```

**Build steps:**
1. Resolve the source:
   - A local folder is used as-is.
   - Anything else is a Hugging Face repo ID. Download with `huggingface_hub.snapshot_download` using `allow_patterns` for `*.safetensors`, `*.json` and tokenizer files only.
2. **Refuse `.bin` and pickle weights** unless the user explicitly allows them. Refuse `trust_remote_code` unless `allow_remote_code=True`.
3. Load with `transformers` using the `LoadPlan` from E3, then wrap with `nnsight.LanguageModel`.
4. If `adapter` is given (LoRA), load it with `peft` and merge for analysis. Keep the adapter ID in `model_id`.
5. Series:
   - Pretraining series use `revision` branches (Pythia uses `step1000` and so on).
   - Post-training series use a list of paths or repo IDs, one per stage.
   - Only one checkpoint is kept in memory at a time. Unload before loading the next.
6. Keep a registry of loaded models in memory, with a least-recently-used limit of two models: one plus a reference.

**Done when:** these three loads work on a laptop:
- `load_model("EleutherAI/pythia-70m", revision="step1000")`
- `load_model("Qwen/Qwen2.5-0.5B-Instruct")`
- a local folder

Each one can generate ten tokens.

**Tests:**
- Safetensors-only enforcement, using a fixture repo folder with a `.bin` file.
- `trust_remote_code` refusal.
- Series ordering.

**Stub version:** loads GPT-2 small only.

**Time:** 2 hours (T1).

**Notes:** Pythia has 154 checkpoints. Never load more than the series asks for. Download time is the main delay, so pre-download the demo models (D2).

### E2. Model map builder

**Purpose:** describe any architecture as a list of components with hook paths.

**Depends on:** E1, C0.

**Files:** `engine/modelmap/build.py`, `engine/modelmap/patterns.py`

**Interface:**

```python
def build_model_map(model: LoadedModel) -> ModelMap
def apply_confirmation(model_map: ModelMap, mapping: dict[str, str]) -> ModelMap
    # mapping: module path -> component kind, from the user's answers
def hook_path(model_map: ModelMap, component_id: str) -> str
```

**Build steps:**
1. Read counts from the config: `num_hidden_layers` / `n_layer`, `num_attention_heads` / `n_head`, `hidden_size` / `d_model`, `vocab_size`.
2. Find the layer stack. This is the `nn.ModuleList` whose length equals the layer count and whose entries are all the same class. If several match, take the deepest.
3. Inside each layer, classify submodules by name patterns (`patterns.py`):
   - attention: `attn`, `attention`, `self_attn`, `mixer`, `mix`
   - MLP: `mlp`, `ffn`, `feed_forward`, `fc`
   - norms: `norm`, `ln`, `layernorm`
4. Embedding: the `nn.Embedding` with `num_embeddings == vocab_size`. Unembedding: the `nn.Linear` with `out_features == vocab_size`.
5. Anything in a layer that cannot be classified goes into `needs_confirmation`, and `confirmed_by_user` stays false.
6. Run one forward pass on four tokens with forward hooks, recording output shapes to confirm the hidden size and catch mistakes.
7. Create attention-head child components from the head count. Head-level hooks are resolved later by E4, by splitting the attention output.
8. Cache the result in the record (E11), keyed by `model_id`.
9. If TransformerLens 3 supports the architecture, optionally record its hook names as aliases. This is optional for the hackathon.

**Done when:** the builder produces correct maps for Pythia-70M (GPT-NeoX), Qwen2.5-0.5B and Gemma 3 270M, matching the hand-written fixtures. It also produces a map with `needs_confirmation` entries for a small custom toy model in `tests/fixtures/toy_custom_model.py` with oddly named modules.

**Tests:** the three real architectures (mark as slow and skip in quick runs), plus the custom toy model.

**Stub version:** return the fixture JSON for known model IDs.

**Time:** 2.5 hours (T1).

**Notes:**
- Mixture-of-experts layers should be labelled `other` with a warning. Patching them can change expert routing (survey section 14.7).
- Don't try to support Mamba-style models in the hackathon. Detect them and say so.

### E11. Record

**Purpose:** store everything about every experiment.

**Depends on:** C0.

**Files:** `engine/record/db.py`, `engine/record/schema.sql`, `engine/record/workspace.py`

**Interface:**

```python
class Record:
    def __init__(self, workspace: Path): ...
    def save_model_map(self, m: ModelMap) -> None
    def get_model_map(self, model_id: str) -> ModelMap | None
    def save_series(self, s: Series) -> None
    def create_experiment(self, question: str, model_id: str, series_id: str | None) -> str
    def save_plan(self, experiment_id: str, plan: "Plan") -> str           # returns plan_id (versioned)
    def create_run(self, plan_id: str) -> str
    def set_run_status(self, run_id: str, status: str, message: str = "") -> None
    def add_results(self, rows: list[ResultRecord]) -> None
    def query_results(self, run_id: str | None = None, check: str | None = None,
                      metric: str | None = None, checkpoint_id: str | None = None) -> list[ResultRecord]
    def save_finding(self, f: Finding) -> None
    def get_findings(self, run_id: str) -> list[Finding]
    def run_dir(self, run_id: str) -> Path       # workspace/runs/<run_id>/
```

**Build steps:**
1. Create SQLite tables for models, series, experiments, plans, runs, results, findings and reports.
2. Store results as rows (they are small), and also write `results.parquet` per run for quick bulk reads.
3. Arrays such as activations, directions and probe weights go in `run_dir`, using `safetensors` for tensors.
4. Every run records the tool version, check version, model revision, dataset versions, seed, device, precision and whether quantization was used.
5. Use one SQLite connection per thread, in WAL mode.

**Done when:** a script can create an experiment, plan and run, add 100 result rows and query them back by checkpoint and metric.

**Tests:** a round trip of every contract type, and concurrent writes from two threads.

**Stub version:** an in-memory dictionary with the same methods.

**Time:** 1.5 hours (T1, or T2 if T1 is behind).

### E4. Interpretability building blocks

**Purpose:** the measurement tools that every risk check uses.

**Depends on:** E1, E2, E3.

**Files:** `engine/blocks/activations.py`, `probes.py`, `directions.py`, `steering.py`, `patching.py`, `generate.py`

**Interface:**

```python
# activations.py
def collect_activations(model: LoadedModel, mmap: ModelMap, prompts: list[str],
                        component_ids: list[str], position: Literal["last", "mean"] = "last",
                        batch_size: int | None = None, cache_dir: Path | None = None) -> dict[str, torch.Tensor]
    # returns component_id -> [n_prompts, hidden] (CPU float32)

# probes.py
@dataclass
class ProbeResult:
    component_id: str; accuracy: float; auroc: float; weights: np.ndarray; bias: float
def train_probe(X_train, y_train, X_test, y_test, C: float = 1.0) -> ProbeResult
def probe_all_layers(acts_train: dict, y_train, acts_test: dict, y_test) -> list[ProbeResult]

# directions.py
def diff_of_means(acts_pos: torch.Tensor, acts_neg: torch.Tensor, normalize: bool = True) -> torch.Tensor
def projection(acts: torch.Tensor, direction: torch.Tensor) -> torch.Tensor
def random_direction_like(direction: torch.Tensor, seed: int) -> torch.Tensor
def cosine(a: torch.Tensor, b: torch.Tensor) -> float

# steering.py
def generate_with_steering(model: LoadedModel, mmap: ModelMap, prompts: list[str],
                           component_id: str, direction: torch.Tensor, strength: float,
                           strength_mode: Literal["absolute", "relative_to_norm"] = "relative_to_norm",
                           max_new_tokens: int = 128) -> list[str]
def residual_norm(model: LoadedModel, mmap: ModelMap, prompts: list[str], component_id: str) -> float

# patching.py
def patch_component(model: LoadedModel, mmap: ModelMap, clean_prompts: list[str],
                    source_prompts: list[str], component_id: str, metric_fn) -> float

# generate.py
def generate(model: LoadedModel, prompts: list[str], max_new_tokens: int = 128,
             chat_template: bool = True) -> list[str]
```

**Build steps:**
1. `generate`: batched generation using the chat template when the tokenizer has one. Greedy decoding by default, seeded for reproducibility.
2. `collect_activations`:
   - Use an nnsight trace to save the outputs of the requested layer modules.
   - Take the last-token or mean vector.
   - Move results to CPU float32 straight away to save GPU memory.
   - Batch using E3, and stream to `cache_dir` when it gets large.
3. `train_probe`: scikit-learn logistic regression with standardization. Report accuracy and AUROC on the held-out split only.
4. `diff_of_means` and related functions: plain torch.
5. Steering:
   - Add `strength * direction` to the residual output at one layer, at every position, during generation.
   - Default the strength to a multiple of that layer's mean residual norm. Goodfire's Kimi report found that absolute coefficients don't transfer between layers, and the working range must be searched per layer.
   - Provide `sweep_strengths()`, which tries a small grid (for example 0.25, 0.5, 1.0, 1.5 times the norm) and returns outputs for each.
6. `patch_component`: run the source prompts, save the component output, then run the clean prompts with that output swapped in, and return the metric.
7. Heads: split the attention output by head (reshape to `[.., n_heads, head_dim]` before the output projection where possible). If the module structure doesn't allow this, report head-level support as unavailable instead of guessing.

**Done when:** on Qwen2.5-0.5B-Instruct on a laptop CPU:
- activations for 64 prompts across all layers are collected in under a few minutes
- a probe separating harmful from harmless instructions reaches high held-out accuracy at some layer
- steering along that direction visibly changes outputs

**Tests:**
- Synthetic data where the answer is known: a planted direction must be found by `diff_of_means`, and a probe on random labels must sit near chance.
- Steering with a zero vector must leave outputs unchanged.

**Stub version:** return random arrays of the right shapes.

**Time:** 4 hours (T1). This is the most important engine piece.

**Notes:** keep everything working on CPU. Use `torch.inference_mode()`. Clear caches between checkpoints.

### E10. Verification

**Purpose:** run controls and shifted-input tests, and decide how strong each finding is.

**Depends on:** E4, C0.

**Files:** `engine/verify/controls.py`, `engine/verify/shift.py`, `engine/verify/strength.py`, `engine/verify/evidence.py`

**Interface:**

```python
def control_random_direction(effect_fn, direction, n_draws: int = 5, seed: int = 0) -> ControlResult
def control_shuffled_labels(train_fn, X, y, n_draws: int = 5, seed: int = 0) -> ControlResult
def control_reference_model(run_fn, reference: LoadedModel) -> ControlResult
def shift_test(effect_fn, variant_dataset_id: str, kind: Literal["paraphrase", "multilingual", "multi_turn"]) -> ControlResult
def decide_strength(controls: list[ControlResult], shift_tests: list[ControlResult],
                    required_controls: list[str], required_shifts: list[str]) -> str
def build_evidence_bundle(finding: Finding, record: Record) -> dict   # what the reviewer reads
```

**Build steps:**
1. Random-direction control: apply the same intervention with random directions of equal norm, several draws. It passes if the real effect beats every draw by the margin set in the check.
2. Shuffled-label control: retrain with shuffled labels several times. It passes if real accuracy is above the 95th percentile of the shuffled runs.
3. Reference-model control: run the same procedure on the clean or base model.
4. Shift tests: rerun the effect on variant datasets (paraphrased, Spanish, multi-turn wrappers from D1). Record pass or fail, never drop a result.
5. `decide_strength`:
   - A missing required control means the finding cannot be marked verified. The engine refuses.
   - Any failed control gives "inconclusive".
   - All controls passed but some shift tests failed gives "suggestive".
   - Everything passed gives "established".
6. `build_evidence_bundle`: the claim, numbers, control results, shift results, coverage, not-shown text, settings (quantized or not), and a few redacted examples.

**Done when:** on the synthetic fixtures, a real planted effect comes out established, and a fake effect comes out inconclusive.

**Tests:** each branch of `decide_strength`, including refusal when a required control is missing.

**Stub version:** everything passes, and strength is always "suggestive".

**Time:** 2 hours (T1).

### E8. Risk check framework and the three hackathon checks

**Purpose:** one module per risk, holding the method, data, controls and coverage rules.

**Depends on:** E4, E10, D1.

**Files:**
- `engine/checks/base.py` (the `RiskCheck` interface from design.md 10.6, plus a `Plan` model)
- `engine/checks/registry.py` (entry-point discovery)
- `engine/checks/sycophancy.py`, `engine/checks/refusal.py`, `engine/checks/backdoor.py`
- `engine/checks/described/`: stubs for hidden bias, drift, data poisoning, evaluation awareness and alignment faking, each with `describe()` filled in and `run()` raising "not available in this version"

**Plan model:**

```python
class Plan(BaseModel):
    plan_id: str | None = None
    check: str
    model_id: str
    reference_model_id: str | None       # base or clean model
    series_id: str | None
    checkpoints: list[str] | None
    datasets: list[str]                  # dataset IDs
    layers: list[int] | Literal["all"]
    controls: list[str]                  # always includes the required ones
    shift_tests: list[str]
    user_context: dict                   # domain, notes
    estimate_seconds: float
    coverage: str
    not_covered: str
    maturity: str
    papers: list[str]
    approved: bool = False
```

**Build steps:**
1. `base.py`:
   - The interface.
   - `validate_plan()`, which refuses any plan that removes a required control or shift test. The user can add controls but not remove required ones; if they insist, the plan runs but the finding is capped at "suggestive".
2. `registry.py`: discover checks through `importlib.metadata.entry_points(group="tool.checks")`, so outside packages can add checks.
3. **Sycophancy check:**
   - Plan: pushback conversations from D1 (`are_you_sure` subset), plus domain variants if the user gives a domain.
   - Behavioral measurement: the rate of abandoning a correct answer after pushback, for the model and the reference.
   - Internal measurement:
     - Contrast pairs (sycophantic and honest completions) go through `collect_activations`.
     - Per-layer probes measure where the behavior is represented.
     - A difference-of-means direction is taken at the best layer.
   - Optional fix: steering against the direction, re-tested on held-out prompts and shift sets.
   - Required controls: random direction, shuffled labels, held-out split.
   - Required shift tests: paraphrase, multilingual, multi-turn.
   - Emits `paint` (probe accuracy per layer) and `focus` (best layers).
4. **Refusal check:**
   - Harmful and harmless instruction sets from D1.
   - Refusal direction by difference of means per layer, in both the model and the reference.
   - Projection strength of harmful prompts on the direction, compared between model and reference.
   - Cosine between the two models' directions.
   - Behavioral refusal rate (refusal-phrase matching plus a small judge set).
   - Required controls: random direction, surface-matched harmless prompts.
   - Required shift tests: paraphrase, multilingual.
5. **Backdoor check:**
   - Generic contrast prompts for a probe in the style of MacDiarmid et al. 2024.
   - Run on prompts with and without candidate triggers.
   - Compare the probe signal between the suspect model and the clean control adapter.
   - Report detection only, never removal.
   - Required controls: clean reference model, shuffled labels.
   - Required shift tests: paraphrase and different input formats.
   - Coverage text always includes "trigger search limited to the listed candidates".
6. Every check fills in `coverage()` and the "not shown" text from a template plus its actual settings.

**Done when:** each check produces a plan, runs on its test model from D2 on a laptop in under about 15 minutes, writes result rows, and produces findings with a strength set by E10.

**Tests:**
- Each check on a tiny model with `layers=[0, 1]` and five prompts (smoke test).
- `validate_plan` refusing a plan with a required control removed.

**Stub version:** a check that writes plausible fake results from the event fixture.

**Time:** framework 1 hour, sycophancy 2 hours, refusal 1.5 hours, backdoor 2 hours (T1). If T1 falls behind, T4 takes the backdoor check.

**Notes:** the sycophancy and refusal checks share most of their code. Build sycophancy first, then copy its shape. The behavioral judge must stay inside the engine and must not send harmful text to the agent.

### E9. Experiment runner

**Purpose:** run approved plans as background jobs that survive clients disconnecting.

**Depends on:** E8, E11.

**Files:** `engine/runner/jobs.py`, `engine/runner/events.py`, `engine/runner/kernel.py`

**Interface:**

```python
class EventBus:
    def publish(self, job_id: str, event: Event) -> None
    def subscribe(self) -> AsyncIterator[Event]          # all events, for WebSocket clients

class JobManager:
    def submit(self, plan: Plan) -> str                   # requires plan.approved; returns job_id/run_id
    def status(self, job_id: str) -> dict
    def ask(self, job_id: str, question: str) -> None     # job pauses, emits "question"
    def answer(self, job_id: str, answer: str) -> None    # resumes
    def stop(self, job_id: str) -> None
    def resume_incomplete(self) -> None                   # on server start

class Kernel:                                             # for agent-written custom analysis
    def start(self, run_dir: Path) -> None
    def execute(self, code: str, timeout_s: int = 300) -> dict   # stdout, stderr, display data
    def shutdown(self) -> None
```

**Build steps:**
1. Run jobs on one worker thread with a queue, so only one GPU-heavy job runs at a time. The event bus uses an `asyncio` queue fan-out.
2. Checks receive a context object with `ctx.emit(event)`, `ctx.ask(question)`, `ctx.record`, `ctx.cancelled()`.
3. Status values: queued, running, needs decision, completed, stopped, failed.
4. Partial results are written as they are produced. On restart, unfinished jobs are marked "stopped (server restarted)" and can be re-run. Full resume is not needed for the hackathon.
5. `Kernel` uses `jupyter_client` to start a Python kernel with `cwd=run_dir`. Environment variables strip network proxies and API keys. Use a timeout per cell. This is a stretch goal in the hackathon; the three checks don't need it.

**Done when:** a sycophancy job submitted through Python keeps running after the submitting script exits, and its events can be read by a second process through the API (after E13).

**Tests:** submit, stop, and an ask/answer round trip with a fake check.

**Stub version:** run checks synchronously in the calling thread.

**Time:** 2 hours (T1 or T2).

### E12. Report generator

**Purpose:** HTML reports with findings, evidence, coverage and provenance.

**Depends on:** E11, C0.

**Files:** `engine/report/build.py`, `engine/report/templates/report.html.j2`, `engine/report/charts.py`

**Interface:**

```python
def build_report(run_id: str, record: Record) -> Path   # workspace/reports/<id>.html
```

**Build steps:**
1. A Jinja2 template with the sections from design.md 15.2.
2. Plotly charts (`charts.py`):
   - per-layer metric with its control band
   - model compared with reference
   - checkpoint timeline
   - shift-test pass/fail table
3. Each number links to its run (`runs/<run_id>/` and the result row ID).
4. A "sampled for human review" section: a random sample of findings mixed in with the others, without marking which were sampled.
5. A settings block: device, precision, quantization, seeds and versions.
6. Plain, readable styling, with light and dark color schemes. Self-contained HTML with Plotly loaded from a pinned CDN version, or inlined if offline use matters.

**Done when:** a report built from the fixture run opens in a browser and shows every section.

**Tests:** build a report from fixtures, then check that the required section headings and the coverage text are present.

**Time:** 2 hours (T2).

### E13. Engine API (MCP, HTTP, WebSocket)

**Purpose:** one process serving agents (MCP) and the web app (HTTP and WebSocket).

**Depends on:** E1, E2, E8, E9, E11, E12. Stubs are fine to start.

**Files:** `engine/api/server.py`, `engine/api/mcp_tools.py`, `engine/api/http.py`, `engine/api/ws.py`

**MCP tools** (names as agents see them):

| Tool | Input | Output | Notes |
|---|---|---|---|
| `get_compute` | none | compute profile | |
| `load_model` | source, revision?, adapter? | model_id, summary | refuses unsafe files |
| `get_model_map` | model_id | model map, needs_confirmation | |
| `confirm_components` | model_id, mapping | updated model map | |
| `register_series` | kind, items | series_id | |
| `list_checks` | none | names, maturity, one-line description | |
| `describe_check` | name | methods, papers, requirements, required controls, limits | |
| `draft_plan` | check, model_id, reference?, series?, user_context | plan with estimate and coverage | engine fills in the required controls |
| `revise_plan` | plan_id, changes | new plan version, warnings | cannot remove required controls |
| `run_plan` | plan_id | job_id | requires approval (see A3) |
| `job_status` | job_id | status, last messages | |
| `answer_job` | job_id, answer | ok | |
| `stop_job` | job_id | ok | |
| `get_findings` | run_id | findings | only verified findings are marked verified |
| `get_results` | run_id, filters | result rows | aggregates only |
| `get_examples` | run_id, n | redacted examples | harmful text is redacted |
| `view_focus` | component_ids, reason | ok | emits `focus` |
| `view_paint` | run_id, metric, checkpoint_id? | ok | emits `paint` |
| `view_timeline` | checkpoint_id | ok | emits `timeline` |
| `build_report` | run_id | report path or URL | |
| `export_artifact` | run_id, kind (probe, steering, monitor_config) | file path | |
| `get_evidence` | finding_id | evidence bundle | for the reviewer |
| `submit_review` | finding_id, verdict, notes, lower_strength_to? | ok | can only lower strength |

**HTTP endpoints for the web app:**
- `GET /api/models`
- `GET /api/models/{id}/map`
- `GET /api/series/{id}`
- `GET /api/results?run_id=&metric=&checkpoint_id=`
- `GET /api/findings?run_id=`
- `GET /api/jobs/{id}`
- `GET /api/reports/{id}` (serves the HTML)
- `GET /api/setup/claude-code` (returns the setup command)

**WebSocket:**
- `/ws/events`: every live event.
- `/ws/chat`: the chat session with the built-in agent (A1).

**Build steps:**
1. Create a FastAPI app. Mount the MCP server from the official MCP Python SDK (FastMCP) at `/mcp` using streamable HTTP.
2. Also support stdio transport (`tool mcp-stdio`), for Claude Code setups that launch the server themselves.
3. Every MCP tool is a thin wrapper that validates input with the contract models and calls the engine.
4. `/ws/events` subscribes to the `EventBus`.
5. Bind to `127.0.0.1` only by default. Add a local token for the web app (printed by the CLI at start-up).

**Done when:**
- The MCP inspector, or Claude Code, connects to `http://127.0.0.1:<port>/mcp`, lists the tools, and `list_checks` returns the three checks plus the described ones.
- A browser receives events on `/ws/events` while a job runs.

**Tests:** call each tool with the stub engine. Check that `revise_plan` refuses to drop a required control, and that `submit_review` refuses to raise a strength.

**Stub version:** every tool returns fixture data. Build this first, in hour 1 to 2, so A1 and W2 can start.

**Time:** 3 hours (T2).

### E14. Command-line interface

**Purpose:** start and manage the engine from a terminal.

**Depends on:** E13.

**Files:** `engine/cli/main.py`

**Commands:**
- `tool serve [--port] [--workspace]`: start the API and web app server, and print the URL and token.
- `tool mcp-stdio [--workspace]`: run as a stdio MCP server, for Claude Code.
- `tool jobs`: list jobs and their status.
- `tool report <run_id> [--open]`
- `tool init [--workspace]`: create the workspace folder.
- `tool doctor`: print the compute profile, versions, and which demo models are already downloaded.

**Done when:** `tool doctor` and `tool serve` work on a laptop.

**Time:** 1 hour (T2).

### E5. SAE support (minimal)

**Purpose:** load existing SAEs so features can be shown for supported models.

**Depends on:** E1, E2, E4.

**Files:** `engine/sae/load.py`, `engine/sae/features.py`

**Interface:**

```python
def available_saes(model_id: str) -> list[dict]          # from SAELens's pretrained list
def load_sae(release: str, sae_id: str) -> Any
def top_features(model: LoadedModel, sae, prompts: list[str], component_id: str, k: int = 20) -> list[dict]
def feature_description(release: str, feature_idx: int) -> str | None   # Neuronpedia, if reachable
```

**Build steps:**
1. Use SAELens to load a Gemma Scope 2 SAE for Gemma 3 270M at one or two layers.
2. `top_features`: encode activations and return the top features by mean activation, with descriptions from Neuronpedia when the network is available.
3. Expose this as an optional step in the sycophancy check: "features most active on sycophantic answers".

**Done when:** the top 10 features for 20 prompts are listed with descriptions for Gemma 3 270M.

**Time:** 2 hours (T4). Optional.

**Notes:** SAE features are labelled "suggestive" unless tested causally. The survey warns that a plausible label does not mean the model uses the feature.

### E6. Training-dynamics tools (timeline only for the hackathon)

**Purpose:** repeat a measurement across checkpoints and show how it changes.

**Depends on:** E1 (series), E4, E8, E11.

**Files:** `engine/dynamics/timeline.py`

**Interface:**

```python
def run_across_checkpoints(series: Series, checkpoint_ids: list[str], measure_fn,
                           ctx) -> list[ResultRecord]   # loads one checkpoint at a time
```

**Build steps:**
1. For each checkpoint: load it, run the measurement, write results with that `checkpoint_id`, emit `timeline` and `paint`, then unload.
2. The demo measurement is per-layer probe accuracy for harmful versus harmless instructions on Pythia-70M, at about six checkpoints (for example steps 1000, 8000, 16000, 32000, 64000 and 143000).
3. Keep it as a mode of any check that can run on a base model. Add it to the sycophancy and refusal check plans as an option when a pretraining series is present. Since Pythia is a base model, use the harmful-recognition probe, not refusal behavior.
4. Leave an interface placeholder for the local learning coefficient through `devinterp` (described, not built).

**Done when:** the Pythia timeline produces results at six checkpoints in reasonable time on a CPU, and the web timeline can step through them.

**Time:** 2 hours (T4).

### E7. Training data attribution (described only)

**Purpose:** link a behavior to training documents.

**Files:** `engine/attribution/README.md`, `engine/attribution/interface.py`

**Build steps:** write the interface (`attribute(behavior_examples, corpus_ref) -> list[DocumentScore]`) and the documentation: Kronfluence or dattri for influence functions, exact span search, and the known reliability limits. No implementation during the hackathon.

**Time:** 30 minutes (T4).

### E15. Training monitor (minimal)

**Purpose:** run checks during a user's training run.

**Depends on:** E8, E11. It talks to the engine through HTTP, or writes to the workspace directly.

**Files:** `engine/monitor/callback.py`, `engine/monitor/torch_hook.py`, `examples/train_with_monitor.py`

**Interface:**

```python
class SafetyMonitorCallback(transformers.TrainerCallback):
    def __init__(self, checks: list[str], every_n_steps: int, engine_url: str | None = None,
                 workspace: Path | None = None, thresholds: dict[str, float] | None = None): ...
    def on_step_end(self, args, state, control, model=None, **kw): ...
```

**Build steps:**
1. Every N steps, run a cheap version of each chosen check: the refusal projection strength and the sycophancy probe accuracy at two layers, on 32 prompts.
2. Write results with `checkpoint_id = f"step{state.global_step}"`.
3. If a value moves past its threshold compared with the first measurement, log a warning and emit an alert event.
4. Example script: a short LoRA fine-tune of Qwen2.5-0.5B-Instruct on a small harmless dataset with the callback attached.

**Done when:** the example runs for about 100 steps and the timeline shows the monitored values. On CPU this may be slow, so run it on a cloud GPU if one is available.

**Time:** 2 hours (T4). Optional.

---

## 5. Datasets and test models

### D1. Datasets

**Purpose:** small, versioned datasets for each check, including the shift-test variants.

**Files:** `datasets/<dataset_id>/data.jsonl`, `datasets/<dataset_id>/card.md`, `engine/data/registry.py`

**Interface:**

```python
def get_dataset(dataset_id: str) -> list[dict]
def list_datasets(check: str | None = None) -> list[dict]   # id, size, source, license, contains_harmful
```

**Datasets to prepare:**

| Dataset ID | Contents | Source | Size |
|---|---|---|---|
| `syco/are_you_sure` | Questions with correct answers, plus pushback turns | `meg-tong/sycophancy-eval` | 200 |
| `syco/pairs` | Sycophantic and honest completion pairs | Built from the above | 200 pairs |
| `syco/paraphrase`, `syco/es`, `syco/multiturn` | Shift-test variants | Written for this tool | 50 each |
| `refusal/harmful` | Harmful instructions | AdvBench, MaliciousInstruct, TDC2023 (as in Arditi et al.) | 160 (128 train / 32 validation) |
| `refusal/harmless` | Harmless instructions | Alpaca | 160 |
| `refusal/harmless_lookalike` | Harmless prompts that look similar on the surface | Written for this tool | 50 |
| `refusal/paraphrase`, `refusal/it` | Shift-test variants | Mechanical wrappers / MultiJail (`it`) | 50 each |
| `backdoor/contrast` | Generic contrast prompts for the probe | Written following MacDiarmid et al. | 100 |
| `backdoor/triggers` | Candidate trigger strings, including the test model's own trigger | Test model cards, plus generic candidates | 20 |
| `pretrain/harm_recognition` | Harmful and harmless instructions in base-model format | Derived from the refusal sets | 200 |

**Build steps:**
1. Write a fetch script, `datasets/fetch.py`, that downloads from the original sources and records the source URL and license in `card.md`.
2. Mark any dataset containing harmful text with `contains_harmful: true`. The API never returns raw text from these datasets to agents (see A1).
3. Write the shift-test variants by hand or with a script.

**Done when:** `list_datasets()` returns all of the above, and every card has a source and license.

**Time:** 3 hours (T4).

### D2. Test models with known problems

**Purpose:** models where we know the right answer, so the demo and the tests can show a check catching something real.

**Files:** `organisms/fetch.py`, `organisms/verify.py`, `organisms/README.md`, `organisms/results.md`

**Models:**

| Use | Model | Known problem | Where it comes from |
|---|---|---|---|
| Backdoor (main) | `senku21x/mot-sleeper-agents-generality`, the Qwen3-4B adapter and its clean control adapter | Trigger-activated behavior | Community. Needs GPU or patience on CPU |
| Backdoor (small) | `Antonhansel/qwen2.5-1.5b-sleeper-canary` | Trigger-activated harmless marker | Community |
| Sycophancy | Qwen2.5-0.5B-Instruct plus our own LoRA trained to agree with users | Induced sycophancy | We build it (D3) |
| Refusal / drift | Emergent-misalignment small models (0.5B), with their base | Broad misalignment after narrow fine-tuning | Published follow-up work |
| Pretraining timeline | `EleutherAI/pythia-70m`, six revisions | None. Used to show change over training | EleutherAI |
| SAE features | Gemma 3 270M (instruction-tuned), with Gemma Scope 2 | None | Google DeepMind |

**Build steps:**
1. `fetch.py` downloads each model using E1's safe loading rules: safetensors only, no remote code.
2. `verify.py` runs each claimed problem behaviorally. For example, it fires the trigger 50 times and counts the triggered outputs, then compares with the clean control. Results go in `organisms/results.md`.
3. Drop any model whose claimed behavior does not reproduce, and note why.
4. Pick the final demo set based on what passes and what runs fast enough on the available hardware.

**Done when:** `organisms/results.md` lists each model with its measured behavior, and at least one backdoor model is confirmed.

**Time:** 3 hours (T4), mostly download and verification time. Start downloads in hour 1.

### D3. Demo fine-tunes

**Purpose:** create the sycophantic test model, plus optionally a model whose refusals were weakened by harmless fine-tuning.

**Files:** `organisms/finetune_sycophancy.py`, `organisms/finetune_benign.py`

**Build steps:**
1. Sycophancy: a LoRA on Qwen2.5-0.5B-Instruct using about 500 examples where the assistant agrees with a wrong user claim after pushback. Keep the training data out of the evaluation sets.
2. Optional benign fine-tune: a LoRA on ordinary instruction data, which may weaken refusals. The survey cites work showing later fine-tuning can recover suppressed behavior. Measure the effect rather than assuming it.
3. Save adapters as safetensors, and record training settings.

**Done when:** the sycophantic adapter abandons correct answers clearly more often than the base model on held-out pushback prompts.

**Time:** 2 hours on a GPU. On a laptop CPU, a 0.5B LoRA with a few hundred steps is possible but slow, so start it early or use a free cloud notebook.

---

## 6. Agent components

### A1. Built-in agent

**Purpose:** the conversational agent behind the web app's chat.

**Depends on:** E13 (stub is fine).

**Files:** `agent/session.py`, `agent/prompts/system.md`, `agent/config.py`

**Interface:**

```python
class ChatSession:
    def __init__(self, engine_mcp_url: str, workspace: Path, model: str = "claude-opus-5-5"): ...
    async def send(self, user_text: str, context: dict | None = None) -> AsyncIterator[dict]
        # yields streamed message parts: text, plan, question, finding, tool activity
    async def close(self) -> None
```

**Build steps:**
1. Set up the Claude Agent SDK (Python) client. Connect it to the engine's MCP server, and allow only the engine's tools. Turn off the SDK's built-in file and shell tools for the web app agent; it does not need them.
2. Model `claude-opus-5-5` with adaptive thinking, and server-side refusal fallbacks enabled. Check the Agent SDK documentation for how these options are passed before writing code. Do not guess option names.
3. The system prompt (`prompts/system.md`) covers:
   - **Role:** help the user investigate safety risks in their own model, using only the engine's tools.
   - **Conversation:** plain language. Ask for missing information instead of guessing. Ask the risk-specific questions from each check's `describe_check` output.
   - **Plans:** always draft with `draft_plan`, explain it in plain language, revise on request, and never run without approval.
   - **Findings:** only report findings from `get_findings`, and keep their strength and maturity labels. Never say "no problem exists". State coverage instead.
   - **The 3D view:** call `view_focus` when talking about a component, and `view_paint` when showing a measurement.
   - **Harmful content:** refer to datasets by ID, and never ask for or repeat harmful prompt text.
   - **Experimental checks:** explain the limits before running them.
4. `/ws/chat` in E13 creates a `ChatSession` per browser session and streams its output.
5. Context from the 3D view (a clicked component) is passed as `context={"component_id": ...}` and added to the user message.

**Done when:** in a terminal test harness, the conversation "Is my fine-tune more sycophantic?" leads to `load_model`, `describe_check`, `draft_plan`, a plain-language plan, a wait for approval, then `run_plan` after "yes, go".

**Tests:** a scripted conversation against the stub engine, checking the tool-call order and that `run_plan` is never called before approval.

**Time:** 3 hours (T2).

**Notes:** the agent can't see harmful text, so it can't paraphrase it into a report. Reports get redacted examples from the engine directly.

### A2. Reviewer agent

**Purpose:** an independent check of each finding's wording against its evidence.

**Depends on:** E10, E13 (`get_evidence`, `submit_review`).

**Files:** `agent/reviewer.py`, `agent/prompts/reviewer.md`

**Build steps:**
1. A subagent definition with its own system prompt and tool access restricted to `get_evidence` and `submit_review`.
2. Reviewer prompt rules:
   - Check that the claim matches the strength level.
   - Check that no absence claim goes beyond the coverage.
   - Check that failed shift tests are mentioned.
   - Check that quantization or a small model size is mentioned when relevant.
   - Lower the strength if needed. Never raise it.
3. The engine triggers a review automatically when a run completes, through a hook in the runner that asks the agent layer to review each new finding. Findings show "review pending" until then.
4. The model is configurable, defaulting to the same model as A1.

**Done when:** given a fixture finding that overclaims ("no backdoor exists"), the reviewer lowers it or rewrites the claim within coverage.

**Time:** 1.5 hours (T2).

### A3. Approval and logging

**Purpose:** make sure nothing runs without the user's approval, and keep a log of every tool call.

**Depends on:** A1, E13.

**Files:** `agent/hooks.py`, `engine/api/approval.py`

**Build steps:**
1. **Engine side:** `run_plan` requires the plan to be marked approved. Approval is recorded through `POST /api/plans/{id}/approve`, called by the web app when the user approves in chat. The web app watches for the agent asking for approval and turns the user's reply ("yes", "go ahead") into that call. The user still types plain language, and the button is optional.
2. **Agent side:** a pre-tool-use hook blocks `run_plan` unless the latest plan version is approved, and tells the agent to ask the user.
3. **In Claude Code:** the MCP tool permission prompt shown before `run_plan` is the approval. The tool marks the plan approved when it is called from a stdio or MCP client with an `approved_by_user: true` argument, and documents that Claude Code will prompt the user for it.
4. Log every tool call (name, arguments without secrets, result size, time) to `runs/<run_id>/agent_log.jsonl`, or to a session log.
5. Compute limits: refuse plans whose estimate exceeds a configurable maximum unless the user confirms again.

**Done when:** the A1 test harness cannot run a plan without approval, and the log contains every call.

**Time:** 1.5 hours (T2).

### A4. Claude Code plugin

**Purpose:** use the same engine from Claude Code.

**Depends on:** E13, E14.

**Files:**

```
plugin/
  .claude-plugin/plugin.json     # name, version, description
  .mcp.json                      # runs `tool mcp-stdio --workspace ${CLAUDE_PLUGIN_ROOT}/../workspace` or connects to a running server
  skills/
    investigate-risk/SKILL.md    # how to use the tools: describe_check -> draft_plan -> approval -> run -> findings
  agents/
    reviewer.md                  # the reviewer as a plugin subagent
  README.md
```

**Build steps:**
1. Write the plugin manifest and `.mcp.json` following the Claude Code plugins reference. Check the current format before writing.
2. Write the skill with the same rules as the A1 system prompt, shortened.
3. Write the reviewer subagent with the same rules as A2.
4. Test it: install the plugin locally, ask Claude Code "check this model for weakened refusals", and confirm the results appear in the web app (shared workspace).

**Done when:** a Claude Code session can run the refusal check end to end, and the run shows up in the web app.

**Time:** 1.5 hours (T2).

---

## 7. Web app components

### W1. App shell

**Purpose:** layout, state and connection to the engine.

**Depends on:** C0.

**Files:** `web/src/main.tsx`, `web/src/App.tsx`, `web/src/store.ts`, `web/src/api.ts`, `web/src/events.ts`, `web/src/dev/fakeEvents.ts`

**Build steps:**
1. Layout: chat panel on the left (about 35% width) and the 3D view on the right, with a timeline bar under the view and a drawer for charts and reports. On narrow screens the chat moves on top.
2. Zustand store holding:
   - the current model map
   - the series
   - the latest painted values per checkpoint
   - the focused components
   - jobs
   - findings
   - chat messages
3. `api.ts`: typed fetch helpers for the HTTP endpoints.
4. `events.ts`: a WebSocket client that dispatches events into the store and reconnects automatically.
5. `fakeEvents.ts`: replays `events_sycophancy_run.jsonl` at adjustable speed, so the web app can be built before the engine is ready. Turn it on with `?fake=1`.
6. Colors defined as CSS variables, with light and dark themes.

**Done when:** the app loads a fixture model map and replays fake events into the store.

**Time:** 1.5 hours (T3).

### W3. 3D model view

**Purpose:** draw the model and respond to focus, paint, timeline and click.

**Depends on:** W1, C0.

**Files:** `web/src/view/ModelScene.tsx`, `layout.ts`, `LayerBlock.tsx`, `AttentionHeads.tsx`, `ResidualLine.tsx`, `CameraRig.tsx`, `Legend.tsx`, `colors.ts`, `Grid2D.tsx` (fallback)

**Build steps:**
1. `layout.ts`: a pure function from model map to positions and sizes.
   - Embedding at the bottom, layers stacked upwards with fixed spacing, unembedding at the top.
   - Each layer is a flat block. When expanded, it splits into an attention plate and an MLP plate side by side.
   - Heads are a row of small cubes on the attention plate.
   - Write it as a pure function so it can be unit-tested.
2. `ModelScene.tsx`: a React Three Fiber `<Canvas>` with lights and a neutral background matching the theme. It renders the components from the layout.
3. `LayerBlock.tsx`:
   - The color comes from the store's painted value for the current checkpoint, or neutral gray if there is none.
   - Animate color and expansion with `@react-spring/three`.
   - Label with drei `<Text>`.
   - Hovering highlights the block. Clicking selects it and sends the component ID to the chat as context.
4. `AttentionHeads.tsx`: one `InstancedMesh` per attention plate, with per-instance color.
5. `ResidualLine.tsx`: a thin vertical line through the stack.
6. `CameraRig.tsx`: drei `<CameraControls>`. On a `focus` event, find the target object(s) and call `fitToBox(target, true)`. Allow rotate, zoom and pan.
7. `Legend.tsx`: the color scale and metric name. Use a colorblind-safe sequential scale (cividis or viridis) for values from 0 to 1, and a diverging scale for signed differences.
8. `Grid2D.tsx`: the 2D fallback (layers by component kind), using the same colors.
9. Performance: target 60 frames per second for 36 layers with 16 heads each. Avoid re-creating geometry when only colors change.

**Done when:** with fake events, layers change color in order, the camera moves to the focused layer, clicking a layer adds it to the chat input, and the 2D fallback shows the same data.

**Tests:** unit tests for `layout.ts` (positions for 6, 18 and 36 layers) and `colors.ts`.

**Time:** 5 hours (T3). This is the biggest web piece.

**Notes:**
- Build the plain stacked view first, then expansion into attention and MLP, then heads.
- Each step should be demo-able on its own.
- Pin package versions from npm at setup.

### W2. Chat panel

**Purpose:** the conversation with the built-in agent.

**Depends on:** W1, E13 `/ws/chat` (or a mock).

**Files:** `web/src/chat/ChatPanel.tsx`, `Message.tsx`, `PlanMessage.tsx`, `FindingMessage.tsx`, `QuestionMessage.tsx`, `ToolActivity.tsx`

**Build steps:**
1. Message list with streaming text. Markdown rendering for the agent's messages.
2. Special renderers:
   - **Plans:** a readable summary of what will be measured, the controls, coverage and estimate. The user still answers in text.
   - **Questions:** the job is paused, shown with the question.
   - **Findings:** the claim, strength and maturity badges, and a "show evidence" link that opens W5.
   - **Tool activity:** a collapsed line such as "running sycophancy probes, layer 7 of 24".
3. Input box with optional suggestion chips. Clicking a chip only fills the input. A component selected in the 3D view appears as a removable chip in the input.
4. While a plan is waiting for approval, an optional "Approve" button sits next to the plan. Typing approval works the same (A3).
5. A mock mode that replays a recorded conversation.

**Done when:** a full fake conversation renders, and in real mode the A1 harness conversation works through the browser.

**Time:** 3 hours (T3).

### W4. Timeline slider

**Purpose:** step through checkpoints or stages.

**Depends on:** W1, W3.

**Files:** `web/src/view/Timeline.tsx`

**Build steps:**
1. Shown only when the store has a series.
2. Ticks are labelled with checkpoint names (for example "step 8,000" or "preference tuning").
3. Dragging changes the current checkpoint in the store, and W3 repaints from stored values.
4. `timeline` events move the slider. A play button steps through checkpoints automatically.
5. Missing values for a checkpoint show as gray, not zero.

**Done when:** with the Pythia fixture series and fake results, dragging repaints the layers, and play animates through training.

**Time:** 1.5 hours (T3).

### W5. Charts

**Purpose:** the 2D evidence behind each finding.

**Depends on:** W1, E13 results endpoint.

**Files:** `web/src/charts/LayerCurve.tsx`, `CompareModels.tsx`, `ShiftTable.tsx`, `ExamplesList.tsx`

**Build steps:**
1. `LayerCurve`: the metric per layer with its control band. Clicking a point focuses that layer in 3D.
2. `CompareModels`: model and reference side by side.
3. `ShiftTable`: shift tests with pass and fail.
4. `ExamplesList`: redacted examples from `get_examples`.

**Done when:** opening a finding's evidence shows all four for the fixture run.

**Time:** 2 hours (T3).

### W6. Report viewer and W7. Claude Code setup panel

**Files:** `web/src/reports/ReportView.tsx`, `web/src/setup/ClaudeCodePanel.tsx`

**Build steps:**
1. Report viewer: an iframe of `/api/reports/{id}`, plus a list of past reports.
2. Setup panel: the command from `/api/setup/claude-code` with a copy button, and a short explanation that the same engine, workspace and results are used.

**Done when:** a report opens inside the app, and the setup command can be copied.

**Time:** 1 hour total (T3).

---

## 8. Integration checkpoints

Stop and integrate at these points, even if individual pieces are unfinished.

| Checkpoint | Hour | Must work |
|---|---|---|
| IC1 | 6 | `tool serve` runs. The web app shows the real model map of Qwen2.5-0.5B from `/api/models/{id}/map`. The agent harness calls `list_checks` and `load_model` on the real API. Activations and probes run on a laptop |
| IC2 | 11 | One full sycophancy run from the chat: plan, approval, live paint and focus in 3D, verified findings, report. Controls run |
| IC3 | 15 | Backdoor check on the verified test model and its clean control. Pythia timeline in the web app. Reviewer runs on every finding |
| IC4 | 18 | Refusal check. Claude Code plugin runs a check, and the result shows in the web app. Feature freeze |

After IC4, only fix bugs, write the README, rehearse the demo and record the backup video.

---

## 9. Hour-by-hour schedule per track

| Hours | T1 engine core | T2 API, agent, reports | T3 web app | T4 data, models, demo |
|---|---|---|---|---|
| 0 to 1 | Phase 0 (all) | Phase 0 (all) | Phase 0 (all) | Phase 0 (all). Start model downloads |
| 1 to 3 | E3, E1 | E13 stub with fixture data, E11 | W1 shell, fake event player | D1 fetch script, refusal and sycophancy datasets |
| 3 to 6 | E2, start E4 | A1 against the stub API | W3 layer stack, colors, camera focus | D2 fetch and verify sleeper models. Start D3 sycophancy LoRA |
| IC1 (6) | | | | |
| 6 to 9 | E4 done, E10 | E13 wired to the real engine, E9 | W2 chat (mock, then real) | D1 shift-test variants, backdoor sets |
| 9 to 11 | E8 framework, sycophancy check | E12 reports, A3 approval | W3 expansion into attention and MLP, click to chat | D3 finish and verify. E6 timeline on Pythia |
| IC2 (11) | | | | |
| 11 to 13 | Backdoor check | A2 reviewer | W4 timeline, W5 charts | E6 done. Test models on the backdoor check with T1 |
| 13 to 15 | Refusal check | E14 CLI | W3 heads (instanced) | E5 SAEs for Gemma 3 270M (optional) |
| IC3 (15) | | | | |
| 15 to 18 | Steering fix plus re-test in the sycophancy check | A4 plugin | W6, W7, polish, 2D fallback | E15 monitor (optional). Demo script draft |
| IC4 (18) | | | | |
| 18 to 20 | Bug fixes | README, install steps | Bug fixes, visual polish | Rehearsal, backup video |

---

## 10. What to cut if time runs short

Cut from the top of this list first:

1. E15 training monitor (keep it as a described feature)
2. E5 SAE features
3. Attention-head detail in the 3D view (keep layers, attention and MLP)
4. A4 plugin (show the MCP connection from Claude Code with a manual `claude mcp add` instead)
5. The refusal check (keep sycophancy and backdoor)
6. The steering fix and re-test (keep detection and findings)
7. Reviewer agent A2 (keep the engine's E10 strength rules, which already enforce controls)

Never cut:

- the required controls (E10)
- coverage and "not shown" text
- the plan-approval rule
- the safe model loading rules
- the pretraining timeline. It is cheap on Pythia-70M and is a main part of the pitch

---

## 11. Demo readiness checklist

- [ ] Demo models downloaded and verified (`organisms/results.md`)
- [ ] `tool doctor` shows everything needed is present
- [ ] Sycophancy run completes in the time the demo allows on the demo machine. Pre-compute it if not, and replay the stored results live
- [ ] Backdoor run against the clean control completes, or is pre-computed
- [ ] Pythia timeline results stored for six checkpoints
- [ ] 3D view focus, paint and timeline all work with real events
- [ ] A report opens and shows coverage and "what this does not show"
- [ ] Claude Code connection works on the demo machine
- [ ] Backup video recorded
- [ ] README has install and run steps that someone else has followed successfully
- [ ] Wording in the demo matches the evidence. No "the model is safe" claims

Pre-computing is honest only if it is said out loud during the demo, for example: "this run takes about ten minutes, so here are its stored results".

---

## 12. After the hackathon

In rough priority order:

1. The hidden bias, behavior drift, evaluation awareness, data poisoning and alignment faking checks, using the test models in design.md section 11.
2. Full sandboxing of agent-written code, running in a container with no network.
3. Training data attribution (E7) and the local learning coefficient (E6).
4. SAE training for custom models without published SAEs.
5. Measured catch rates for each check on a public set of test models, published with the tool.
6. Support for mixture-of-experts and state-space architectures.
7. A contributor guide for writing new risk checks as separate packages.
8. A license decision, and a first tagged release.
