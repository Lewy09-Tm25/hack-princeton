# Shared data formats

Field-by-field reference for the data formats shared between the Python engine (`engine/contracts.py`) and the TypeScript web app (`web/src/contracts.ts`). Update all three files together whenever any format changes.

---

## 1. `Component`

Describes one addressable part of a language model.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `str` / `string` | Yes | Stable component ID such as `"embed"`, `"L14"`, `"L14.attn"`, `"L14.attn.h3"`, `"L14.mlp"`, or `"unembed"` |
| `kind` | `"embedding" \| "layer" \| "attention" \| "attention_head" \| "mlp" \| "norm" \| "unembedding" \| "other"` | Yes | Role of the component in the architecture |
| `index` | `int \| null` | No | Zero-based layer index when the component belongs to a layer |
| `head` | `int \| null` | No | Zero-based attention head index for `"attention_head"` components |
| `n_heads` | `int \| null` | No | Number of query attention heads on `"attention"` components |
| `module` | `str \| null` | No | Dotted PyTorch module path used for hooks (omitted for virtual sub-components such as individual attention heads) |
| `children` | `list[str]` / `string[]` | No (defaults to `[]`) | Ordered IDs of direct child components |

---

## 2. `ModelMap`

Uniform structural map of a loaded model.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `str` / `string` | Yes | Internal or Hugging Face model identifier |
| `revision` | `str \| null` | No | Git branch, tag, or commit revision (for example `"main"` or `"step1000"`) |
| `arch_family` | `str \| null` | No | Architecture family name (for example `"gpt_neox"`, `"qwen2"`, `"gemma3"`) |
| `n_layers` | `int` / `number` | Yes | Number of transformer blocks in the main stack |
| `hidden_size` | `int \| null` | No | Residual stream dimension |
| `confirmed_by_user` | `bool` / `boolean` | Yes | `true` when all components were classified or confirmed by the user |
| `needs_confirmation` | `list[str]` / `string[]` | No (defaults to `[]`) | Dotted module paths that the automatic map builder could not classify |
| `components` | `list[Component]` / `Component[]` | Yes | Flat list of all model components; hierarchy is expressed via `children` |

---

## 3. `SeriesItem` and `Series`

Ordered sequence of pretraining checkpoints or post-training stages.

### `SeriesItem`

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint_id` | `str` / `string` | Yes | Identifier used in results and timeline events (for example `"step1000"` or `"stage:dpo"`) |
| `label` | `str` / `string` | Yes | Plain human-readable label (for example `"step 1,000"` or `"preference tuning"`) |
| `order` | `int` / `number` | Yes | Zero-based position in the series |
| `source` | `str` / `string` | Yes | How to load this item (for example `"revision:step1000"` or a local path/repo ID) |

### `Series`

| Field | Type | Required | Description |
|---|---|---|---|
| `series_id` | `str` / `string` | Yes | Unique identifier for the series |
| `kind` | `"pretraining" \| "post_training"` | Yes | Whether the series steps through pretraining checkpoints or post-training stages |
| `model_family` | `str` / `string` | Yes | Base model or family shared across items (for example `"EleutherAI/pythia-70m"`) |
| `items` | `list[SeriesItem]` / `SeriesItem[]` | Yes | Ordered checkpoints or stages |

---

## 3b. `Plan`

Experiment plan drafted by `draft_plan` / `revise_plan` and executed only when `approved` is `true`.

| Field | Type | Required | Description |
|---|---|---|---|
| `plan_id` | `str \| null` | No | Assigned when stored in `Record` |
| `experiment_id` | `str \| null` | No | Parent experiment ID in `Record` |
| `version` | `int` / `number` | No (defaults to `1`) | Monotonic plan revision number |
| `check` | `str` / `string` | Yes | Risk check name |
| `model_id` | `str` / `string` | Yes | Target model ID |
| `reference_model_id` | `str \| null` | No | Clean or base reference model ID |
| `series_id` | `str \| null` | No | Checkpoint series ID when running across checkpoints/stages |
| `checkpoints` | `list[str] \| null` | No | Selected checkpoint IDs when `series_id` is set |
| `datasets` | `list[str]` / `string[]` | Yes | Dataset IDs used by the run |
| `layers` | `list[int] \| "all"` | No (defaults to `"all"`) | Layer indices to inspect |
| `controls` | `list[str]` / `string[]` | Yes | Control kinds (always includes required controls) |
| `shift_tests` | `list[str]` / `string[]` | Yes | Shift-test kinds |
| `user_context` | `dict` / `Record<string, unknown>` | No (defaults to `{}`) | Domain notes and user preferences |
| `estimate_seconds` | `float` / `number` | Yes | Estimated wall-clock runtime on the detected hardware |
| `coverage` | `str` / `string` | Yes | Plain-language statement of what the plan examines |
| `not_covered` | `str` / `string` | Yes | Plain-language statement of what the plan does not cover |
| `maturity` | `"established" \| "experimental"` | Yes | Maturity of the check's method |
| `papers` | `list[str]` / `string[]` | Yes | Source paper citations |
| `approved` | `bool` / `boolean` | No (defaults to `false`) | Must be `true` before `run_plan` executes; reset to `false` on revision |
| `is_placeholder` | `bool` / `boolean` | No (defaults to `false`) | `true` when using placeholder datasets or synthetic test models |

---

## 4. `ControlResult`

Outcome of a control experiment or shifted-input test.

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | `str` / `string` | Yes | Name of the control or shift test (for example `"shuffled_labels"`, `"random_direction"`, `"paraphrase"`, `"multilingual"`, `"multi_turn"`) |
| `value` | `float \| null` | No | Measured control or shift value when numeric |
| `passed` | `bool \| null` | No | `true` if passed, `false` if failed, `null` if not yet evaluated or `"not run"` |
| `notes` | `str \| null` | No | Optional short note (for example `"not run: no multilingual dataset approved"`) |

---

## 5. `ResultRecord`

One row per measurement stored in the experiment record.

| Field | Type | Required | Description |
|---|---|---|---|
| `run_id` | `str` / `string` | Yes | Identifier of the experiment run |
| `check` | `str` / `string` | Yes | Risk check name (for example `"sycophancy"`, `"refusal"`, `"backdoor"`) |
| `component_id` | `str` / `string` | Yes | Component measured (for example `"L14"` or `"model"` for behavioral rates) |
| `checkpoint_id` | `str \| null` | No | Checkpoint or stage ID when part of a series |
| `metric` | `str` / `string` | Yes | Metric name (for example `"probe_accuracy"`, `"probe_auroc"`, `"flip_rate"`) |
| `value` | `float` / `number` | Yes | Measured numeric value |
| `control` | `ControlResult \| null` | No | Paired control measurement for this component/metric |
| `dataset_id` | `str` / `string` | Yes | Dataset identifier from the dataset registry |
| `split` | `"train" \| "held_out"` | Yes | Data split used for the reported value |
| `seed` | `int` / `number` | Yes | Random seed used |
| `device` | `str` / `string` | Yes | Execution device (`"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, `"ndif"`) |
| `precision` | `str` / `string` | Yes | Floating-point precision (`"fp32"`, `"bf16"`, `"fp16"`) |
| `quantized` | `bool` / `boolean` | Yes | Whether weights were quantized when measured |
| `is_placeholder` | `bool` / `boolean` | No (defaults to `false`) | `true` when produced from synthetic placeholder data or a random-weight test model |

---

## 6. `Finding` and `ReviewerVerdict`

### `ReviewerVerdict`

| Field | Type | Required | Description |
|---|---|---|---|
| `agrees` | `bool` / `boolean` | Yes | Whether the reviewer agrees with the claim as worded |
| `notes` | `str` / `string` | No (defaults to `""`) | Explanation of any wording change or strength reduction |
| `corrected_claim` | `str \| null` | No | Rewritten claim bounded to the evidence and coverage |
| `lower_strength_to` | `"suggestive" \| "inconclusive" \| null` | No | Reduced strength level (never higher than the engine's strength) |

### `Finding`

| Field | Type | Required | Description |
|---|---|---|---|
| `finding_id` | `str` / `string` | Yes | Unique identifier for the finding |
| `run_id` | `str` / `string` | Yes | Experiment run that produced the finding |
| `check` | `str` / `string` | Yes | Risk check name |
| `claim` | `str` / `string` | Yes | Plain-language summary bounded by strength and coverage |
| `strength` | `"established" \| "suggestive" \| "inconclusive"` | Yes | Strength computed by engine verification (and optionally lowered by the reviewer) |
| `maturity` | `"established" \| "experimental"` | Yes | Maturity of the underlying interpretability method |
| `evidence` | `list[str]` / `string[]` | Yes | References to `run_id` and metric rows backing the claim |
| `controls` | `list[ControlResult]` / `ControlResult[]` | Yes | Required control outcomes |
| `shift_tests` | `list[ControlResult]` / `ControlResult[]` | Yes | Required shifted-input test outcomes |
| `coverage` | `str` / `string` | Yes | Explicit statement of layers, datasets, languages, and seeds examined |
| `not_shown` | `str` / `string` | Yes | Explicit statement of what the measurement does not establish |
| `verified` | `bool` / `boolean` | No (defaults to `false`) | `true` only after all required controls and shift tests ran on real (non-placeholder) models and datasets |
| `is_placeholder` | `bool` / `boolean` | No (defaults to `false`) | `true` when produced from synthetic placeholder data or random-weight test models; cannot be `verified: true` |
| `reviewer_verdict` | `ReviewerVerdict \| null` | No | Reviewer subagent verdict (`null` while review is pending) |

---

## 7. `Event`

Live events streamed from `/ws/events` to the web app. Every event has `type` and `payload`:

| `type` | Required `payload` fields | Optional `payload` fields | Description |
|---|---|---|---|
| `"focus"` | `component_ids: string[]`, `reason: string` | — | Smoothly moves the 3D camera to frame the listed components |
| `"paint"` | `metric: string`, `values: Record<string, number>`, `scale: "sequential" \| "diverging"` | `checkpoint_id?: string \| null`, `run_id?: string \| null` | Colors model components using `cividis` (`"sequential"`, 0 to 1) or blue-to-orange (`"diverging"`, signed differences) |
| `"timeline"` | `checkpoint_id: string` | — | Selects a checkpoint or stage in the timeline slider |
| `"progress"` | `job_id: string`, `step: number`, `total: number`, `message: string` | — | Reports background job progress |
| `"question"` | `job_id: string`, `text: string` | — | Pauses a job to ask the user a decision question |
| `"finding"` | `finding_id: string` | — | Signals that a finding is ready to fetch |
| `"job_status"` | `job_id: string`, `status: "queued" \| "running" \| "needs_decision" \| "completed" \| "stopped" \| "failed"` | `message?: string \| null` | Updates the lifecycle state of a background job |
