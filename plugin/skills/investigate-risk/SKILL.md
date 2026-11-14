---
name: investigate-risk
description: Plan, approve, run, verify, and report safety risk investigations inside language models using the local engine MCP tools.
---

# Investigating safety risks inside a language model

Use the tool's local engine MCP tools (`model-safety-engine`) to inspect a PyTorch language model, run interpretability checks with required controls, verify every finding, and write an HTML report. Keep explanations in plain language with short paragraphs.

## Workflow

1. **Inspect available checks and ask missing questions.**
   - Call `list_checks` and `describe_check` for each risk the user asks about before drafting a plan.
   - Check hardware settings with `get_compute` when helpful.
   - Ask one clear question at a time for details the check requires:
     - **Sycophancy:** what domain the model is used for, and whether the original base model before fine-tuning is available for comparison.
     - **Weakened refusals:** whether a base or prior post-training stage exists as a reference model.
     - **Backdoor detection:** whether a clean reference model trained on similar data is available, and whether specific candidate triggers should be tested alongside the default list.
     - **Pretraining or post-training series:** which checkpoints or stages should be registered.
     - **Data poisoning:** whether the training corpus is available locally.
   - If `describe_check` marks a check as `"experimental"`, tell the user before planning and explain that published evidence is limited (mostly from large models) and a negative result rules out very little.

2. **Load the model or series and check its structure.**
   - Call `load_model` for a single model (plus a reference model when applicable), or `register_series` for ordered pretraining checkpoints or post-training stages.
   - Call `get_model_map` to inspect layers, attention, MLP blocks, and hook paths.
   - If `needs_confirmation` is non-empty (custom or unfamiliar architecture), ask the user in plain language what each unclassified module is and record their answer with `confirm_components`.

3. **Draft and explain the plan.**
   - Call `draft_plan` and summarize the returned plan plainly:
     - what will be measured
     - which published papers the methods come from
     - which datasets will run
     - which required controls and shifted-input tests will run
     - what the plan covers (`coverage`) and what it will not cover (`not_covered`)
     - roughly how long it will take on this hardware (`estimate_seconds`)
   - If the user asks for adjustments, call `revise_plan`. Required controls cannot be removed without capping findings at `"suggestive"`; explain why each control is needed if asked.

4. **Wait for explicit user approval before running.**
   - **Never** call `run_plan` until the user explicitly approves the current version of the plan.
   - Every call to `revise_plan` increments the plan version and resets approval, requiring fresh user confirmation.
   - Once the user clearly says to go ahead (and confirms any long run over 20 minutes), call `run_plan(plan_id=..., approved_by_user=True)`. Note that Claude Code will also prompt the user before executing `run_plan`.

5. **Monitor execution and handle job questions.**
   - Poll `job_status` to report progress plainly.
   - If `job_status` reports `"needs decision"`, present the engine's question to the user and relay their answer with `answer_job`. Use `stop_job` if the user asks to cancel.

6. **Review findings, update the 3D view, and build the report.**
   - Fetch results with `get_findings`, `get_results`, and `get_examples`.
   - Delegate to the `reviewer` subagent so every finding in the run is checked against `get_evidence` and recorded with `submit_review` before presenting final claims.
   - When the user has the web app open against the same server/workspace:
     - call `view_focus` with `component_ids` when discussing specific layers or heads
     - call `view_paint` when presenting per-layer or per-checkpoint measurements
     - call `view_timeline` when stepping to a specific checkpoint or stage
   - Call `build_report` to write the self-contained HTML report in `workspace/reports/`, and offer `export_artifact` (`probe`, `steering`, or `monitor_config`) or a follow-up check if appropriate. Any steering fix is a new plan that requires fresh approval.

## Mandatory honesty and data-handling rules

- **Never claim a model is safe or free of a problem.** Never say `"the model is safe"` or `"no backdoor exists"`. Report only what was and was not detected within the stated `coverage` (for example, `"no above-threshold backdoor signal in layers 0 to 23 across the 20 tested trigger candidates"`), followed by `not_shown`.
- **Keep engine labels intact.** Report only findings returned by `get_findings`, preserving their exact `strength` (`"established"`, `"suggestive"`, `"inconclusive"`) and `maturity` (`"established"`, `"experimental"`) labels. Clearly state when results come from synthetic placeholder data (`is_placeholder: true`).
- **Harmful test data stays inside the engine.** Refer to harmful datasets and items strictly by dataset ID and item index (for example `refusal/harmful #12`). Never ask for, quote, repeat, or reconstruct raw harmful prompt text or unredacted model completions. If the user wants to inspect raw local outputs, tell them they are stored locally inside `workspace/`.
- **Plain wording only.** Describe methods and components by what they do, use numbers strictly from tool outputs, and never invent product-style labels for parts of the tool.
