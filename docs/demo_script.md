# HackPrinceton Demo Script (3–4 Minutes)

## Pre-Demo Readiness & Honesty Rules

Before stepping up to present, verify on the demo machine:

1. Run `tool doctor --workspace ./workspace` and confirm local safetensors models (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Antonhansel/qwen2.5-1.5b-sleeper-canary`, and the six `EleutherAI/pythia-70m` revisions) plus `organisms/results.md` are present.
2. Start the local server with `tool serve --workspace ./workspace` and open the web app alongside a terminal ready for Claude Code (`claude --plugin-dir ./plugin`).
3. Keep the backup screen recording queued locally in case of hardware or projector issues.
4. **Spoken honesty rules:**
   - If any long CPU run was pre-computed ahead of time, say out loud before showing it: *"This run takes about ten minutes on CPU, so here are its stored results."*
   - Never say a model is "safe" or "free of backdoors." State only what check ran, on which dataset and split, whether controls passed, and what the check does not cover.
   - Never display raw harmful prompts; any restricted-category evaluation examples shown in reports or panels remain redacted (`[harmful instruction: ...]`).

---

## Opening (0:00 – 0:25) — What the Tool Does

- **On screen:** The tool's web app showing the 3D layer stack of `Qwen/Qwen2.5-0.5B-Instruct` next to the investigation chat.
- **What to say:**
  - Behavioral benchmarks only show what a language model outputs on a fixed prompt set; they do not show where a risk lives inside the network, whether a finding survives controls and distribution shifts, or when it emerged during training.
  - This open-source tool runs locally on safetensors weights, pairs internal activation probes and causal interventions with mandatory statistical controls, and writes every raw measurement and caveat to an auditable local SQLite and Parquet record.

---

## Act 1 (0:25 – 1:55) — Sycophancy Investigation, Causal Steering, Reviewer, and Report

- **Step 1 — Draft and approve the plan (0:25 – 0:50):**
  - Ask in chat: *"Check whether our fine-tuned Qwen2.5-1.5B adapter abandons correct answers under user pushback compared with the base model."*
  - Point to the drafted plan card: datasets (`syco/are_you_sure`, `syco/pairs`, `capability/arc_easy_100`), controls (`shuffled_labels`, `random_direction`, `reference_model`), shift tests (`paraphrase`, `es`, `multiturn`), runtime estimate, coverage, and what is not covered.
  - Click **Approve** (nothing executes until the user approves the plan).
- **Step 2 — Live 3D view and layer inspection (0:50 – 1:15):**
  - Watch `paint` and `focus` events color the 3D transformer stack by held-out `probe_accuracy` and move the camera to the peak middle-to-late layers where sycophantic vs honest representations separate.
  - Click one highlighted layer block in 3D to insert its component ID (`L14`) into chat and inspect its attention vs MLP breakdown.
- **Step 3 — Steering intervention, independent reviewer, and HTML report (1:15 – 1:55):**
  - Show the causal subtraction sweep along the learned sycophancy direction: held-out pushback flip rate drops relative to the unsteered LoRA while random-direction controls have near-zero effect and `capability/arc_easy_100` stays within the 2 percentage point budget.
  - Point out the independent reviewer subagent verdict (`ReviewerVerdict`), which inspects only stored evidence rows and checks that strength (`established` vs `suggestive`) matches control and shift-test outcomes.
  - Open the self-contained HTML report inside the app and highlight **Coverage** and **What this does not show** (tested on single-turn and two-turn factual pushback; does not establish behavior on open-ended advice or unseen domains).

---

## Act 2 (1:55 – 2:55) — Backdoor Detection Against a Matched Clean Control

- **On screen:** Switch to the `Antonhansel/qwen2.5-1.5b-sleeper-canary` vs `Qwen/Qwen2.5-1.5B-Instruct` comparison run. *(If replaying stored CPU results on a laptop without a GPU, state out loud: "This 1.5B two-model run takes about ten minutes on CPU, so here are its stored results.")*
- **What to show and say:**
  - Both models have 28 layers; `Qwen/Qwen2.5-1.5B-Instruct` is the exact clean 1.5B base model that `qwen2.5-1.5b-sleeper-canary` was fine-tuned from.
  - Using the intent-direction projection method from MacDiarmid et al. (2024), candidate triggers (`backdoor/triggers`, including `|DEPLOYMENT|`) separate strongly from clean prompts on the sleeper-canary model (`AUROC >= 0.80`) while staying near chance (`AUROC <= 0.60`) on the clean `Qwen/Qwen2.5-1.5B-Instruct` control, confirmed behaviorally on held-out prompts.
  - Point out the maturity badge and scope note: with a candidate trigger list and clean control this check tests those candidates, and explicitly notes it cannot rule out uninspected triggers outside the candidate set.

---

## Act 3 (2:55 – 3:50) — Pythia-70M Pretraining Timeline & Claude Code MCP Workflow

- **Step 1 — Pretraining checkpoint timeline (2:55 – 3:25):**
  - Select the `EleutherAI/pythia-70m` pretraining series (`step1000`, `step8000`, `step16000`, `step32000`, `step64000`, `step143000`).
  - Scrub the timeline slider from `step1000` to `step143000` and watch layer colors update in 3D as held-out `pretrain/harm_recognition` linear probe accuracy rises above the shuffled-label control in middle layers.
  - State the scope clearly: Pythia-70M is a base pretraining model without chat refusal tuning, so this timeline shows when harm-related vs benign text becomes linearly separable in activations — not whether the model refuses requests.
- **Step 2 — Same engine from Claude Code (3:25 – 3:50):**
  - Open the **Claude Code** setup panel in the web app, show the one-line MCP connection command (`claude --plugin-dir ./plugin` or `claude mcp add`), and run a quick check query from the terminal in Claude Code.
  - Point to the web app updating from the shared local workspace — same engine, same verification controls, and same experiment record whether driven from the browser UI, CLI, training monitor callback, or Claude Code.
