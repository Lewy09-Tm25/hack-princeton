# Training data attribution (component E7 — described only)

Status: interface and documentation only in this version (`engine/attribution/interface.py`). Calling `attribute(behavior_examples, corpus_ref)` raises `NotImplementedError`.

## Purpose

When a team investigates suspected data poisoning or an unexpected behavior that emerged during pretraining or fine-tuning, checkpoint timelines (`engine/dynamics/timeline.py`) narrow down *when* the behavior appeared. Training data attribution is the complementary step aimed at narrowing down *which training documents* contributed to that behavior so a human auditor can inspect them.

## Planned methods

1. **Gradient-based influence approximations (`kronfluence` or `dattri`):**
   - Given a small set of `behavior_examples` (prompts paired with the undesired completion) and a reference `corpus_ref` to the training dataset, Kronecker-factored approximate curvature (EK-FAC, via `kronfluence`) or random-projection TRAK scores (via `dattri`) estimate how down-weighting each training document would change the log-likelihood margin of the target behavior.
   - Computation runs over selected transformer MLP or attention layers to fit local memory budgets, returning ranked `DocumentScore` entries (`method="influence"`).

2. **Exact text-span search (`search_exact_spans`):**
   - Poisoning and planted backdoor triggers often rely on verbatim strings, rare formatting headers, or repeated boilerplate templates inside a small fraction of the training corpus.
   - Suffix-array or inverted-index search over `corpus_ref` locates exact matches for candidate trigger strings or distinctive output phrases (`method="span_match"`). Unlike gradient approximations, exact span matches directly show whether a string was present in training data.

## Required controls

- **Unrelated-behavior control (`docs/design.md` section 11.6):** High-gradient outlier documents (such as noisy formatting dumps or long token sequences) often rank near the top of influence lists regardless of the behavior queried. Any attribution implementation must run the same attribution query on unrelated baseline behaviors and discount documents that appear indiscriminately across unrelated queries.
- **Redacted excerpts:** If the corpus or behavior involves restricted safety categories, stored `DocumentScore.excerpt_redacted` values must remain inside the local workspace and redact restricted text before reaching an agent or HTML report.

## Known reliability limits

- **2024 LLM influence-function evaluations (`docs/design.md` section 11.6):** Empirical studies in 2024 evaluating influence functions on autoregressive language models found that first-order Hessian approximations often correlate weakly with true leave-one-out or group-removal retraining effects at scale, due to non-convexity, optimizer state across many epochs, and shared token statistics.
- **Leads, not proof:** Even when implemented, attribution scores must be presented as candidate documents for human inspection rather than proof that removing the top-ranked documents will eliminate the behavior. Only retraining without those documents and re-running the check verifies a causal effect.
