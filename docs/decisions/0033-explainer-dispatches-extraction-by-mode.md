# 0033. Explainer dispatches extraction by mode

Date: 2026-06-25
Status: Accepted

## Context
`Explainer.explain()` runs judge-based narrative extraction inside a gate (`config.run_guardrails and config.extract_narrative and isinstance(request, TabularExplanationRequest)`). Before this commit the gate called `extract_narrative_claims`, the feature-importance extractor, unconditionally: `ExplanationConfig.mode` was computed by `_validate_mode` but never consulted at the extraction site. With the counterfactual path now complete (CF prompt template in ADR 0029, CF templated generator in ADR 0030, single-counterfactual model in ADR 0031, CF extraction and grader in ADR 0032), a user constructing `mode="counterfactual"` with `extract_narrative=True` got a counterfactual narrative parsed by the feature-importance extractor. That extractor expects rank, sign, and value claims; a counterfactual narrative's "would change from X to Y" sentences either fail resolution and pile into the hallucinations channel or get coerced into nonsense feature-importance claims. The result was a real but quiet bug: the feature-importance grader would then score the malformed extraction and emit authoritative-looking numbers that reflect a mismatched extraction shape. The counterfactual notebook about to land would have demonstrated that bug; this ADR pre-empts it.

## Decision
Dispatch the extraction call inside `Explainer.explain()` on `ExplanationConfig.mode`. When `mode == "counterfactual"`, call `extract_counterfactual_claims` and populate a new `ExplanationResult.counterfactual_extraction: CounterfactualExtraction | None` field. Any other mode (`"feature_importance"`, and the hybrid `"feature_importance_counterfactual"`) keeps the existing `extract_narrative_claims` path and populates `narrative_extraction` exactly as before. The new field sits next to `narrative_extraction`, defaults to `None`, and preserves `extra="forbid"`. The `judge_llm` guard stays above the branch: it is mode-agnostic and pre-extraction, so both paths raise the same `ValueError` when `judge_llm` is missing while `extract_narrative=True`. `guardrail_tokens_used` is populated by whichever branch ran, and the advisory-`GuardrailResult` failure-append pattern is identical in both branches. The hybrid mode falls through to the feature-importance branch because no library-provided hybrid generator exists; a hybrid request reaching the extraction step means the user paired a single-mode generator with a hybrid config, and feature-importance extraction is the closest correct behavior we can offer without speculative plumbing.

## Rationale
- `narrative_extraction` is typed `NarrativeExtraction | None`; widening it to a union with `CounterfactualExtraction` would force every downstream consumer to narrow the type on every read. Two separate typed fields keep the consumer side honest and follow ADR 0008's orthogonality precedent (`ExtractionGrades` and `NarrativityGrades` did not merge; the extraction records follow suit).
- Mode is the canonical signal for which extraction shape the narrative carries. The feature-importance and counterfactual prompt templates and generators are chosen by mode at construction, and the LLM's output shape follows. Routing extraction on mode keeps one source of truth.
- The shared `judge_llm` guard stays above the branch because the prerequisite, that asking for extraction at all requires a judge LLM, is mode-agnostic. Duplicating the check inside each branch would duplicate the message and risk drift if it changes.
- Building hybrid extraction now would require running both extractors, reshaping `guardrail_tokens_used` to hold a token count per extraction, and growing `ExplanationResult` a hybrid-typed field or two-populated-fields semantics, with no consumer that reads both. No hybrid generator exists to produce a representative narrative, so this is deferred until the hybrid mode concretely lands.
- The bug being fixed is a quiet correctness bug, not a crash. A regression test (feature-importance mode still populates `narrative_extraction`) and a positive test (counterfactual mode populates `counterfactual_extraction`) make the fix durable.

## Consequences
- `ExplanationResult` gains one field, `counterfactual_extraction: CounterfactualExtraction | None = None`. Existing constructions stay valid because the field defaults; `extra="forbid"` still rejects unknown keys.
- `ExplanationResult.narrative_extraction` is now `None` whenever `mode == "counterfactual"`. Code that reaches into `narrative_extraction` without checking mode first will see `None` after this commit where it previously saw a malformed `NarrativeExtraction`. This is a desirable fix: the prior object was the feature-importance extractor's best-effort over a counterfactual narrative, not a useful extraction.
- `Explainer.explain()` gains a small dispatch (the `if mode == "counterfactual"` branch plus the CF call). The existing feature-importance branch body is unchanged byte-for-byte; its only diff is its new placement under `else:`.
- Four new unit tests pin the behavior: counterfactual dispatch populates the CF field; feature-importance dispatch still populates the FI field; a counterfactual parse failure logs an advisory; `judge_llm=None` with `mode="counterfactual"` raises the same shared `ValueError` as the feature-importance path.
- The counterfactual notebook can now call `explainer.explain(request)` and read `result.counterfactual_extraction` directly, mirroring the feature-importance quickstart flow.

## Rejected alternatives
- **Widen `narrative_extraction` to `NarrativeExtraction | CounterfactualExtraction | None`.** Rejected: it forces every consumer to type-narrow on every read and loses the per-mode type signature that catches passing a counterfactual extraction to `grade_extraction` at type-check time. ADR 0008 already preferred separate aggregates over a union at the metric layer; the extraction layer follows.
- **Dispatch by `isinstance(self.generator, ...)` instead of mode.** Rejected: the generator class is a construction detail, while mode is the user's stated intent and the validated invariant per `_check_explicit_mode`. A user could pair a feature-importance generator with `mode="counterfactual"`; mode is what tells the extractor what shape the narrative is.
- **Build hybrid extraction now.** Rejected: no hybrid generator exists, so there is no representative narrative to extract over. Dual extraction would also force `guardrail_tokens_used` to reshape and `ExplanationResult` to grow a hybrid record, with no consumer that reads both. Deferred until a hybrid generator lands.
- **Duplicate the `judge_llm`-None guard inside each branch.** Rejected: the prerequisite is mode-agnostic, so one shared check is the honest place for it. Duplication risks message drift and adds churn the next time the guard's reason changes.
- **Move dispatch into a new `extract` helper on the generator.** Rejected: the generator produces the narrative; extraction is a post-hoc judge concern. Coupling them would invert the ADR 0018 separation, where `LLMNarrativeGenerator` deliberately does not own extraction.

## References
- ADR 0007 - resolution at extraction time (the FI extractor contract this dispatch parallels).
- ADR 0008 - orthogonality across grade aggregates (precedent for separate typed fields, not a union).
- ADR 0018 - NarrativeGenerator abstraction (where the judge_llm-required ValueError lives).
- ADR 0031 - single counterfactual per request.
- ADR 0032 - CF fidelity scoring (this dispatch makes its grading path reachable from explain).
- `src/xains/explainer.py` - the dispatch.
- `src/xains/types.py` - the new counterfactual_extraction field.
- `tests/unit/test_explainer.py` - the four pinning tests.
