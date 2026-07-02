# 0040. Hybrid mode runs dual extraction

Date: 2026-06-25
Status: Accepted

## Context
The `feature_importance_counterfactual` mode produces a two-section narrative (ADR 0037 templated generator, ADR 0039 LLM template): a feature-importance section explaining the prediction and a counterfactual section describing what would flip it. Claim extraction, however, was still single-path. ADR 0033 dispatched extraction by mode with a two-way branch (`counterfactual` runs the CF extractor, everything else runs the FI extractor) and explicitly deferred the hybrid case, letting it fall through to feature-importance extraction only, because at that time no hybrid generator existed and a hybrid request could reach the Explainer only by a user pairing a single-mode generator with the hybrid config. ADR 0033 also flagged that a hybrid narrative carries both claim types and that `guardrail_tokens_used`, a single `dict[str, int]` sized for one judge call, would need to account for two extractions once the hybrid path was real. With both hybrid generation paths now in place, a hybrid narrative contains genuine feature-importance claims and genuine counterfactual claims, and extracting only one half discards the other.

## Decision
Replace the two-way extraction dispatch with a three-way dispatch. `mode == "counterfactual"` runs the counterfactual extractor only; `mode == "feature_importance"` runs the feature-importance extractor only; `mode == "feature_importance_counterfactual"` runs both extractors over the same generated text. Each extractor sees the whole narrative rather than a pre-split section, because the two claim types are distinct (the FI extractor looks for rank/sign/value claims, the CF extractor for before/after claims) and each naturally resolves its own section; not splitting keeps extraction robust to narratives that do not divide cleanly on the section boundary. The two extractions populate their result fields independently: `narrative_extraction` is set if the FI extraction succeeded, `counterfactual_extraction` if the CF extraction succeeded, so a partial success (one extractor returns an advisory failure) still yields the half that worked, with each advisory failure appended to `guardrails` independently. `guardrail_tokens_used` is the element-wise sum of the two judge calls' token dicts, computed by a new `_merge_token_counts(a, b)` helper: both None yields None, one None yields a shallow copy of the other, two dicts yield the per-key sum over the union of keys. The field keeps its `dict[str, int] | None` type and its meaning (tokens the guardrail layer consumed); the hybrid simply consumed the sum of two calls.

## Rationale
- A hybrid narrative has both claim types, so both must be extracted. Extracting only the FI half (the ADR 0033 fall-through) silently dropped every counterfactual claim, which defeats the point of a hybrid explanation and would make the counterfactual fidelity metrics unusable on hybrid output.
- Running both extractors over the whole text avoids coupling extraction to the exact two-section format. Splitting on the section boundary would be more precise but brittle: an LLM hybrid narrative may not split cleanly, and if an extractor ever picks up the wrong section that is a real quality signal, not something to mask by pre-splitting.
- Partial population matches the existing single-mode behavior. Each single-mode branch already sets its extraction field only on success and records a failure without discarding the result; the hybrid branch applies the same pattern twice, so a CF parse failure does not cost the user the FI claims and vice versa.
- Summing the token dicts keeps the field's contract intact. `guardrail_tokens_used` means tokens the guardrail layer used; a hybrid used two judge calls, so the honest value is their sum, with no schema change and no per-extractor keying that downstream consumers would have to learn.

## Consequences
- `mode="feature_importance_counterfactual"` now populates both `narrative_extraction` and `counterfactual_extraction`; the counterfactual claims are no longer discarded.
- `guardrail_tokens_used` for a hybrid explanation is the sum of the FI and CF judge calls, resolving the accounting reshape ADR 0033 deferred. The single-mode paths are unchanged (one call, one dict).
- A new module-level helper `_merge_token_counts` is covered by five direct unit tests (both None, each side None, two dicts summed, disjoint keys). Four dispatch tests pin the hybrid behavior: both channels populate, the token sum is exact, and each partial-failure direction leaves the other channel intact with the advisory failure recorded.
- The ADR 0033 comment in `explainer.py` is updated from "hybrid falls through to FI" to the three-way description.
- The remaining hybrid piece is grading: a `HybridGrades` that composes the feature-importance and counterfactual grade types, plus a `grade_hybrid` entry point and render support. That is a separate commit.

## Rejected alternatives
- **Keep the ADR 0033 fall-through (FI extraction only for hybrid).** Rejected: it discards all counterfactual claims from a narrative that contains them, making the counterfactual metrics meaningless on hybrid output.
- **Split the narrative on the section boundary and run each extractor on its own section.** Rejected: it couples extraction to the exact `\n\n` two-section format, which an LLM narrative may not honor, and it hides cross-section leakage that is worth seeing. Running both over the whole text is simpler and more robust.
- **Fail the whole extraction if either extractor fails.** Rejected: it throws away a good half over a bad half. Partial population is more useful and matches how the single-mode paths already treat their own advisory failures.
- **Key `guardrail_tokens_used` by extractor (a nested dict).** Rejected: it changes the field's shape and forces every consumer to handle a hybrid-only structure. The element-wise sum keeps one flat `dict[str, int]` whose meaning does not change with mode.

## References
- ADR 0033 - Explainer dispatches extraction by mode (the two-way dispatch this extends; the deferral this resolves).
- ADR 0037 - templated hybrid generator (produces the two-section narrative).
- ADR 0039 - hybrid tabular prompt template (the LLM two-section narrative).
- `src/xains/explainer.py` - the three-way dispatch and `_merge_token_counts`.
- `tests/unit/test_explainer.py` - the four dispatch tests and five merge-helper tests.
