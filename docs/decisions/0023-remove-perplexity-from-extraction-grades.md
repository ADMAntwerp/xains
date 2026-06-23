# 0023. Remove perplexity from ExtractionGrades

Date: 2026-06-19
Status: Accepted

## Context
`ExtractionGrades` is the verbalization-fidelity aggregate produced by `grade_extraction`. It carried a `perplexity: float | None` field plus a `perplexity_provider: PerplexityProvider | None` keyword argument that, when supplied, would call `provider.compute(narrative_text)` and stash the result on the aggregate. The same whole-text perplexity is already exposed by `grade_narrativity` as `NarrativityGrades.ppl_ordered`. Design.md section 10 lists perplexity under narrativity, and ADR 0008 explicitly made `NarrativityGrades` and `ExtractionGrades` orthogonal: different inputs (extraction + request vs. raw text), different cost profiles (no provider calls vs. O(N) provider calls), different paper origins (Ichmoukhamedov et al. vs. Cedro & Martens 2026). The `perplexity` field on `ExtractionGrades` was a leftover that crossed the boundary in the wrong direction.

## Decision
Drop the `perplexity` field from `ExtractionGrades` and the `perplexity_provider` keyword argument from `grade_extraction`. `ExtractionGrades` now carries only verbalization-fidelity metrics: `sign_faithfulness`, `value_faithfulness`, `rank_correlation`, `coverage`, `hallucination_count`, `readability`, and `prompt_version`. Callers who want a whole-text perplexity reading already get it from `grade_narrativity(text, provider).ppl_ordered`. The `PerplexityProvider` Protocol, `OpenAICompatibleEchoProvider`, `HuggingFacePerplexityProvider`, `DisabledProvider`, and `cumulative_perplexity` are untouched - they remain the perplexity surface for narrativity (ADR 0008, ADR 0009).

## Rationale
- Realigns the code with ADR 0008's orthogonality: narrativity-shaped concepts belong on `NarrativityGrades`, not on the fidelity aggregate.
- Removes a duplicate computation surface. The same whole-text perplexity was reachable from two places with the same provider call shape; the narrativity path is the one design.md section 10 names.
- Shrinks the `grade_extraction` signature. The provider parameter encouraged callers to mix the two scoring axes in a single call when ADR 0008 prescribed two distinct orchestrators.
- Pre-1.0 hard cutover (no shim, no deprecation cycle) matches the precedent set by ADR 0009 (`APIPerplexityProvider` removal) and ADR 0014 (`score_*` to `grade_*` rename).

## Consequences
- `ExtractionGrades` becomes seven fields instead of eight. `extra='forbid'` on the model rejects any caller passing `perplexity=...` to the constructor.
- `grade_extraction(extraction, request, schema, narrative_text, k=10)` is the new signature. Callers passing `perplexity_provider=...` raise `TypeError` at call time.
- The README sample output for `ExtractionGrades` no longer shows a `perplexity` line.
- The quickstart notebook constructs the `OpenAICompatibleEchoProvider` in Step 6 (alongside `grade_extraction`) but only passes it to `grade_narrativity` in Step 7. The provider instantiation stays in Step 6 so the same instance is reused for narrativity; the cell outputs regenerate on next live execution.
- The two grader unit tests that exercised the provider-call shape are removed; two new tests pin the negative contracts (rejected field, rejected kwarg).

## Rejected alternatives
- **Keep `perplexity` on `ExtractionGrades` and have `grade_narrativity.ppl_ordered` simply duplicate it.** Rejected: duplication is the problem, not the fix. ADR 0008 already located perplexity on the narrativity side.
- **Deprecation cycle (warn for one release, then remove).** Rejected: pre-1.0, zero external users on PyPI under the `xains` name, no compatibility cost to pay. The ADR 0009 / 0014 precedent is hard cutover.
- **Move the `perplexity` field to a new "extras" model carried alongside `ExtractionGrades`.** Rejected: adds a third aggregate for a metric the second already owns.

## References
- ADR 0008 - narrativity metrics: `NarrativityGrades` is orthogonal to `ExtractionGrades`.
- ADR 0009 - perplexity providers (kept; this ADR does not touch the provider surface).
- ADR 0014 - rename `score_*` to `grade_*` (precedent for pre-1.0 hard cutover).
- `docs/design.md` section 10 - perplexity is a narrativity axis.
- `src/xains/metrics/grader.py` - the field and parameter removal.
