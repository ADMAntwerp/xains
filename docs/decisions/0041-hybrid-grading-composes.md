# 0041. Hybrid grading composes the FI and CF graders

Date: 2026-06-25
Status: Accepted

## Context
ADR 0032 introduced counterfactual grading (`grade_counterfactual` -> `CounterfactualGrades`) alongside the existing feature-importance grading (`grade_extraction` -> `ExtractionGrades`), and deferred one question: whether the grade aggregates should share an abstract base. It deferred deliberately, stating that the shared shape should be factored from three real cases rather than guessed from two, and naming the hybrid `feature_importance_counterfactual` mode as the third case that would reveal the right shape; it also noted the hybrid grading might compose the two graders rather than inherit a base. The hybrid mode now exists end to end: a templated generator (ADR 0037), an LLM prompt template (ADR 0039), and dual extraction that populates both `narrative_extraction` and `counterfactual_extraction`, possibly partially (ADR 0040). What remained was grading, and with the third concrete case in hand the deferred question is answerable.

## Decision
Hybrid grading composes; it does not inherit. Add `HybridGrades`, a pydantic model with `model_config = ConfigDict(extra="forbid")` (matching the other two grade models) that holds two optional fields, `feature_importance: ExtractionGrades | None` and `counterfactual: CounterfactualGrades | None`. Add `grade_hybrid(narrative_extraction, counterfactual_extraction, request, schema, narrative_text, k=10) -> HybridGrades`, which calls `grade_extraction(narrative_extraction, request, schema, narrative_text, k=k)` when the feature-importance extraction is present and `grade_counterfactual(counterfactual_extraction, request, schema)` when the counterfactual extraction is present, and returns a `HybridGrades` holding whichever sub-grades were produced. When an extraction is None the corresponding sub-grade is None; when both are None the function returns an empty `HybridGrades` without raising. No abstract grades base is introduced. `render_grades` is not changed: it already accepts `extraction=` and `counterfactual=` independently and emits both sections, so a `HybridGrades` renders by unpacking its two fields into that existing call, which the `grade_hybrid` docstring documents.

## Rationale
- The third concrete case argues for composition, not a base. The only surface shared across the three grade aggregates is a `coverage: float` and a `prompt_version: str`, and those very names collide with different meanings: feature-importance coverage and counterfactual coverage measure different quantities, and the two `prompt_version` values come from different prompts. A base that hoisted those fields would merge things that are not the same; a base that did not would be near-empty. Composition keeps each aggregate intact and unambiguous.
- Nesting resolves the field-name collision cleanly. Under `HybridGrades`, the two coverages are reachable as `feature_importance.coverage` and `counterfactual.coverage`, each unambiguous, where a flattened model could not carry both.
- Partial grading mirrors the extraction dispatch. ADR 0040 populates the two extractions independently, so a hybrid result can carry one extraction and not the other; grading must degrade the same way. Grading only the present half, and returning an empty aggregate when neither is present, keeps the grader as permissive as the extractor that feeds it, rather than raising on a result the pipeline is designed to produce.
- A named composed type and entry point give API symmetry without a premature abstraction. `grade_hybrid` sits beside `grade_extraction` and `grade_counterfactual` as a peer, so a hybrid user calls one grading function like the other two modes do. `HybridGrades` is a concrete leaf model, not a base class others implement, so it does not run afoul of the guidance that abstractions need two or more concrete implementations: it is composition, not an interface over multiple implementers.

## Consequences
- `mode="feature_importance_counterfactual"` now has a grading entry point returning both halves' grades in one object, completing the hybrid path's generate-extract-grade chain.
- `HybridGrades` and `grade_hybrid` are exported from `xains.metrics` and top-level `xains`, alphabetically alongside the existing grade symbols. `HybridGrades` needs no directions dict; it is a container, and its sub-grades carry their own direction constants.
- `render_grades` is unchanged; hybrid rendering is the documented unpack. This avoids a hybrid-specific render branch for a renderer that already composes.
- Eight unit tests pin the behavior: both-present composes to the standalone grades, `k` forwards to the feature-importance half, each single-present case leaves the other None, both-None yields an empty aggregate without raising, extra fields are rejected, and the unpack-to-render round trip emits both sections.
- The abstract-grades-base question ADR 0032 deferred is now closed: the answer is compose. If a future fourth case ever shows a genuine shared shape, a base can be revisited, but the three current cases do not motivate one.

## Rejected alternatives
- **Introduce an abstract grades base the three aggregates inherit.** Rejected: the shared surface is a colliding `coverage` and `prompt_version` with different meanings, so a base either merges unlike fields or is near-empty. Three concrete cases show no shape worth factoring.
- **Flatten the two aggregates into one hybrid model.** Rejected: the `coverage` and `prompt_version` name collisions cannot coexist in one flat model without renaming, which would obscure that each value belongs to a distinct half. Nesting keeps them separate and clear.
- **Skip a new type and have callers invoke both graders directly.** Rejected: it leaves the hybrid mode without the API symmetry the other two modes have, and without a named result the composition is an unwritten convention. A thin composed type and one entry point are cheap and discoverable.
- **Require both extractions and raise when one is None.** Rejected: dual extraction can legitimately populate one half and not the other (ADR 0040), so a grader that raised on the partial case would reject results the pipeline is designed to produce. Optional sub-grades match the extractor.

## References
- ADR 0032 - counterfactual fidelity scoring (deferred the abstract-grades-base question to this third case).
- ADR 0040 - hybrid dual extraction (the partial-population behavior grading mirrors).
- `src/xains/metrics/grader.py` - `HybridGrades` and `grade_hybrid`.
- `tests/unit/test_metrics_grade_hybrid.py` - the eight pinning tests.
