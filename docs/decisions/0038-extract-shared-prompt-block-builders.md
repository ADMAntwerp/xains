# 0038. Extract shared prompt block builders

Date: 2026-06-25
Status: Accepted

## Context
Two tabular prompt templates, `FeatureImportanceTabularPromptTemplate` (ADR 0017) and `CounterfactualTabularPromptTemplate` (ADR 0029), independently built their body blocks inline inside `render()`. The feature-importance template contained the contribution ordering (abs(importance) descending with tie-widening at the k-th boundary) and per-line formatting; the counterfactual template contained the flip lead and change-lines formatting. Both blocks are pure functions of already-validated inputs, needing no LLM call and no external state, but the code was duplicated across the two templates. A third consumer, the forthcoming hybrid prompt template that renders both sections in one prompt, would either duplicate the logic a third time or import from one of the existing templates. Neither is acceptable: three copies drift, and cross-template imports at the render layer entangle files that should be siblings.

## Decision
Extract the two block builders into `xains.prompts._blocks` as pure string-returning helpers: `build_contribution_block(contributions, schema, top_k) -> str` (contribution ordering plus per-line formatting) and `build_counterfactual_block(scenario, schema, include_method) -> str` (flip lead plus change lines for a pre-built `CounterfactualScenario`). The helpers take pre-validated, pre-computed inputs. Guards stay in the templates' `render()` methods one layer above: the `isinstance(request, TabularExplanationRequest)` check, the unknown-feature check on `request.contributions`, the `predicted_class in schema.target.classes` check, and the `build_scenarios(request, schema)` call that itself validates factual/CF classes and changed-feature names. `build_counterfactual_block` receives the already-built `CounterfactualScenario` rather than calling `build_scenarios` itself, because `render()` needs the scenario for its own metadata (the factual label for the `{prediction}` placeholder). Byte-for-byte output preservation is proven by running the two existing template test suites unmodified after the refactor: all 49 tests pass without a single edit.

## Rationale
- DRY without touching the guard boundaries. The helpers are pure string builders; the templates keep every guard and every validator call. The extract line is drawn where the pure-function contract begins.
- The helpers take pre-built inputs, not the raw request. `build_contribution_block` receives the contribution list directly; `build_counterfactual_block` receives the already-built `CounterfactualScenario`. This keeps signatures narrow and lets the hybrid template compose the same helpers over the same input shapes without re-parsing the request.
- Existing template tests pass unmodified. The refactor is behavior-preserving by construction: the extracted code is textually equivalent to the inline code, only relocated. The 49 existing template tests plus 18 new `_blocks` unit tests jointly pin both the extracted contract and the end-to-end rendered output.
- This unblocks the forthcoming hybrid prompt template (ADR 0039). With the shared helpers, the hybrid template imports both from `xains.prompts._blocks` and composes them into a two-section prompt, instead of duplicating the logic or importing from a sibling template.

## Consequences
- New module `src/xains/prompts/_blocks.py` with two helpers. The underscore prefix marks it private-to-the-package: consumers outside `xains.prompts` should not import from it.
- `FeatureImportanceTabularPromptTemplate.render()` and `CounterfactualTabularPromptTemplate.render()` each shed roughly 15 lines of inline block logic in favor of a single helper call.
- 18 new unit tests in `tests/unit/test_prompts_blocks.py` pin the helpers' individual contracts (ordering, tie-widening, top-k cut, unit suffix, sign formatting, empty inputs, missing schema feature). The two existing template test suites remain unchanged and continue to pin end-to-end rendered output.
- Zero behavior change for existing users. `render()` returns identical `(system, user)` strings as before for every input.

## Rejected alternatives
- **Duplicate the block logic in the forthcoming hybrid template.** Rejected: three copies of the same ordering-and-formatting code, three places that must drift together whenever the wire format changes. Extracting once, now, is the least entropy.
- **Have the hybrid template instantiate and combine two full templates.** Rejected: each full template renders a complete `(system, user)` pair, so stitching two would concatenate two system prompts (contradictory instructions) or call one template only for its user block (an awkward API misuse). The block-level extraction targets the reusable unit: the body block, not the surrounding prompt.
- **Keep block-building inline and refactor later when the hybrid arrives.** Rejected: the hybrid template needs the shared helpers to exist for its `render()` to compose them. Deferring means landing the hybrid with duplicated logic or blocking it on a later refactor. Extract first, prove it byte-for-byte, land the hybrid on top.
- **Extract into a public `xains.prompts.blocks` module (no underscore prefix).** Rejected: the helpers are implementation glue for the templates, not a user-facing API. The private module keeps the top-level import list of `xains.prompts` focused on the classes users actually construct.

## References
- ADR 0017 - editable prompt templates (guards and substitution stay in render()).
- ADR 0029 - counterfactual tabular prompt template (one of the two callers).
- ADR 0037 - templated hybrid generator (the path that surfaced the duplication).
- ADR 0039 - hybrid prompt template (forward reference: the third caller these helpers enable).
- `src/xains/prompts/_blocks.py` - the two helpers.
- `src/xains/prompts/feature_importance_tabular.py` - now delegates block building.
- `src/xains/prompts/counterfactual_tabular.py` - now delegates block building.
- `tests/unit/test_prompts_blocks.py` - the helper tests.
