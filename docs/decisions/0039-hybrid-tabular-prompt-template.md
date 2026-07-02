# 0039. Hybrid tabular prompt template

Date: 2026-06-25
Status: Accepted

## Context
The `feature_importance_counterfactual` mode has a deterministic templated generator (ADR 0037) but no LLM prompt template, so the LLM path for the hybrid mode did not exist. A hybrid explanation answers two questions in one narrative: why the model predicted the factual outcome (feature importance) and what would change it (counterfactual). The two single-purpose prompt templates already exist, and ADR 0038 extracted their body-block rendering into shared helpers (`build_contribution_block`, `build_counterfactual_block`) precisely so a third template could compose both blocks without duplicating logic. The counterfactual generation prompt (ADR 0036) learned that a narrative must state each changed feature's before and after values explicitly or the fidelity metric scores it low; a hybrid narrative's counterfactual half has the same requirement.

## Decision
Add `HybridTabularPromptTemplate` in `src/xains/prompts/hybrid_tabular.py`, a self-contained `PromptTemplate` that renders one two-section prompt. Its `render()` composes the two shared helpers: `build_contribution_block(request.contributions, schema, config.top_k_features)` for the feature-importance block and `build_counterfactual_block(scenario, schema, self._include_method)` for the counterfactual block, where `scenario = build_scenarios(request, schema)`. Its `DEFAULT_SYSTEM_TEMPLATE` instructs a two-part narrative (first explain the prediction from the contributions, then describe the counterfactual) and carries the ADR 0036 before/after instruction scoped to the counterfactual half ("for each feature that changes in the counterfactual, state explicitly the value it changes from and the value it changes to"). Its `DEFAULT_USER_TEMPLATE` presents both blocks under labeled headings. `_BUILTIN_NAMES` is the union of the two templates' placeholder sets, adding both `contributions` and `counterfactual`. The `{prediction}` placeholder is the factual predicted label (`schema.target.classes[predicted_class]`), matching the feature-importance template; the counterfactual's target class appears only inside the counterfactual block via `scenario.cf_label`. The template guards that the request is tabular, that `request.counterfactual` is not None, that every contribution names a schema feature, and that the predicted class is in the schema; the counterfactual-side validation is delegated to `build_scenarios`. It follows the ADR 0017 editable-template contract (custom `system_template`/`user_template`, `extra_placeholders` with a conflict check) and the ADR 0029 export policy (the class is exported from `xains.prompts`; the default-template constants stay on the submodule). The `include_method` flag mirrors the counterfactual template and toggles the method suffix on the counterfactual block only.

## Rationale
- Composition via the shared helpers. The block-building was extracted in ADR 0038 for exactly this consumer; the hybrid template imports both helpers and renders each section, with no duplicated ordering or formatting logic. A change to either block's format flows through all three templates.
- The factual label is the prediction being explained. The narrative explains why the model predicted the factual outcome, then what would flip it, so `{prediction}` is the factual label, consistent with the feature-importance template. The counterfactual target surfaces only inside the counterfactual block, where it belongs.
- The before/after instruction is scoped to the counterfactual half. The feature-importance half has no before/after structure, so the instruction is worded "for each feature that changes in the counterfactual" to avoid confusing the contribution explanation. This carries the ADR 0036 fix into the hybrid so its counterfactual half scores on the fidelity metric rather than omitting before-values.
- Guards mirror both source templates. The hybrid needs the feature-importance guards (unknown-feature, predicted_class) and the counterfactual guard (counterfactual not None, plus the checks inside `build_scenarios`), because it renders both halves. Keeping them in `render()` matches where the two source templates put them and where the tests pin them.

## Consequences
- `mode="feature_importance_counterfactual"` now has an LLM prompt template, so the hybrid narrative can be produced by an LLM as well as by the templated generator (ADR 0037).
- The template composes the ADR 0038 helpers, so it inherits their behavior and any future fix to block formatting.
- 16 unit tests pin the composition (both blocks in the user prompt), the two-part and before/after system-prompt instructions, the factual-label placeholder, the counterfactual-scoped `include_method`, all four guards, and the editable-template contract.
- The next hybrid piece is dual extraction: a hybrid narrative carries both feature-importance claims and counterfactual claims, so the Explainer must run both extractors on the two-section text. That is a separate commit; until it lands, a hybrid narrative reaching extraction is handled by the feature-importance branch (ADR 0033).

## Rejected alternatives
- **Reuse one of the existing templates and append the other block by hand.** Rejected: each existing template renders a full `(system, user)` pair with its own single-purpose instructions; bending one to carry both blocks would fight its system prompt. A self-contained hybrid template with its own two-part instruction is clearer.
- **Omit the before/after instruction from the hybrid.** Rejected: the counterfactual half has the same failure mode ADR 0036 fixed; leaving it out would knowingly ship low-fidelity hybrid counterfactual narratives.
- **Use the counterfactual target as `{prediction}`.** Rejected: the narrative explains the factual prediction first; the factual label is the subject, and the counterfactual target is internal to the counterfactual block.
- **Export the default-template constants at the package level.** Rejected: the three templates' constants share names and would collide, the same reason ADR 0029 kept them submodule-only.

## References
- ADR 0017 - editable prompt templates (the contract this template follows).
- ADR 0029 - counterfactual tabular prompt template and the constant-export policy.
- ADR 0036 - counterfactual generation prompt states before and after values (carried into the hybrid).
- ADR 0037 - templated hybrid generator (the deterministic sibling of this template).
- ADR 0038 - shared prompt block builders (the helpers this template composes).
- `src/xains/prompts/hybrid_tabular.py` - the template.
- `tests/unit/test_prompts_hybrid_tabular.py` - the 16 pinning tests.
