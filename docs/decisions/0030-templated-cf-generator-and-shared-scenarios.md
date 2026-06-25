# 0030. Templated counterfactual generator and shared scenario helper

Date: 2026-06-25
Status: Accepted

## Context
ADR 0028 added the `changed_features(factual, cf)` diff. ADR 0029 added `CounterfactualTabularPromptTemplate` (the LLM path), which derived per-counterfactual scenario data inline: validating the factual and counterfactual `predicted_class` against `schema.target.classes`, mapping both to labels, calling `changed_features`, and checking each changed name against `schema.features`. The feature-importance path has two generators (ADR 0018's `NarrativeGenerator` base, with `LLMNarrativeGenerator` and the LLM-free `TemplatedNarrativeGenerator` from ADR 0019); the counterfactual path had only the LLM prompt template. Two gaps remained: there was no LLM-free counterfactual generator, and there was no end-to-end test exercising `mode="counterfactual"` through `Explainer.explain()`. A templated counterfactual generator needs the same scenario derivation the prompt template already did, so duplicating that logic would invite drift.

## Decision
Introduce three things in one commit. First, a shared scenario helper: `xains.counterfactuals.build_scenarios(request, schema) -> list[CounterfactualScenario]`, plus a `CounterfactualScenario` pydantic model (`ConfigDict(extra="forbid")`) with `factual_label`, `cf_label`, `changes: list[ChangedFeature]`, `method: str | None`, and 1-based `index`. It iterates `request.counterfactuals` in order (no ranking, ADR 0004), validates both predicted classes against `schema.target.classes`, maps them to labels, calls `changed_features`, and validates each changed name against `schema.features`. It decides neither numbering nor prose; those are each consumer's surface choice. Second, `CounterfactualTabularPromptTemplate.render` is refactored to consume `build_scenarios` instead of deriving scenarios inline; the rendered output is unchanged (the 16 existing template tests pass unmodified). Third, `xains.generation.TemplatedCounterfactualGenerator`, an LLM-free `NarrativeGenerator` mirroring `TemplatedNarrativeGenerator`, consumes `build_scenarios` and renders counterfactual-specific prose: one sentence per scenario, the first leading with the flip ("To change the prediction from <factual> to <cf>, <feature> would need to change from <before> to <after>"), multiple changes joined with an Oxford comma, multiple scenarios joined with "Alternatively, " (flip implicit on subsequent), an empty-change scenario rendered as "no feature changes were detected". `GenerationResult.text` and `latency_ms` are set; the LLM-only fields (`prompt`, `model_name`, `raw_llm_response`) are `None`. Two end-to-end tests run `mode="counterfactual"` through `Explainer.explain()`: one via the templated generator (deterministic exact-string prose, no LLM), one via `LLMNarrativeGenerator` with a `MockLLMProvider` and the CF prompt template (asserts the canned text returns and the rendered prompt carries the flip and changed-feature lines).

## Rationale
- One scenario helper as the single source of truth means the validation, label mapping, and diff happen in exactly one place. If any of that logic changes, both the prompt template and the templated generator follow automatically; they cannot drift.
- The prompt template refactor is output-preserving by contract: the existing tests are the regression guard, and they pass unmodified, so the move from inline derivation to `build_scenarios` changed structure without changing behavior.
- A templated (LLM-free) counterfactual generator satisfies the same need the feature-importance path met with `TemplatedNarrativeGenerator`: deterministic, dependency-free narratives for testing, offline use, and paper-baseline reproduction. It also makes the generator base honor the CLAUDE.md two-implementations rule on the counterfactual side.
- Counterfactual-specific prose (not the prompt template's instructional block) because the templated generator's output is read by a human, not consumed by an LLM. The flip-led sentence answers the question a counterfactual poses; the block format would read as a machine spec.
- Prose carries no units (conversational), while the prompt template's block does. The two surfaces serve different readers; this asymmetry is intentional.
- The two end-to-end tests close the coverage gap: before this commit, no test ran the counterfactual mode through the full Explainer. The templated path gives a deterministic assertion; the mock-LLM path proves the prompt template wires through the generator into a result.

## Consequences
- New module `src/xains/counterfactuals/scenarios.py` (`CounterfactualScenario`, `build_scenarios`), exported from `xains.counterfactuals`.
- New module `src/xains/generation/templated_counterfactual.py` (`TemplatedCounterfactualGenerator`), exported from `xains.generation` and re-exported top-level from `xains`.
- `src/xains/prompts/counterfactual_tabular.py` refactored to call `build_scenarios`; behavior unchanged, validated by the unmodified template tests.
- New tests: `test_counterfactuals_scenarios.py`, `test_generation_templated_counterfactual.py`, and two end-to-end cases appended to `test_explainer.py`.
- The counterfactual path now has both generator kinds, matching the feature-importance path's shape.
- Counterfactual fidelity scoring (per-feature value and sign correctness, invented features, categorical change handling) is a separate later commit; this commit covers generation only.
- Text, image, and graph counterfactual scenarios are out of scope; each modality's notion of "change" differs and gets its own decision when shipped.

## Rejected alternatives
- **Let the templated generator call `changed_features` directly and re-derive labels and validation itself.** Rejected: duplicates the label mapping, class validation, and feature checks in two places, which is the drift the shared helper exists to prevent.
- **Leave the prompt template untouched and accept duplicated derivation.** Rejected: the same reason; the small refactor (guarded by passing tests) is cheaper than two copies of the scenario logic aging independently.
- **Reuse the prompt template's instructional block as the templated generator's prose.** Rejected: the block is a machine spec for an LLM, not reader-facing prose. A human-facing generator should produce sentences, not a labeled scenario block.
- **Include units in the templated prose.** Rejected for now: conversational prose reads better without them, and the prompt template already carries units for the LLM. This is a reversible surface choice, not a contract.
- **Restate the flip target on every "Alternatively" scenario.** Rejected as the default: for the common binary case the flip is the same across scenarios, so repeating it is noise. The implicit form is the default; a future change can make it conditional if multi-target counterfactual sets prove common.
- **Skip the end-to-end tests and rely on the unit tests.** Rejected: the unit tests cover the helper, the template render, and the generator in isolation, but nothing exercised `mode="counterfactual"` through the actual Explainer. The two e2e tests close that gap.

## References
- ADR 0001 - scope boundary (pydantic-only core; tabular-first).
- ADR 0004 - counterfactual payload shape (list, order preserved, no ranking).
- ADR 0018 - narrative-generator abstraction (the base both generators share).
- ADR 0019 - templated narrative generator (the feature-importance LLM-free analogue this mirrors).
- ADR 0028 - `changed_features` diff (the per-feature change source).
- ADR 0029 - counterfactual tabular prompt template (refactored here to use the shared helper).
- `src/xains/counterfactuals/scenarios.py` - the shared helper.
- `src/xains/generation/templated_counterfactual.py` - the templated generator.
- `tests/unit/test_explainer.py` - the two end-to-end counterfactual cases.
