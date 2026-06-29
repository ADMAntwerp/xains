# 0036. Counterfactual generation prompt states before and after values

Date: 2026-06-25
Status: Accepted

## Context
`change_fidelity` (ADR 0032) scores a counterfactual narrative by checking, per changed feature, that the narrative states both the before value and the after value matching the ground-truth counterfactual. The default counterfactual system prompt (`DEFAULT_SYSTEM_TEMPLATE` in `counterfactual_tabular.py`) asked the LLM to describe the counterfactual scenario but did not ask it to state each feature's starting value. The shared narrative rules in `DEFAULT_NARRATIVE_RULES` push for fluent, lexically diverse, non-list prose, which led the LLM to write forward-only sentences ("increase your savings to the higher band") that named the target state but omitted the origin, and to paraphrase exact values. With the before value absent, the strict before-and-after fidelity check scored the claim incorrect even when the narrative was otherwise faithful. Diagnosis across real DiCE-sourced narratives showed before values were frequently unstated, which the extraction then recorded as null, which then failed the metric. This is the generation-side half of making fidelity reflect narrative quality; ADR 0034 (numeric typing) and ADR 0035 (categorical canonicalization) addressed the extraction and comparison sides.

## Decision
Add one sentence to `DEFAULT_SYSTEM_TEMPLATE`: for each feature that changes, state explicitly the value it changes from and the value it changes to, using those values exactly as written rather than rephrasing them. The sentence is placed after the scenario description and before the audience and tone line. The shared `DEFAULT_NARRATIVE_RULES` are unchanged, and the feature-importance system template is unchanged. The counterfactual user template, which already injects the canonical before and after values in the rendered scenario block, is unchanged: the generator already has the values, the new instruction tells the LLM to state both rather than only the target.

## Rationale
- The fidelity metric scores both before and after, so the narrative must state both. The previous prompt left the before value to chance, which made an otherwise faithful narrative score low for omitting the origin.
- Asking for the values "as written" nudges the LLM toward the canonical labels the user template already supplies, which complements the extraction-side canonicalization (ADR 0035) rather than relying on it alone.
- The instruction lives in the counterfactual system template only, so it does not change feature-importance narratives, which have no before/after structure.
- The sentence is a plain instruction, not a rigid format. It asks for explicit values while leaving the surrounding prose fluent, so it coexists with the narrative rules' fluency guidance rather than forcing a list.

## Consequences
- Default counterfactual narratives now state each changed feature's from-value and to-value, so the fidelity metrics score them on substance rather than penalizing omitted before-values.
- `DEFAULT_SYSTEM_TEMPLATE` gains one sentence; a user passing a custom `system_template` is unaffected and opts out by construction.
- One new test pins that the rendered default system prompt contains the from/to and "exactly as written" instruction. Existing prompt-template tests, which assert the user-prompt scenario block and custom-template behavior, are unaffected.
- The instruction is LLM-mediated guidance, not a guarantee. It improves the common case; the deterministic numeric coercion (ADR 0034) and the extraction-side categorical canonicalization (ADR 0035) remain the robustness layers when the LLM still paraphrases.

## Rejected alternatives
- **Loosen the metric to ignore the before value (after-only scoring).** Rejected: the before value is part of a counterfactual's meaning, and a metric that ignores it would accept a narrative that misstates the origin. The generation prompt is the right place to ensure the before value is present.
- **Add a strict "verbatim, do not round, do not rename" directive with examples.** Rejected for the default: it risks robotic, list-like output that fights the narrative rules' fluency goal. The chosen wording asks for explicit values without dictating format. An A/B check showed a softer "describe both values" wording backfired (it invited more paraphrase), so the wording settled on stating from/to values as written.
- **Put the instruction in the shared narrative rules.** Rejected: the rules are shared with the feature-importance path, which has no before/after counterfactual structure. The instruction belongs in the counterfactual system template.
- **Rely solely on extraction canonicalization (ADR 0035) and leave generation unchanged.** Rejected: canonicalization cannot recover a before value the narrative never stated. Generation must produce it.

## References
- ADR 0032 - counterfactual fidelity scoring (before and after both required).
- ADR 0034 - numeric-string coercion in the comparison.
- ADR 0035 - categorical canonicalization at extraction.
- `src/xains/prompts/counterfactual_tabular.py` - the amended `DEFAULT_SYSTEM_TEMPLATE`.
- `tests/unit/test_prompts_counterfactual_tabular.py` - the pinning test.
