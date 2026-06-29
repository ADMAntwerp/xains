# 0035. Categorical canonicalization at counterfactual extraction

Date: 2026-06-25
Status: Accepted

## Context
`change_fidelity` (ADR 0032) compares each extracted claim's stated before/after value against the ground-truth counterfactual. For categorical and ordinal features the comparison is string equality against the schema's canonical category label (for example `<100`, `>=200`, `no checking`). The extraction step is an LLM call. Its prompt, built by `_build_cf_user_prompt`, listed each feature as `name (dtype): description` but never showed the feature's category vocabulary, even though every categorical and ordinal `FeatureSchema` carries a `categories` list. With the canonical labels absent from the prompt, the judge LLM transcribed whatever surface phrasing the narrative used. A faithful narrative that said a balance moves to "200 or above" or savings to "between 100 and 500" was extracted verbatim as those phrases, which then failed string equality against `>=200` and `100<=X<500`. The numeric-typing half of this problem was fixed in ADR 0034; this ADR addresses the categorical surface-form half.

## Decision
`_build_cf_user_prompt` now surfaces each feature's allowed values and instructs the judge to canonicalize. The per-feature line appends ` | allowed values: <comma-separated categories>` whenever the feature carries a non-empty `categories` list (categorical and ordinal). The task block gains an instruction: when a feature lists allowed values, its `before` and `after` must each be one of those exact allowed values; the judge maps the narrative's wording to the closest allowed value, and records null when the wording matches no allowed value. The instruction maps the narrative's own words to the schema vocabulary; it does not reveal which value is the counterfactual's target, so it does not leak ground truth into the extraction (the CF extractor is still not shown the counterfactual, per ADR 0032). Only the counterfactual extraction prompt changes; the feature-importance extraction prompt (`_build_user_prompt`) is untouched.

## Rationale
- The judge cannot canonicalize to a vocabulary it never sees. The `categories` list was already on the schema and already passed to the builder; surfacing it is the minimal change that makes canonicalization possible.
- Canonicalization belongs at extraction, where the narrative's words and the schema vocabulary meet. The grader compares exact labels; making the extractor emit canonical labels keeps the grader simple and the comparison honest, rather than teaching the grader fuzzy categorical matching.
- Mapping the narrative's wording to the closest allowed value preserves the no-leak property. The extractor resolves the narrative's own phrasing; it is not told the counterfactual's value. The allowed-value list is public vocabulary, not the answer.
- The change is scoped to the counterfactual path. The feature-importance extractor has a different task shape and is not affected; duplicating the loop body there would be out of scope.

## Consequences
- A categorical or ordinal feature renders in the CF extraction prompt as `name (dtype): description | allowed values: a, b, c`. Numeric and other feature lines are unchanged (no suffix).
- The judge is instructed to emit canonical category labels, so a faithful narrative that paraphrases a category ("under 100") is extracted as the schema label (`<100`) and matches ground truth.
- Two new tests pin the behavior: the prompt surfaces a categorical feature's allowed values and the canonicalization instruction; a numeric-only schema carries no per-feature allowed-values suffix. The existing mock-response extraction tests are unaffected (they do not assert prompt text).
- Canonicalization is LLM-mediated, so it is best-effort, not guaranteed. The deterministic numeric-string fix (ADR 0034) and this prompt-side categorical fix are complementary: the first is exact, the second improves the common categorical case.

## Rejected alternatives
- **Teach the grader fuzzy categorical matching.** Rejected: it would move semantic judgment into the metric, making it opaque and harder to trust. Canonical labels at extraction keep the grader a simple exact comparison.
- **Show the categories only, without the canonicalization instruction.** Rejected: data without instruction is decoration the LLM may ignore. The A/B check showed the instruction is what changes behavior.
- **Add canonicalization to the feature-importance extractor too.** Rejected: out of scope; the FI task shape differs and has no counterfactual before/after claims. Revisit only if an FI need appears.
- **Pass the counterfactual's target value to the extractor to canonicalize against.** Rejected: it would leak ground truth into the extraction and inflate the metric, violating ADR 0032's deliberate choice not to show the CF to the extractor.

## References
- ADR 0032 - counterfactual fidelity scoring; the extractor is deliberately not shown the counterfactual.
- ADR 0034 - numeric-string coercion in the comparison (the sibling fix for numeric typing).
- `src/xains/guardrails/extraction.py` - `_build_cf_user_prompt` surfaces categories and instructs canonicalization.
- `tests/unit/test_guardrails_extraction_counterfactual.py` - the two pinning tests.
