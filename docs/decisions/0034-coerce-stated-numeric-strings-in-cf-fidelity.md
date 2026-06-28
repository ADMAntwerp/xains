# 0034. Coerce stated numeric strings in counterfactual fidelity

Date: 2026-06-25
Status: Accepted

## Context
`change_fidelity` (ADR 0032) scores a counterfactual narrative by comparing each extracted claim's stated before/after values against the ground-truth counterfactual. The comparison runs through `_value_matches`, whose numeric branch guarded both sides with `_is_numeric` and returned incorrect when either was not a real `int`/`float`. The extraction step is an LLM call that returns claims as JSON. In practice the judge LLM frequently returns a numeric feature's value as a string (`"3190"`) rather than a number (`3190`), even when the narrative stated the value correctly. Under the old guard that string failed `_is_numeric` and the claim scored incorrect, so a faithful narrative lost fidelity purely over the extractor's JSON typing. Observed on real DiCE-sourced narratives: a narrative that correctly stated `credit_amount` going from 3190 to 3757 scored `change_fidelity=0.5` because the extracted values arrived as strings. The existing type-guard tests pinned a different, still-valid case (a non-numeric word like `"low"` is incorrect); they did not cover numeric strings.

## Decision
The numeric branch of `_value_matches` coerces the stated side before comparison via a new `_coerce_stated_number` helper. A stated value that is a real number, or a string that parses as a float (whitespace tolerated), is coerced to that number and compared with `math.isclose`. A stated value that is a bool, `None`, or a non-numeric string yields `None` from the helper and scores incorrect, with no exception. Only the stated side is coerced. The ground-truth side keeps the strict `_is_numeric` guard, because ground truth comes from `build_scenarios` off the actual feature values and is trusted to be correctly typed; a string on the ground side is a malformed scenario and correctly scores incorrect.

## Rationale
- The string typing is an extraction artifact, not a fidelity failure. A narrative that states the right number should score correct regardless of whether the judge LLM serialized it as `3190` or `"3190"`. Coercing the stated side fixes the comparison on value, which is what the metric is meant to measure.
- Coercing only the stated side keeps the ground-truth invariant honest. Ground truth is library-produced and typed; loosening its guard would hide real malformation. The asymmetry encodes "trust our own ground truth, tolerate the LLM's typing".
- Bools and non-numeric strings remain incorrect. `_coerce_stated_number` excludes bool explicitly (booleans are not domain numbers) and returns `None` for unparseable strings, so the previously-pinned cases (`"low"`, `True`/`False` on a numeric feature) keep scoring incorrect. The change is additive: it only newly accepts numeric strings.
- The fix lives at the comparison layer, not the prompt, so it is deterministic. It does not depend on coaxing the LLM to emit numbers as numbers; however the extractor types a number, the comparison handles it.

## Consequences
- A numeric claim whose stated value is a numeric string now scores correct on that value. Real narratives that previously lost fidelity to JSON typing now score on substance.
- `_value_matches`'s numeric branch and docstring change; the ground-side guard and the categorical/ordinal/boolean equality branch are unchanged.
- Four new tests pin the behavior: numeric string correct, whitespace-padded numeric string correct, wrong numeric string still incorrect, non-numeric string still incorrect. The existing `"low"` and bool type-guard tests pass unchanged.
- This addresses only numeric typing. Categorical surface-form mismatch (a narrative saying "under 100" where the canonical label is `<100`) is a separate concern handled at the extraction prompt (ADR 0035).

## Rejected alternatives
- **Coerce both sides.** Rejected: the ground side is trusted-typed; coercing it would mask a malformed scenario (a string where a number belongs) that should surface as incorrect.
- **Fix it in the extraction prompt only (instruct the LLM to return numbers as numbers).** Rejected as the sole fix: it is probabilistic and LLM-dependent. The comparison-layer coercion is deterministic. The prompt-side improvements are pursued separately and do not remove the need for a robust comparison.
- **Relax `_is_numeric` globally to accept numeric strings.** Rejected: `_is_numeric` is shared with the feature-importance path; widening its meaning everywhere is broader than this fix needs. The coercion is local to the counterfactual stated-value comparison.
- **Leave the strict guard and accept low scores.** Rejected: it makes `change_fidelity` misleading on faithful narratives, which defeats the metric's purpose.

## References
- ADR 0032 - counterfactual fidelity scoring (the metric and `_value_matches`).
- ADR 0035 - categorical canonicalization at extraction (the sibling fix for category surface-form).
- `src/xains/metrics/counterfactual_fidelity.py` - `_coerce_stated_number` and the numeric branch.
- `tests/unit/test_metrics_counterfactual_fidelity.py` - the four pinning tests.
