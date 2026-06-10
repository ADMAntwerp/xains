# 0016. Rename `factual` mode to `feature_importance`

Date: 2026-06-10
Status: Accepted

## Context

ADR 0012 finalized the mode vocabulary as `"factual"`,
`"counterfactual"`, `"factual_counterfactual"`. In subsequent use the
`"factual"` name kept generating low-grade friction, because in the
counterfactual-explanation literature **"factual" denotes the actual
input datapoint** — the instance being explained — not an explanation
style. The mode named `"factual"` actually means "verbalize the
prediction via feature-importance contributions"; the CF-literature
sense of "factual" appears elsewhere in the code, e.g.
`Explainer._warn_if_counterfactual_does_not_flip` uses `factual_class`
to refer to the input's predicted class versus the counterfactual's.

Two distinct senses, one word. This ADR separates them.

## Decision

Rename, as a breaking pre-1.0 cutover (no shim, no deprecation):

| Old | New |
| --- | --- |
| `"factual"` (mode value) | `"feature_importance"` |
| `"factual_counterfactual"` (mode value) | `"feature_importance_counterfactual"` |
| `"counterfactual"` (mode value) | **unchanged** |
| `FactualTabularPromptTemplate` (class) | `FeatureImportanceTabularPromptTemplate` |
| `src/xainarratives/prompts/factual_tabular.py` (module) | `.../feature_importance_tabular.py` |
| `tests/unit/test_prompts_factual_tabular.py` | `.../test_prompts_feature_importance_tabular.py` |

The CF-literature sense of "factual" — the input datapoint — is
**deliberately retained** in `Explainer._warn_if_counterfactual_does_not_flip`
(the `factual_class` local var and the warning prose `"as the factual"`).
That code uses "factual" correctly in its CF-literature meaning, and the
rename of the mode-vocabulary use of "factual" frees the word to mean
only that.

## Rationale

- **Semantic accuracy.** The mode is named after what it produces (a
  narrative built from feature-importance contributions), not after a
  CF-literature term that collides with it. After this rename,
  "factual" in the codebase means only "the input datapoint" — its
  CF-literature sense — with no remaining collision.
- **Renamed `factual_counterfactual` too, for coherence.** Leaving it
  alone would have produced a vocabulary
  (`feature_importance` / `counterfactual` / `factual_counterfactual`)
  with the old mode name still embedded in the combined mode — the
  rename's whole point would be lost. The new
  `feature_importance_counterfactual` reads as "feature-importance
  contributions woven with counterfactual reasoning," which is exactly
  what the combined mode does.
- **Parallel structure over a vaguer name.**
  `feature_importance_counterfactual` pairs syntactically with
  `feature_importance` (FI-only) and `counterfactual` (CF-only). Names
  like `combined` or `hybrid` would hide that structure and force
  readers back to the docs.
- **Preserving `factual_class` is the point, not an accident.** The
  warning code uses "factual" correctly in the CF-literature sense (the
  input vs the counterfactual). The rename frees that sense to stand on
  its own. Renaming `factual_class` would re-collide the two meanings in
  the opposite direction.

## Consequences

- Breaking. `ExplanationConfig(mode="factual")` and
  `mode="factual_counterfactual"` both now raise `ValidationError` (the
  `ExplanationMode` `Literal` rejects them). Callers must use
  `mode="feature_importance"` or `mode="feature_importance_counterfactual"`.
- Breaking. The class `FactualTabularPromptTemplate` no longer exists.
  Import `FeatureImportanceTabularPromptTemplate` from the same package
  path (`xainarratives.prompts`). The submodule path also changed:
  `xainarratives.prompts.factual_tabular` →
  `xainarratives.prompts.feature_importance_tabular`.
- The internal CF-literature use of "factual" in
  `_warn_if_counterfactual_does_not_flip` (`factual_class` local var,
  `"as the factual"` warning text) is **retained intentionally** — see
  Rationale. Any future reviewer who wants to rename it should re-read
  this ADR.
- Notebook quickstart and README updated to demonstrate the new mode
  names and class name.
- Documented in CHANGELOG under `### Changed (BREAKING)`.

## Rejected alternatives

- **Leave the names as-is.** The ambiguity is real and was a continuous
  source of friction; the rename costs little (no shipped users yet) and
  pays back permanently.
- **Rename only `"factual"` → `"feature_importance"`; leave
  `"factual_counterfactual"` alone.** Would produce a half-renamed
  vocabulary that still embeds the old name in the combined-mode
  identifier. The point of the rename — eliminating the mode-vs-datapoint
  collision — would be defeated.
- **Use `"attribution_based"` / `"attribution_counterfactual"`.**
  "Attribution" is the upstream CF tool's output (SHAP, LIME, Captum,
  etc.). This library doesn't compute attributions and shouldn't pretend
  to; naming the mode "attribution_*" overclaims. `feature_importance`
  describes what the prompt template renders to the LLM (a ranked list of
  feature contributions) without overclaiming.
- **Use `"evidence"` / `"evidence_counterfactual"`.** Too vague —
  counterfactual reasoning is also a kind of evidence; the name doesn't
  disambiguate.
- **Rename `factual_class` and the warning prose in
  `_warn_if_counterfactual_does_not_flip` too.** Rejected: that code uses
  "factual" in its correct CF-literature sense (the datapoint), which is
  exactly the meaning this rename frees up. Renaming it would re-collide
  the two senses.

## References

- ADR 0003 (data model) — first introduced the vocabulary that included
  `factual`. Preserved as historical record.
- ADR 0012 (mode vocabulary) — finalized
  `factual / counterfactual / factual_counterfactual`. This ADR
  supersedes the naming choice; ADR 0012 stays as historical record of
  the structural decision (required field, three explicit values).
- `Explainer._warn_if_counterfactual_does_not_flip`
  (`src/xainarratives/explainer.py`) — preserves "factual" in the
  CF-literature sense; serves as the canonical reference point for that
  meaning going forward.
