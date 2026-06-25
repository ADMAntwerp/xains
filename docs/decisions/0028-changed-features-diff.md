# 0028. Changed-features diff for tabular counterfactuals

Date: 2026-06-25
Status: Accepted

## Context
ADR 0004 fixed the counterfactual payload shape: `ExplanationRequest.counterfactuals` is a list of pre-computed instances; for tabular CFs, `TabularCounterfactual.changed_features: list[str] | None` is an optional user-provided override and the library promised to "compute changed_features if omitted (simple diff against factual)". That computation has not existed in code yet. Upcoming commits introduce a counterfactual prompt template and an LLM-free templated CF narrative; both need the diff as their input. This commit delivers the diff alone, as a pure function with no LLM dependency, so the prompt-template and generator work can land on top of a tested foundation.

## Decision
Introduce `xains.counterfactuals.changed_features(factual: dict[str, Any], cf: TabularCounterfactual) -> list[ChangedFeature]`, a pure function plus a small pydantic model `ChangedFeature(name, before, after)` with `ConfigDict(extra="forbid")`. The function honors `cf.changed_features` when set (override mode, no value check) and otherwise iterates `cf.features` reporting only the keys whose values differ from the factual (diff mode). A key absent from the factual raises `ValueError` naming the offending feature in either mode. The factual instance is passed explicitly as `dict[str, Any]`, not derived from a request handle - callers compose by passing `request.features`. Scope is tabular only; text / image / graph CFs get their own decisions if and when they are addressed. The new subpackage lives at `src/xains/counterfactuals/` and exposes the function and model via `xains.counterfactuals.__init__` only (no top-level `xains.changed_features` re-export, matching how `from_feature_importance` is reachable only via `xains.integrations`).

## Rationale
- ADR 0004 already named the semantics; this commit just codifies them.
- A pure function over plain data is the smallest viable surface: no class to construct, no provider to inject, no implicit lookups. Easy to test in isolation, easy to call from the upcoming CF prompt template and the templated CF generator.
- `ChangedFeature` as a pydantic `BaseModel` with `extra="forbid"` matches the house style (every existing data type in `src/xains/types.py` and `src/xains/schema.py` uses that shape). A `dataclass` would be a one-off; a `NamedTuple` would lose pydantic's `ConfigDict` discipline and re-introduce the "validates differently from every other type" risk.
- Passing the factual explicitly (rather than recovering it from a `TabularExplanationRequest`) lets future callers diff against any factual source (tests, scripted tools, batch runs) without constructing a request. It also keeps the function single-purpose.
- Missing-in-factual as `ValueError` instead of a silent skip: a CF referencing a feature the factual does not carry is a data-pipeline bug; surfacing it loudly per CLAUDE.md's "fail loud on user error" stance prevents the downstream narrative from inventing context.
- No-change degenerate case (CF equals factual, no override): returns `[]`. The no-flip warning that already lives on `Explainer._warn_if_counterfactual_does_not_flip` is about `predicted_class`, not features; layering a second warning here would muddy the contract. Empty list is the honest answer for "no feature changes".

## Consequences
- New subpackage `src/xains/counterfactuals/` with `__init__.py` and `diff.py`.
- New public names: `xains.counterfactuals.changed_features` (function), `xains.counterfactuals.ChangedFeature` (pydantic model).
- Eight unit tests in `tests/unit/test_counterfactuals_diff.py` pin: full-CF diff, partial-CF diff, override-honored-without-value-check, missing-from-factual in diff mode raises, missing-from-factual in override mode raises, identical CF returns `[]`, value types preserved, model rejects extra fields.
- Future commits in this PR will add a `CounterfactualTabularPromptTemplate` and a templated CF generator on top of this function. They will not need to re-implement the diff.
- Text / image / graph counterfactuals are not addressed by this ADR; their own diffs (token-level, region-level, edge / node-level) are separate decisions when those modalities ship CF support.

## Rejected alternatives
- **`dataclass` for `ChangedFeature`.** Rejected: the house style is pydantic with `ConfigDict(extra="forbid")` across `Prediction`, `Contribution`, `Counterfactual`, etc. Mixing styles for one tiny type costs more than it saves.
- **Derive the factual implicitly from a `TabularExplanationRequest` argument.** Rejected: couples the function to a richer type than it needs and prevents callers from diffing against arbitrary factuals (tests, batch tools). `request.features` is one cheap explicit pass at the call site.
- **Silently skip a CF key that is absent from the factual.** Rejected: hides a data-pipeline bug. Narrative quality degrades silently; LLM-as-judge will not catch it because the resulting narrative is internally consistent. Loud `ValueError` matches the rest of the library's posture.
- **Warn (not raise) on identical-CF (no changes).** Rejected: the no-flip warning on `Explainer` already covers the predicted-class case; adding a second warning here would either duplicate or conflict. Empty list is the truthful answer.
- **Place the module under `xains.integrations.` or `xains.metrics.`.** Rejected: integrations is for input-shape adapters (`from_*`); metrics is for scoring. Counterfactual support is its own concern; a dedicated subpackage gives the upcoming template and generator a natural home.
- **Cover text / image / graph counterfactuals in the same function.** Rejected: each modality has a different notion of "change" (token-level, region-level, edge / node operations). Bundling them produces a function with four discriminator branches and no shared logic. Tabular alone first, separate ADRs for the rest.

## References
- ADR 0004 - counterfactual payload shape (the promise this commit cashes in).
- ADR 0001 - scope boundary / dependency discipline (no new runtime deps).
- CLAUDE.md - pydantic-first house style, surgical changes, fail loud on user error.
- `src/xains/types.py` - `TabularCounterfactual`, `TabularExplanationRequest`.
- `src/xains/counterfactuals/diff.py` - this commit's module.
- `tests/unit/test_counterfactuals_diff.py` - the eight pinning tests.
