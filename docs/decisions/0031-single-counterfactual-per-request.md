# 0031. Single counterfactual per request

Date: 2026-06-25
Status: Accepted

## Context
ADR 0004 fixed the counterfactual payload as a list: `ExplanationRequest.counterfactuals: list[CounterfactualInstance] | None` with `min_length=1`, on the reasoning that tools like DiCE return multiple counterfactuals per instance and the library should verbalize them in order. The CF generation work (ADRs 0028 to 0030) built on that list: the prompt template numbered scenarios ("Scenario 1", "Scenario 2"), the templated generator joined them with "Alternatively, ", and the fidelity scorer unioned ground-truth values across scenarios. In practice a request carries one counterfactual explanation, which may itself require changing several features together to flip the prediction. The list semantics (a set of independent alternative counterfactuals) are a different thing from "one counterfactual that changes multiple features", and conflating them added numbering, alternation, and union machinery that the intended single-explanation use never exercises. Selecting among alternative counterfactuals is an upstream concern the caller resolves before calling the library.

## Decision
A request carries exactly one counterfactual, or none. Replace `_BaseExplanationRequest.counterfactuals: list[CounterfactualInstance] | None = Field(min_length=1)` with `counterfactual: CounterfactualInstance | None = None`. The single counterfactual may change any number of features; that is the multi-feature recipe, rendered as one flip block. This supersedes the list-of-instances decision in ADR 0004. Consequences across the CF stack: `build_scenarios(request, schema)` returns a single `CounterfactualScenario` (not a list) and the scenario's `index` field is removed; `CounterfactualTabularPromptTemplate` renders one flip block with no "Scenario N" numbering and its user-prompt placeholder is renamed `{counterfactuals}` to `{counterfactual}`; `TemplatedCounterfactualGenerator` renders one sentence (Oxford-comma joined across the changed features) with the "Alternatively, " alternation removed; `change_fidelity` and `cf_coverage` score against exactly one (before, after) pair per changed feature, dropping the cross-scenario union; `Explainer._validate_mode` / `_warn_if_counterfactual_does_not_flip` check the single `request.counterfactual`; the `from_feature_importance` adapter takes `counterfactual=` instead of `counterfactuals=`.

## Rationale
- The intended input is one counterfactual explanation. Modelling it as a list invited numbering, alternation, and union code that the common path never uses and that read as accidental complexity.
- "One counterfactual changing several features" and "several alternative counterfactuals" are distinct. The first is the supported case and is fully expressed by a single counterfactual whose feature set has more than one entry. The second is a selection problem the caller owns; the library should not present a menu of alternatives it cannot rank.
- A singular field is honest: the type says what is true. A length-one list that is never allowed to grow is a lie the reader has to discover.
- Removing the multi-CF surface simplifies three consumers (template, generator, fidelity) and removes a whole class of "which scenario does this claim belong to" ambiguity from scoring, since a flat extraction has no scenario index anyway.
- Pre-1.0 with no external users, the breaking rename is cheap now and progressively more expensive later.

## Consequences
- `types.py`: the field is `counterfactual: CounterfactualInstance | None = None`. Passing the old `counterfactuals=` raises (the model is `extra="forbid"`).
- `scenarios.py`: `build_scenarios` returns one `CounterfactualScenario`; the `index` field is gone.
- `counterfactual_tabular.py`: no scenario numbering; placeholder `{counterfactual}`; templates reworded to the singular.
- `templated_counterfactual.py`: one sentence; no "Alternatively, " path.
- `counterfactual_fidelity.py`: ground-truth map holds one (before, after) pair per feature; the union helper is gone.
- `explainer.py`, `integrations/feature_importance.py`: updated to the singular field and adapter keyword.
- Tests across the CF stack updated: multi-CF numbering, alternation, and union tests are removed (the features no longer exist); all single-CF behaviour is preserved with the singular field.
- ADR 0004 is superseded in the part that fixed the list shape; its other reasoning (the library does not search, rank, or filter counterfactuals) still holds for the single counterfactual.
- Future text / image / graph counterfactuals inherit the single-counterfactual shape.

## Rejected alternatives
- **Keep the list, constrain it to length one (`min_length=1, max_length=1`).** Rejected: the type would still read as a list and leave the numbering / alternation / union code in place as unreachable branches. A singular field is the honest model and lets the dead code be deleted.
- **Keep the list as ADR 0004 specified and just never pass more than one in practice.** Rejected: the library would carry multi-CF surface (numbering, alternation, union) that no supported use exercises, and the type would misrepresent the contract.
- **Support a true set of alternative counterfactuals with ranking.** Rejected: selecting and ordering alternatives needs inputs the library does not have (mutability, diversity, distance-metric choice), the same reasoning ADR 0004 used to refuse a selection policy. The caller resolves alternatives upstream and passes the one to verbalize.

## References
- ADR 0004 - counterfactual payload shape (superseded in part: the list decision).
- ADR 0028 - changed-features diff (one counterfactual's per-feature changes).
- ADR 0029 - counterfactual tabular prompt template (numbering removed here).
- ADR 0030 - templated CF generator and shared scenarios (alternation and list return removed here).
- `src/xains/types.py` - the singular `counterfactual` field.
- `src/xains/counterfactuals/scenarios.py` - single-scenario `build_scenarios`.
