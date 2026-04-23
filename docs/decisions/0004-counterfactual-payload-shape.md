# 0004. Counterfactual payload shape: list of pre-computed instances

Date: 2026-04-23
Status: Accepted

## Context

Counterfactual-explanation methods (DiCE, Wachter, Alibi, Growing Spheres,
…) typically return **multiple** counterfactuals per query instance —
either because diversity was an explicit objective or because the user
wants to present alternatives.

Possible payload shapes:

1. Exactly one counterfactual per request.
2. A set, and the library ranks / selects which to verbalize.
3. A list of ≥ 1 counterfactuals; the library verbalizes them as provided.

## Decision

Option 3. `ExplanationRequest.counterfactuals: list[CounterfactualInstance]
| None`. Length ≥ 1 when present. Single-CF is the degenerate case.

The library does not rank, filter, or reorder the list — that is the
user's upstream decision. The default counterfactual prompt template
verbalizes them in the order provided.

## Consequences

- Matches how DiCE and similar tools actually produce output.
- Users who have one hand-crafted CF can still pass a one-element list.
- Prompt templates must handle "here are N ways the outcome could have
  been different" coherently. Acceptable complexity.
- The `changed_features` field on each tabular CF is computed by the
  library if omitted (simple diff against factual), with user override
  allowed.

## Alternatives considered

- Option 1: forces users with multiple CFs to make N separate library
  calls and stitch the result themselves.
- Option 2: pushes selection policy into the library, which needs inputs
  (mutability, diversity, distance metric choice) we do not have.
