# 0003. Data model: pydantic v2 discriminated unions

Date: 2026-04-23
Status: Accepted

## Context

We have four modalities (tabular, text, image, graph) and three explanation
modes (factual, contrastive, counterfactual). Contributions, requests, and
counterfactuals all vary by modality.

Three options for the type hierarchy:

1. One mega-class with `Optional` fields for every modality.
2. Separate classes per modality, user picks the right one, static typing
   alone enforces correctness.
3. Separate classes per modality unified under a discriminated union —
   callers can pass *any* of them through a shared parameter and pydantic
   picks the right model at parse time.

## Decision

Option 3. Each polymorphic family (`Contribution`, `*ExplanationRequest`,
`*Counterfactual`) is a `typing.Annotated[Union[...], Field(discriminator=...)]`.

Discriminator fields:

- `Contribution.type` — `"tabular" | "token" | "region" | "node" | "edge"`
- `ExplanationRequest.modality` — `"tabular" | "text" | "image" | "graph"`
- `CounterfactualInstance.type` — same four as modality

Each request subclass narrows its `contributions` field to the compatible
contribution types (e.g. `TabularExplanationRequest.contributions:
list[TabularContribution]`).

## Consequences

- Round-trip serialization (JSON / dict) works uniformly across modalities.
- Validation is automatic and loud — wrong contribution type for a given
  modality fails at request construction, not at prompt-render time.
- Type checkers (mypy strict) handle discriminated unions well enough.
- Requires pydantic ≥ 2.0 and Python ≥ 3.11 (for good `typing.Annotated`
  UX). We already require both.

## Alternatives considered

- Option 1 (mega-class): hides invariants, produces bad error messages.
- Option 2 (separate classes, no union): works, but every caller who wants
  to accept "any request" has to write the union by hand, duplicating the
  discriminator logic unreliably.
