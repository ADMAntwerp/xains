# 0013. Remove vestigial ExplanationConfig fields

Date: 2026-05-22
Status: Accepted

## Context

ExplanationConfig declared two boolean fields, `include_confidence` and
`include_caveats`, both defaulting to `True`. An audit
(`grep -rn "include_confidence\|include_caveats"` across `src/` and
`tests/`) found ZERO consumers — no prompt template, explainer step,
guardrail, metric, scorer, or test reads them. Users setting either
field expected behavior change; none occurred. The knobs misled callers
into believing they were controlling output that was, in fact, never
gated by either field.

## Decision

Remove both fields from `ExplanationConfig`. Because the model uses
`extra="forbid"`, passing `include_confidence=...` or
`include_caveats=...` now raises `ValidationError` — a deliberate
breakage to surface any hypothetical user code that was relying on the
no-op.

## Rationale

A dead knob is worse than a missing one. It signals "you can control
this" while controlling nothing. Removing keeps the public API surface
honest. If confidence-mentioning or caveat-gating behavior is needed in
the future, add it then with a name that says what it does (e.g.
`mention_probability`) and wire it to a real prompt-template branch —
don't carry a vestigial promise.

## Consequences

- Breaking: `ExplanationConfig(include_confidence=...)` now raises
  `ValidationError`. Same for `include_caveats`. Documented in CHANGELOG
  under `### Changed (BREAKING)`.
- The two fields are released from the library's surface; future API
  additions in this space start from a clean slate.

## Rejected alternatives

- **Wire them up.** Would need: design what "include confidence" means
  in prose (mention probabilities? "with X% confidence"?), thread it
  through every prompt template, test the behavior. Out of scope;
  speculative until a use case demands it.
- **Mark as deprecated, keep accepting them.** More complexity, no
  benefit — they did nothing, deprecating a no-op doesn't preserve any
  user-visible behavior.

## References

- Audit: `grep -rn "include_confidence\|include_caveats"` across `src/`
  and `tests/` returns only the two declaration lines in
  `src/xainarratives/config.py`.
- ADR 0012 (mode-vocabulary finalized) deferred a parallel
  `contrast_class` field cleanup to a follow-up; this ADR is the same
  spirit applied to a different vestigial field.
