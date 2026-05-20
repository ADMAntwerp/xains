# 0012. Explanation-mode vocabulary finalized

Date: 2026-05-19
Status: Accepted. Supersedes the mode-vocabulary portion of ADR 0003.

## Context

ADR 0003 sketched a four-value mode vocabulary:
`factual`, `contrastive`, `counterfactual`, `auto`. The library inferred
the mode from request shape when `mode="auto"` was set: counterfactuals
present -> counterfactual; contrast_class set -> contrastive; otherwise
factual.

Two problems with that design surfaced once real callers exercised the API:

1. **Auto-resolution is implicit.** The user passes a request shape; the
   library infers intent. A missing field silently changes the resolved
   mode. Misconfiguration surfaces at runtime, not at config construction.
2. **"contrastive" had ambiguous semantics.** Was it "explain factual
   relative to a contrast class" or "weave factual and counterfactual
   together"? Going forward the latter is needed; the former has no
   concrete consumer.

## Decision

Three explanation modes, exactly:

- `factual` - narrative explains the actual prediction from
  feature-importance contributions only. No counterfactuals needed.
- `counterfactual` - narrative explains counterfactual(s) only.
  Request must carry counterfactuals; factual contributions are not
  verbalized.
- `factual_counterfactual` - narrative weaves both the factual
  contributions and the counterfactual(s). Request must carry both.

`auto` is removed. `ExplanationConfig.mode` has no default - it is a
required field. `contrastive` is renamed to `factual_counterfactual`
with the redefined semantics above.

Single canonical `ExplanationMode = Literal[...]` in `types.py`;
`config.py` imports it. The previous parallel `ExplanationModeOrAuto`
type is deleted.

`Explainer._resolve_mode` is renamed `_validate_mode`. It no longer
resolves anything - it validates the explicit mode against the request
and returns it. `Explainer.__init__`'s fallback when `config=None`
constructs `ExplanationConfig(mode="factual")` rather than relying on a
default.

## Rationale

- **Why no auto-resolution.** Inference conflates "what shape did the
  user pass?" with "what did the user want?". A user may pass
  counterfactuals as context for a factual explanation; auto would
  force counterfactual mode. Explicit mode lets the user state intent;
  the request shape just has to support it.
- **Why `factual_counterfactual` instead of `contrastive`.** Naming by
  composition makes the semantics self-documenting. "contrastive"
  needed a separate sentence to define; `factual_counterfactual` pairs
  with `counterfactual` (CF-only) and `factual` (factual-only)
  explicitly.
- **Why required.** With three values and no default, every config
  call site states intent. Previous sites that omitted `mode=` got
  "auto" - routinely not what the caller meant.
- **Why keep the Explainer config fallback.** Removing the fallback
  would force every `Explainer(...)` construction to build a full
  config. The PR's goal is "mode is explicit," not "all configuration
  is mandatory." The fallback supplies `mode="factual"` explicitly,
  which preserves the convenience without reintroducing inference.

## Consequences

- `ExplanationConfig(mode=...)` is required at every call site. Tests
  and the quickstart notebook updated.
- `Explainer._validate_mode` no longer resolves - it validates the
  explicit mode against the request, then returns.
- `ExplanationModeOrAuto` is deleted; `ExplanationMode` in `types.py`
  is the single source of truth.
- `ExplanationResult.mode` now uses the three-value vocabulary.
- `contrast_class` on `_BaseExplanationRequest` is orphaned after this
  PR. Its mode-resolution references in `explainer.py` are deleted, but
  the field itself stays on the data model; removal is deferred to a
  follow-up PR with its own ADR.
- ADR 0011 mentions "contrastive/counterfactual templates" in passing;
  noted here as mildly stale, but not amended (the configurable
  `narrative_rules` mechanism is unchanged).

## Rejected alternatives

- **Keep auto-resolution behind a strict-mode flag.** Adds a knob to
  control whether inference happens. More state, more confusion. The
  whole point is to remove implicit behavior.
- **Three modes but keep an optional default ("factual").** Half-
  measure: users still don't state intent for the common case.
  Explicit-everywhere is simpler.
- **Drop the Explainer config fallback entirely.** Overreach. The PR
  is about explicit mode, not mandatory config; removing the fallback
  conflates the two and degrades ergonomics for the common case.
- **Add `factual_contrastive` (factual + contrast class) alongside
  `factual_counterfactual`.** Speculative - no concrete use case.
  Defer until a contrast-class scenario emerges.

## References

- ADR 0003 (data-model; mode-vocabulary portion superseded).
- ADR 0011 (configurable narrative_rules).
