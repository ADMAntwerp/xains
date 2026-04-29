# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While `0.y.z`, minor versions may contain breaking changes.

## [Unreleased]

### Added

- `xainarratives.metrics` subpackage: `sign_faithfulness`,
  `value_faithfulness`, `rank_correlation`, `coverage`,
  `hallucination_count`, `readability`. All pure functions; degenerate
  inputs return `None` rather than raising.
- `ExtractionScores` model + `score_extraction(extraction, request,
  schema, narrative_text, k=10, perplexity_provider=None)` integration
  function.
- `PerplexityProvider` Protocol (`@runtime_checkable`) with two
  concretes: `DisabledProvider` (always returns `None`; default) and
  `APIPerplexityProvider` (wraps a caller-supplied callable).
- `FeatureClaim.narrative_name` (required) and
  `FeatureClaim.resolved_to: str | None` fields, capturing the LLM's
  resolution from narrative mention to schema feature name.
- `NarrativeExtraction.hallucinations: list[FeatureClaim]` channel for
  unresolved narrative mentions.
- Three new pydantic validators on `NarrativeExtraction`: rank
  permutation over features and hallucinations together; key /
  `resolved_to` consistency for resolved features; `resolved_to is None`
  for every hallucination.
- `textstat` as an optional dependency
  (`pip install "xainarratives[textstat]"`) for the readability metric.
- ADR 0007: resolution at extraction time.

### Changed

- `_EXTRACTION_PROMPT_VERSION` bumped from `"1"` to `"2"`. The wire
  format adds `narrative_name` per feature claim and a separate
  `hallucinations` array.
- `NarrativeExtraction.features` is now keyed by **schema feature
  name** rather than the narrative's name for the feature. Resolution
  happens at extraction time, not at scoring time.
- `extract_narrative_claims` now rejects `features` keys not in the
  schema's resolution vocabulary as a parse failure (advisory
  `GuardrailResult`, no exception).
- `score_extraction` swallows `ImportError` from `readability` so a
  missing `textstat` install degrades to `readability=None` rather than
  cascading to the whole scorer. `readability()` itself keeps the
  strict `ImportError` contract for direct callers.
- ADR 0006 status updated to "superseded in part by 0007".

### Removed

- Support for prompt-version `"1"` extractions. Hard cutover; pre-1.0
  project, no external users, no compatibility shim.

## [0.0.1] - 2026-04-23

### Added

- Initial skeleton: pydantic schema / types / config for all four modalities
  (tabular, text, image, graph).
- `LLMProvider` Protocol + `MockLLMProvider`.
- `PromptTemplate` ABC + `EchoPromptTemplate`.
- `Explainer` orchestrator with sync `explain()`.
- ADRs 0001–0004 recording scope, API style, data-model, and counterfactual-payload
  decisions.
