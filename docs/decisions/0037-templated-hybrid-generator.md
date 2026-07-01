# 0037. Templated hybrid generator composes FI and CF narratives

Date: 2026-06-25
Status: Accepted

## Context
The `feature_importance_counterfactual` mode has existed in the `ExplanationMode` literal and been accepted by mode validation (it requires `request.counterfactual`), but no generator produced a hybrid narrative. A hybrid explanation answers two questions at once: why the model predicted the outcome (feature importance) and what would change it (counterfactual). Two templated generators already produce each half independently: `TemplatedNarrativeGenerator` reads `request.contributions` and `TemplatedCounterfactualGenerator` reads `request.counterfactual` via `build_scenarios`. Neither consults `config.mode`; each guards only that the request is tabular. A hybrid request carries both contributions and a counterfactual, so both existing generators already accept it unchanged. This is the first of several hybrid pieces (a templated generator here; an LLM prompt template, dual extraction, and hybrid grading follow in later commits).

## Decision
Add `TemplatedHybridGenerator`, a `NarrativeGenerator` that composes the two existing templated generators rather than reimplementing either. Its `generate` calls `TemplatedNarrativeGenerator.generate` for the feature-importance section and `TemplatedCounterfactualGenerator.generate` for the counterfactual section, then joins their texts with a blank line (`\n\n`) into a two-section narrative. It guards that the request is a `TabularExplanationRequest` and that `request.counterfactual` is not None (a hybrid narrative needs a counterfactual), raising `TypeError` and `ValueError` respectively. It leaves the LLM-specific `GenerationResult` fields (`prompt`, `model_name`, `raw_llm_response`, `tokens_used`) None, matching the other templated generators. The `method` constructor argument is forwarded to the feature-importance sub-generator and `include_method` to the counterfactual sub-generator, mirroring their own constructors. The generator is exported from `xains.generation` and top-level `xains`.

## Rationale
- Composition over reimplementation. The two sub-generators already produce correct, tested output for each half; delegating to them means the hybrid inherits their behavior and fixes for free, and there is no duplicated rendering logic to drift.
- The two-section shape (blank-line separated) keeps the halves textually separable, which the later dual-extraction step relies on: the feature-importance extractor finds its rank/sign/value claims in the first section and the counterfactual extractor finds its before/after claims in the second, without the two fighting over intermingled sentences.
- Neither sub-generator checks `config.mode`, so composing them for the hybrid mode needs no change to either; the hybrid request already satisfies both.
- The counterfactual guard fails loud. A hybrid request without a counterfactual is a caller error; raising is consistent with the mode validation in `Explainer` that already requires `request.counterfactual` for this mode.

## Consequences
- `mode="feature_importance_counterfactual"` now has a working deterministic generator; a user can produce a hybrid narrative with no LLM.
- The generator delegates, so any change to the feature-importance or counterfactual templated output flows through to the hybrid automatically.
- Six unit tests pin the behavior: the text is the FI narrative, a blank line, then the CF narrative; both sections are present; output is deterministic; LLM-metadata fields are None; a non-tabular request raises; a missing counterfactual raises.
- This commit covers only the templated path. The LLM hybrid path (a prompt template that renders both blocks) and the hybrid extraction and grading are separate, later commits. Until dual extraction lands, a hybrid narrative reaching `Explainer` extraction is still handled by the feature-importance branch (ADR 0033).

## Rejected alternatives
- **Reimplement the FI and CF rendering inside the hybrid generator.** Rejected: it duplicates logic that already exists and tested, inviting drift between the standalone and hybrid outputs.
- **Weave the two halves into a single integrated paragraph.** Rejected for now: a woven narrative intermingles feature-importance and counterfactual sentences, which complicates the later per-section extraction. The two-section form keeps the halves cleanly separable.
- **Skip the counterfactual guard and let the CF sub-generator fail.** Rejected: an explicit guard on the hybrid generator gives a clearer error tied to the hybrid contract, matching the mode validation elsewhere.
- **Make the hybrid an LLM-only mode.** Rejected: the templated path is cheap, deterministic, and useful as a baseline, exactly as it is for the FI and CF modes; there is no reason to deny the hybrid the same.

## References
- ADR 0030 - templated counterfactual generator.
- ADR 0031 - single counterfactual per request.
- ADR 0033 - Explainer dispatches extraction by mode (hybrid currently falls through to the FI branch).
- `src/xains/generation/templated_hybrid.py` - the generator.
- `tests/unit/test_generation_templated_hybrid.py` - the six pinning tests.
