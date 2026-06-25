# 0032. Counterfactual fidelity scoring

Date: 2026-06-25
Status: Accepted

## Context
The feature-importance path scores how faithfully a narrative reports the attributions it was given: `extract_narrative_claims` turns the narrative into structured per-feature claims (rank, sign, value), and `grade_extraction` compares them to the request, producing `ExtractionGrades` (sign and value faithfulness, rank correlation, coverage, hallucination count). The counterfactual path had generation (ADRs 0028 to 0031) but no scoring. A counterfactual narrative makes a different kind of claim: it says a feature would change from one value to another to flip the prediction. Whether the narrative reports those before and after values correctly, mentions all the features that actually changed, and does not invent changes, is the counterfactual analogue of fidelity. The existing FI extraction schema (one rank/sign/value per feature) cannot express a before/after pair, so it cannot be reused directly. Per ADR 0031 a request now carries exactly one counterfactual, which simplifies the ground truth to one (before, after) pair per changed feature.

## Decision
Add a counterfactual fidelity layer that mirrors the FI fidelity layer. A new LLM extraction, `extract_counterfactual_claims(text, request, schema, judge_llm)`, returns a `CounterfactualExtraction`: per-feature `CounterfactualFeatureClaim` records (`narrative_name`, `resolved_to`, `stated_before`, `stated_after`, `stated_direction`) keyed by schema feature name, plus an `invented` channel for mentions that do not resolve to a schema feature. It reuses the ADR 0007 resolution discipline verbatim (resolve to schema names; when in doubt, place in `invented`). The extraction prompt deliberately does not show the actual counterfactual to the LLM; it shows only the schema vocabulary and the factual prediction, so the LLM extracts what the narrative literally says rather than the ground truth. Three pure metric functions in `counterfactual_fidelity.py` score the extraction against `build_scenarios` ground truth: `change_fidelity` (fraction of resolved claims about actually-changed features where both `stated_before` and `stated_after` match, strict), `cf_coverage` (fraction of actually-changed features the narrative mentions), and `invented_features` (count of unresolved mentions, the analogue of `hallucination_count`). `grade_counterfactual(extraction, request, schema)` composes them into a flat `CounterfactualGrades` aggregate (`change_fidelity`, `coverage`, `invented_features`, `prompt_version`), with `COUNTERFACTUAL_GRADE_DIRECTIONS` (change_fidelity up, coverage up, invented_features down). `render_grades` gains a `counterfactual=` parameter rendering a "Counterfactual fidelity" section through the same `_render_section` machinery, in canonical order between extraction and narrativity.

## Rationale
- A separate flat `CounterfactualGrades`, not a merge into `ExtractionGrades` and not an abstract base, follows the orthogonality precedent of ADRs 0008, 0023, and 0025: each distinct scoring concern is its own aggregate, and fields are never mode-dependent nulls. The abstract base across grade aggregates is deferred until the hybrid feature-importance-plus-counterfactual mode is a third concrete case, so the shared shape can be factored from three real examples rather than guessed from two.
- LLM extraction (rather than scoring the narrative text structurally) mirrors the FI path and is robust to paraphrase: a narrative may say "lower the debt ratio to 0.2" rather than naming the schema feature and value literally, and the LLM resolves that the same way it does for FI.
- The extraction prompt withholds the counterfactual on purpose. Showing the LLM the real before and after values would let it fill them in when the narrative is vague, making `change_fidelity` score near 1.0 regardless of narrative quality. Withholding the ground truth is what makes the metric measure the narrative rather than the extractor.
- `change_fidelity` requires both before and after to match. A narrative that states only the target value has not fully stated the change; scoring it correct would reward incomplete claims. A partially stated claim is incorrect, not skipped.
- Value comparison branches on `schema.feature(name).dtype`: only `numeric` uses `math.isclose` (with a numeric type guard so a non-numeric stated value scores incorrect instead of raising); `ordinal`, `categorical`, `boolean`, and `text` use equality. Ordinal uses equality because the schema requires ordinal features to carry `categories: list[str]`, so their values are category labels, not numbers; routing ordinal to `isclose` would score every ordinal claim incorrect.
- `stated_direction` is captured at extraction but not scored in this commit. It is free diagnostic data and reserves a future `direction_fidelity` metric without a re-extraction.
- Grading stays a separate downstream call, not part of `Explainer.explain`, matching how `grade_extraction` and `grade_narrativity` already work.

## Consequences
- New extraction: `CounterfactualFeatureClaim`, `CounterfactualExtraction` (in guardrails types), `extract_counterfactual_claims` (in guardrails extraction), exported from `xains.guardrails` and top-level.
- New metrics: `change_fidelity`, `cf_coverage`, `invented_features` (in `metrics/counterfactual_fidelity.py`), exported from `xains.metrics` and top-level.
- New grader: `CounterfactualGrades`, `grade_counterfactual`, `COUNTERFACTUAL_GRADE_DIRECTIONS` (in `metrics/grader.py`), exported from `xains.metrics` and top-level.
- `render_grades` gains a `counterfactual=` parameter and a "Counterfactual fidelity" section, rendered through the existing `_render_section` path.
- `grade_counterfactual` takes `(extraction, request, schema)` only: it drops the `narrative_text` and `k` parameters that `grade_extraction` carries, because no counterfactual metric reads the raw narrative or applies a top-k cap.
- Tabular only. Text, image, and graph counterfactual scoring is out of scope; each modality's notion of a value change differs.
- The hybrid feature-importance-plus-counterfactual mode is out of scope; its grading may compose the FI and CF graders rather than inherit a base, and is the case that will reveal the right abstract-grades shape.

## Rejected alternatives
- **Merge counterfactual fields into `ExtractionGrades`.** Rejected: the FI aggregate would carry counterfactual fields that are always null in FI mode and vice versa, reversing the orthogonality cleanup of ADRs 0023 and 0025.
- **Introduce an abstract grades base now.** Rejected: with only two concrete aggregates the shared surface is one or two fields; the base would be speculative. Defer it to the hybrid third case.
- **Score the narrative structurally against `build_scenarios` without an LLM.** Rejected: it cannot handle paraphrase or feature synonyms and would only work for the templated generator, not for LLM-written narratives, which are the main case.
- **Show the counterfactual to the extraction LLM.** Rejected: it leaks the ground truth into the extractor and makes the fidelity metric near-constant regardless of narrative quality.
- **Skip partially stated claims in `change_fidelity` rather than scoring them incorrect.** Rejected: skipping inflates scores for vague narratives that name a change without committing to its values.
- **Route ordinal through `isclose` like numeric.** Rejected: the schema models ordinals as string categories, so `isclose` would score every ordinal claim incorrect; ordinal uses equality.
- **Carry a `narrative_text` parameter on `grade_counterfactual` for signature symmetry with `grade_extraction`.** Rejected: it would be unread; an honest signature is better than a matching dead parameter.

## References
- ADR 0007 - resolution at extraction time (reused for the counterfactual claim resolution).
- ADR 0008 - narrativity metrics orthogonality (separate aggregate precedent).
- ADR 0023, ADR 0025 - removal of perplexity and readability from `ExtractionGrades` (orthogonality precedent).
- ADR 0024, ADR 0026 - render directions and scored_only (the render path the CF section reuses).
- ADR 0028 - changed-features diff (the per-feature change source).
- ADR 0031 - single counterfactual per request (one (before, after) pair per feature).
- `src/xains/guardrails/extraction.py` - `extract_counterfactual_claims`.
- `src/xains/metrics/counterfactual_fidelity.py` - the three metric functions.
- `src/xains/metrics/grader.py` - `grade_counterfactual`, `CounterfactualGrades`.
