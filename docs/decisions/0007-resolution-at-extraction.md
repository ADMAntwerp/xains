# 0007. Resolution at extraction time (revising 0006)

Date: 2026-04-29
Status: Accepted

## Context

ADR 0006 deferred feature-name resolution to the scoring layer ("set-
membership normalization is the job of the downstream scoring layer").
PR 5's design surfaced two costs of that placement:

1. **Lossy normalization at scoring time.** A scoring layer doing string-
   similarity over narrative names is a heuristic — exactly the rule-
   based approach 0006 explicitly rejected for `no_invented_features`.
   Reintroducing it for scoring would push the synonym-vs-hallucination
   ambiguity to the wrong layer.
2. **The LLM has more context to resolve.** At extraction time the model
   sees the schema vocabulary, the prediction, the contributions, AND
   the narrative. At scoring time we would discard most of that and re-
   compute resolution from name strings alone. Strictly worse signal.

A third issue surfaced when designing the scoring functions: every
metric that compares narrative claims against the request's contributions
(`sign_faithfulness`, `value_faithfulness`, `rank_correlation`) needs a
schema-keyed lookup. Keeping resolution at scoring time forces every
metric to re-resolve, with the same fragile heuristic.

## Decision

Move feature-name resolution into the extraction step. The LLM emits:

- `features` keyed by **schema feature names** — every key in this dict
  is the LLM's resolution of a narrative mention to the schema
  vocabulary.
- `hallucinations` as a separate list — narrative mentions the LLM could
  not confidently resolve.

Each `FeatureClaim` also stamps the original `narrative_name` (audit
trail) and `resolved_to` (schema name for resolved features, `None` for
hallucinations).

The extraction prompt receives the schema feature list explicitly as the
**resolution vocabulary** and uses the rule "**when in doubt,
hallucinate**" — biasing the LLM toward false-positive hallucinations
(visible in scoring) rather than silent misattribution (invisible).

`_EXTRACTION_PROMPT_VERSION` bumps to `"2"`. Hard cutover; no support
for `"1"` extractions in PR 5+.

The rank-permutation invariant runs over `features.values()` and
`hallucinations` together — narrative-order ranks are dense over the
union of resolved and unresolved mentions.

## Consequences

- `FeatureClaim` gains `narrative_name: str` (required, min_length=1)
  and `resolved_to: str | None`.
- `NarrativeExtraction` gains `hallucinations: list[FeatureClaim]`. Its
  `features` dict is keyed by schema name. Three validators enforce:
  (a) rank permutation over the union; (b) `features[name].resolved_to
  == name` for every entry; (c) `claim.resolved_to is None` for every
  hallucination.
- `extract_narrative_claims` rejects `features` keys that are not in the
  schema's resolution vocabulary as a parse failure (advisory
  `GuardrailResult`, no exception).
- Scoring metrics (`sign_faithfulness`, `value_faithfulness`,
  `rank_correlation`, `coverage`) work directly on schema names via
  dict lookup. No string-similarity step anywhere in the metrics layer.
- `coverage` counts distinct schema features in `extraction.features`;
  hallucinations do not contribute to coverage (by design — they did not
  resolve).
- `hallucination_count` is a metric in its own right; the audit trail is
  preserved without polluting `coverage`.
- Prompt-version `"1"` extractions are not readable. The project is
  pre-1.0; a compatibility shim would carry maintenance cost for zero
  external users.

## Rejected alternatives

- **C-lenient-2: narrative-keyed dict with `resolved_to` field, no
  separate hallucinations list.** Rejected because every consumer
  (scoring metrics, `coverage`, audit code) wants either the resolved
  features OR the hallucinations, never both interleaved. Partitioning
  at the storage layer is one filter the consumer doesn't have to write.
- **Heuristic resolution at scoring time** (string normalization,
  Levenshtein, embedding similarity). Rejected — same reasoning that
  applied to `no_invented_features` in ADR 0006. Synonym vs
  hallucination is a semantic call, not a string-distance call.
- **Keep `"1"` alongside `"2"` with a compatibility shim.** Rejected
  because the project is pre-1.0 with no external users on `"1"`.
  Compatibility cost > zero, value = zero.
- **Auto-fall-back: if the LLM emits a key not in the schema
  vocabulary, silently move that entry to `hallucinations`.** Rejected
  because the prompt explicitly tells the LLM that `features` keys must
  be schema names. A non-schema key is a contract violation; silent
  recovery trains the LLM (and developers) to ignore the contract.
  Treat as parse failure with a clear audit record.

## References

- ADR 0006: Guardrails and narrative extraction layer (this ADR
  revises it in part).
- Ichmoukhamedov, T., Hinns, J., & Martens, D. (2024). *How good is my
  story? Towards quantitative metrics for evaluating LLM-generated XAI
  narratives.* arXiv preprint arXiv:2412.10220.
