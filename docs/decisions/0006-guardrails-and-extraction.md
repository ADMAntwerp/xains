# 0006. Guardrails and narrative extraction layer

Date: 2026-04-27
Status: Accepted; superseded in part by 0007 (resolution-at-extraction)

## Context

LLM-generated explanations need verification before they reach end users.
Two failure modes matter most:

1. The narrative names the wrong outcome — catastrophic. The user reads
   "approved" when the model said "denied".
2. The narrative mis-describes a feature's contribution — wrong direction,
   wrong rank, wrong value, or invented attribute. Subtler, but undermines
   trust over time.

Pure rule-based detection of feature-attribution mistakes runs into a hard
wall: synonyms ("debt-to-income" vs "debt load", "credit utilization" vs
"revolving balance") are valid prose, not hallucinations, and rule-based
approaches built on token overlap, identifier shape, or length thresholds
cannot distinguish them. The question is fundamentally semantic.

## Decision

A two-tier guardrails layer.

1. **Rule-based: `class_name_mentioned`.** Strict failure severity,
   modality-agnostic. Exact case-insensitive substring match against
   `schema.target.classes[prediction.predicted_class]`. Catches the
   catastrophic case cheaply.

2. **LLM extraction: `extract_narrative_claims`.** A single LLM call
   produces structured per-feature claims following the schema in
   Ichmoukhamedov et al. 2024 (arXiv:2412.10220), Fig. 3. Records what the
   narrative says — `rank`, `sign`, `value`, `assumption` per feature.
   Does not score; PR 5 builds scoring metrics on top of the extraction.

Both checks gated by `config.run_guardrails` (master switch, default
`True`). Extraction additionally gated by `config.extract_narrative`
(default `True`). An optional `Explainer.judge_llm` parameter routes the
extraction call to a separate provider; defaults to reusing the generator.

Field names in the extraction match the paper verbatim: `rank`, `sign`,
`value`, `assumption`.

Prompt versioning: `_EXTRACTION_PROMPT_VERSION = "1"` is stamped on every
`NarrativeExtraction`. Bumped whenever the prompt text or output schema
changes, so PR 5 can distinguish extractions produced under different
contracts.

### Rationale

- **Why `class_name_mentioned` *is* rule-based.** Class labels are short
  canonical strings declared in the schema; the LLM has no reason to
  paraphrase them, and if it does, that itself is a failure worth
  flagging. Exact substring match is synonym-proof (because there are no
  expected synonyms), modality-agnostic, and ~10 lines of code.
- **Why a rule-based feature-invention check was rejected.** Three design
  iterations of `no_invented_features` (length-≥4 token filter,
  identifier-shape filter, broader curated vocabulary) all failed the
  synonym test. A good explanation legitimately rephrases `dti` as "debt
  load". Rule-based feature-name resolution is fundamentally semantic; it
  belongs in extraction + PR 5 scoring, not in guardrails.
- **Why extraction over verdict.** Separating "what does the narrative
  claim" (LLM) from "is that correct" (deterministic math) is more
  auditable, more re-scorable, and lets verdicts be re-derived on
  existing extractions when scoring metrics evolve.
- **Why the paper's exact field names.** Readers familiar with
  Ichmoukhamedov et al. recognize the structure immediately; PR 5's
  scoring code maps paper equations to fields one-to-one. Verbatim is
  cheaper than translated.

## Consequences

- `ExplanationResult` gains three optional fields:
  `guardrails: list[GuardrailResult] | None`,
  `narrative_extraction: NarrativeExtraction | None`,
  `guardrail_tokens_used: dict[str, int] | None`. `None` means "did not
  run"; populated values are real results.
- `ExplanationConfig` gains `run_guardrails: bool = True` and
  `extract_narrative: bool = True`. Default-on for alpha; users opt out
  for cost in production.
- Token accounting splits cleanly: `tokens_used` (generator only),
  `guardrail_tokens_used` (extraction only). No summation done by the
  library. Same key shape as ADR 0005 (`{"input", "output", "total"}`).
- `Explainer.__init__` gains `judge_llm: LLMProvider | None = None`.
  Defaults to reusing `self.llm`. The name reflects the role ("a separate
  LLM that inspects the generator's output"), not the current
  implementation.
- Non-tabular requests (text, image, graph) run only
  `class_name_mentioned`. Extraction for non-tabular modalities is a
  future PR with different per-modality schema needs.
- Parse failures (malformed JSON, schema violation, rank-permutation
  violation) produce `narrative_extraction = None` and append an advisory
  failure `GuardrailResult` to `guardrails`. Token spend on the failed
  call is still recorded in `guardrail_tokens_used`.

## Rejected alternatives

- **LLM-as-judge giving pass/fail verdicts directly.** Rejected because
  verdicts can't be re-scored when metrics evolve, and "the judge said
  no" is harder to audit than "the judge claims feature X had sign +1;
  importance was −0.3; that's a sign mismatch."
- **A `Guardrail` Protocol with multiple implementations.** Rejected
  because abstractions in this codebase need ≥2 concrete implementations
  of the same shape (CLAUDE.md). `class_name_mentioned` and
  `extract_narrative_claims` have different signatures and return shapes
  — they don't share an interface usefully.
- **Discriminated union for `GuardrailResult.details`.** Rejected because
  three guardrails total (one rule, one extraction, one extraction-
  failure path) is below the union threshold. Flat `dict[str, Any]` is
  sufficient. Revisit if guardrail count grows past ~5.
- **Auto-retry on malformed extraction JSON.** Rejected because parse
  failures are typically deterministic (same model + same prompt → same
  failure). Retry doubles cost without changing the outcome. Failure
  becomes an advisory record, not an exception.
- **Truncating the explanation text before sending to the extraction
  prompt.** Rejected because explanations are already capped at
  `config.max_length_words`. Adding a second length budget creates
  surprises when the first one is hit.

## Revised by 0007

The statement that "set-membership normalization is the job of the
downstream scoring layer" was reversed by ADR 0007 — feature-name
resolution now happens at extraction time, not at scoring time. The rest
of this ADR (rule-based check, two-tier architecture, paper-verbatim
field names, prompt versioning) remains in force.

## References

- Ichmoukhamedov, T., Hinns, J., & Martens, D. (2024). *How good is my
  story? Towards quantitative metrics for evaluating LLM-generated XAI
  narratives.* arXiv preprint arXiv:2412.10220.

BibTeX:

```bibtex
@article{ichmoukhamedov2024good,
  title={How good is my story? Towards quantitative metrics for evaluating LLM-generated XAI narratives},
  author={Ichmoukhamedov, Timour and Hinns, James and Martens, David},
  journal={arXiv preprint arXiv:2412.10220},
  year={2024}
}
```
