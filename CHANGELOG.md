# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While `0.y.z`, minor versions may contain breaking changes.

## [Unreleased]

### Changed (BREAKING)

- `ExplanationConfig.mode` is now a required field with no default.
  Construction without `mode=` raises `ValidationError`. Previously
  defaulted to `"auto"`.
- Mode vocabulary is now `"factual"`, `"counterfactual"`,
  `"factual_counterfactual"`. The previous values `"auto"` and
  `"contrastive"` are removed. `"contrastive"` is renamed and redefined
  as `"factual_counterfactual"` (a narrative weaving both factual
  contributions and counterfactual(s)).
- `Explainer._resolve_mode` renamed to `_validate_mode` — the method no
  longer infers mode from the request shape; it validates the explicit
  mode and returns it.
- ADR 0012: explanation-mode vocabulary finalized (supersedes the mode
  portion of ADR 0003).
- Removed `include_confidence` and `include_caveats` fields from
  `ExplanationConfig`. They had no consumer in the library and never
  affected behavior. Setting them now raises `ValidationError`.
  ADR 0013.
- Scoring API renamed for lexical consistency (the word "score" reads
  like a prediction/confidence score, which is not what these functions
  produce):
  `score_extraction` → `grade_extraction`,
  `score_narrativity` → `grade_narrativity`,
  `ExtractionScores` → `ExtractionGrades`,
  `NarrativityScores` → `NarrativityGrades`.
  The source module `xainarratives.metrics.scorer` is renamed to
  `xainarratives.metrics.grader`; top-level re-exports
  (`xainarratives.ExtractionGrades`, etc.) keep the same import path
  modulo the new name. ADR 0014.
- `OpenAICompatibleEchoProvider` API key is now optional and keyword-only.
  It resolves from `api_key=` if passed, else from the environment variable
  named by `api_key_env_var` (default `OPENAI_API_KEY`), raising
  `ValueError` if neither is set. The constructor is now keyword-only past
  `base_url`. ADR 0015.

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
- Seven paper narrativity metrics from Cedro & Martens 2026
  (arXiv:2604.18311): `csr`, `dcpr`, `ccpr`, `cecpr`, `fdr`, `ttcpr`,
  `vcpr`. All pure `(text, provider) -> float | None`; degrade to
  `None` on degenerate inputs.
- `NarrativityScores` model + `score_narrativity(text, provider)`
  orchestrator. Captures the 7 derived metrics plus 9 auxiliary
  primitives (`ppl_ordered`, `ppl_shuffled`, `decay_constant`,
  `dist2`, `ttr`, `vr`, `cr`, `cer`, `n_sentences`) for paper
  replication.
- `xainarratives.metrics._internal/` private subpackage:
  `tokenize` (NLTK sentence/POS, regex word tokenizer), `curve_fit`
  (scipy exponential decay fitter), `lexicons` (vendored JSON loaders
  + greedy phrase counter), `perplexity_utils` (cumulative perplexity
  over sentence prefixes).
- Vendored lexicons under
  `src/xainarratives/metrics/_internal/data/`: 142-entry connectives
  (Das et al. 2018, ACL W18-5042) and 19-entry cause-effect markers
  (paper Appendix A). Loaders assert expected counts at load time.
- `narrativity` optional dependency
  (`pip install "xainarratives[narrativity]"`) bundling
  `nltk>=3.9,<4` and `scipy>=1.13,<2`.
- ADR 0008: narrativity metrics — paper-faithful composition over
  Protocol changes.
- `HuggingFacePerplexityProvider` — local autoregressive perplexity via
  `transformers` + `torch`. Eager-loads model + tokenizer in `__init__`,
  auto-detects CUDA, truncates oversize inputs with one `UserWarning` per
  provider instance. Default `model_name="gpt2"` (~500 MB cached
  download on first use); paper replication wants
  `meta-llama/Llama-3.1-8B`.
- `OpenAICompatibleEchoProvider` — hits any OpenAI-compatible
  `/v1/completions` endpoint with `echo=True, logprobs=1, max_tokens=1`
  (Together.ai, vLLM, TGI's OpenAI shim, OpenAI's legacy completions).
  Dual-shape response parser handles both `choices[0].logprobs` and
  `prompt[0].logprobs`. Catches `openai.OpenAIError` and returns `None`
  per Protocol contract.
- `perplexity-hf` optional dependency
  (`pip install "xainarratives[perplexity-hf]"`) bundling
  `transformers>=4.40,<5` and `torch>=2.0,<3`.
- `perplexity-api` optional dependency
  (`pip install "xainarratives[perplexity-api]"`) bundling
  `openai>=1.30,<2`.
- ADR 0009: perplexity providers — two concretes, no shared base.
- Executable quickstart notebook at `notebooks/01_quickstart.ipynb`:
  end-to-end pipeline on a 30-row OpenML German Credit slice (load,
  one-hot encode, RF + SHAP, build request, generate + extract, score
  extraction + narrativity). Outputs committed for GitHub rendering.
- Vendored `notebooks/data/german_credit_sample.csv` and its
  deterministic regenerator `scripts/generate_german_credit_sample.py`
  (seed 42; runs once before the notebook ever executes).
- `notebook` optional dependency
  (`pip install "xainarratives[notebook]"`) bundling `jupyter`, `shap`,
  `scikit-learn`, `pandas`.
- ADR 0010: ship a quickstart Jupyter notebook.
- Configurable narrative-generation rules: `ExplanationConfig.narrative_rules`
  (a string field, default `DEFAULT_NARRATIVE_RULES`) is injected into the
  system prompt by `FactualTabularPromptTemplate`. The default is the
  four-rule operational definition of an XAI Narrative from Cedro & Martens
  2026; users override it by passing a custom value. Applies to all
  narrative-generating templates by convention.
- ADR 0011: configurable narrative-generation rules.

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
- `grade_extraction` swallows `ImportError` from `readability` so a
  missing `textstat` install degrades to `readability=None` rather than
  cascading to the whole scorer. `readability()` itself keeps the
  strict `ImportError` contract for direct callers.
- ADR 0006 status updated to "superseded in part by 0007".

### Removed

- Support for prompt-version `"1"` extractions. Hard cutover; pre-1.0
  project, no external users, no compatibility shim.
- `APIPerplexityProvider` (abstract callable-wrapper placeholder). Zero
  callers in the codebase, failed the CLAUDE.md "abstractions need ≥2
  implementations" rule. Replaced by `HuggingFacePerplexityProvider` and
  `OpenAICompatibleEchoProvider`.

## [0.0.1] - 2026-04-23

### Added

- Initial skeleton: pydantic schema / types / config for all four modalities
  (tabular, text, image, graph).
- `LLMProvider` Protocol + `MockLLMProvider`.
- `PromptTemplate` ABC + `EchoPromptTemplate`.
- `Explainer` orchestrator with sync `explain()`.
- ADRs 0001–0004 recording scope, API style, data-model, and counterfactual-payload
  decisions.
