# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While `0.y.z`, minor versions may contain breaking changes.

## [Unreleased]

### Added

- `build_scenarios` + `CounterfactualScenario` (in `xains.counterfactuals`):
  the single source of truth for per-counterfactual scenario data (flip
  labels, changed features, order). `CounterfactualTabularPromptTemplate`
  now consumes it (output unchanged). `TemplatedCounterfactualGenerator`
  (in `xains.generation`): LLM-free counterfactual narratives, mirroring
  `TemplatedNarrativeGenerator`. End-to-end `mode="counterfactual"` tests
  through `Explainer.explain()` for both the templated and LLM paths.
  ADR 0030.
- `xains.counterfactuals.changed_features(factual, cf)` and the
  `ChangedFeature(name, before, after)` model. Pure function over a factual
  dict and a `TabularCounterfactual`; honors `cf.changed_features` as an
  explicit override else computes the diff; raises `ValueError` when a
  reported key is absent from the factual. Tabular only this PR.
  Foundation for counterfactual narrative templates. ADR 0028.
- `CounterfactualTabularPromptTemplate` (re-exported from `xains.prompts`).
  Verbalizes `request.counterfactuals` in the order provided (no ranking,
  per ADR 0004), leads with the prediction flip, lists changed-features as
  `name: before -> after [unit]`. Single CF renders un-numbered; multiple
  CFs are numbered `Scenario 1`, `Scenario 2`, .... `include_method=True`
  appends `(method: <cf.method>)` when the CF carries that provenance
  field; default is off. Editable per ADR 0017
  (`system_template=` / `user_template=` / `extra_placeholders=`).
  Tabular only this PR. ADR 0029.

### Changed (BREAKING)

- The `DEFAULT_SYSTEM_TEMPLATE` and `DEFAULT_USER_TEMPLATE` constants for
  `FeatureImportanceTabularPromptTemplate` are no longer re-exported from
  the `xains.prompts` package level (they would collide with the new
  `CounterfactualTabularPromptTemplate` constants of the same names).
  Import them from the submodule instead:
  `from xains.prompts.feature_importance_tabular import DEFAULT_SYSTEM_TEMPLATE`.
  ADR 0029.

## [0.1.1] - 2026-06-25

### Changed

- README: install via pip/uv, .env setup documented, scope note on attribution methods, three new badges, and minimal-example XAIN framing.
- `render_grades` formats metric floats to 2 decimals (`f"{value:.2f}"`); ints and `None` are unchanged.

## [0.1.0] - 2026-06-23

### Changed (BREAKING)

- `ExplanationConfig.mode` is now a required field with no default.
  Construction without `mode=` raises `ValidationError`. Previously
  defaulted to `"auto"`.
- Mode vocabulary is now `"feature_importance"`, `"counterfactual"`,
  `"feature_importance_counterfactual"`. The previous values `"auto"` and
  `"contrastive"` are removed. `"contrastive"` is renamed and redefined
  as `"feature_importance_counterfactual"` (a narrative weaving both factual
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
- Mode vocabulary renamed for semantic accuracy (the prior `"factual"`
  collided with the counterfactual-explanation literature's meaning —
  "factual" is the input datapoint, not an explanation style; the mode
  actually means "explain via feature-importance contributions"):
  `"factual"` → `"feature_importance"`,
  `"factual_counterfactual"` → `"feature_importance_counterfactual"`.
  `"counterfactual"` unchanged. `FactualTabularPromptTemplate` →
  `FeatureImportanceTabularPromptTemplate`; module path
  `xainarratives.prompts.factual_tabular` →
  `xainarratives.prompts.feature_importance_tabular`. The CF-literature
  use of "factual" in `Explainer._warn_if_counterfactual_does_not_flip`
  (`factual_class` local var + warning prose) is deliberately preserved
  as the now-unambiguous "input datapoint" sense. ADR 0016 (supersedes
  the mode naming in ADR 0012).
- `Explainer` now takes `generator=` (a `NarrativeGenerator`:
  `LLMNarrativeGenerator` or a future templated generator) instead of
  `prompt_template=` / `llm=`. `judge_llm` is required when
  `extract_narrative=True` — the silent `self.llm` fallback is removed;
  `explain()` raises `ValueError` otherwise.
  `ExplanationResult.{prompt, model_name, raw_llm_response}` widened
  to `str | None` (templated generators produce no LLM metadata).
  ADR 0018.
- Renamed the package from `xainarratives` to `xain`.
  `from xainarratives import ...` becomes `from xain import ...`;
  `pip install "xainarratives[extra]"` becomes
  `pip install "xain[extra]"`. ADR 0021.
- Renamed the package from `xain` to `xains` (the PyPI name `xain` was unavailable). `from xain import ...` becomes `from xains import ...`; the distribution is `xains`. ADR 0022.
- `ExtractionGrades` no longer carries a `perplexity` field and `grade_extraction` no longer accepts a `perplexity_provider` keyword argument. Whole-text perplexity remains available as `NarrativityGrades.ppl_ordered` via `grade_narrativity`. ADR 0023.
- `ExtractionGrades` no longer carries a `readability` field; `grade_extraction` no longer computes it. The `readability(extraction, narrative_text)` helper, the `textstat` optional extra, and the public re-exports remain - readability becomes opt-in, called directly. ADR 0025.

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
  system prompt by `FeatureImportanceTabularPromptTemplate`. The default is the
  four-rule operational definition of an XAI Narrative from Cedro & Martens
  2026; users override it by passing a custom value. Applies to all
  narrative-generating templates by convention.
- ADR 0011: configurable narrative-generation rules.
- `FeatureImportanceTabularPromptTemplate` now accepts `system_template`,
  `user_template`, and `extra_placeholders` (all keyword-only, defaulted)
  for editable prompts with `{placeholder}` substitution.
  `DEFAULT_SYSTEM_TEMPLATE` and `DEFAULT_USER_TEMPLATE` are exported from
  `xainarratives.prompts`. The quickstart notebook prints the rendered
  prompt before sending. ADR 0017.
- `TemplatedNarrativeGenerator` - LLM-free feature-importance
  narratives. Verbalizes ranked contributions as prose with no LLM
  call; method-agnostic by default (`method="SHAP"` reproduces
  Cedro 2026's templated-baseline wording); editable lead/clause
  templates; raw values from `request.features`; tabular-only. Slots
  into `Explainer(generator=)` and flows through the same extraction
  + grading path as LLM narratives (LLM-free generation, LLM-graded).
  The shared `substitute()` helper now has its second user. ADR 0019.
- OpenAI and OpenRouter narrative-generation providers:
  `OpenAIProvider` (reads `OPENAI_API_KEY`) and `OpenRouterProvider`
  (reads `OPENROUTER_API_KEY`, optional `HTTP-Referer`/`X-Title`
  headers), both thin presets over a new public
  `OpenAICompatibleProvider` base usable directly for any
  OpenAI-compatible endpoint (Together, Groq, vLLM, ...). Eager key
  resolution, lazy SDK import. New `openai` pip extra. All providers
  now top-level importable: `from xainarratives import
  AnthropicProvider, OpenAIProvider, OpenRouterProvider,
  OpenAICompatibleProvider, ...`. ADR 0020.
- `render_grades(extraction=None, narrativity=None) -> str` plus
  source-of-truth direction dicts (`EXTRACTION_GRADE_DIRECTIONS`,
  `NARRATIVITY_GRADE_DIRECTIONS`). Scored metrics render with desired-direction
  arrows (`↑`/`↓`); auxiliary narrativity primitives render without arrows;
  `prompt_version` is omitted. All three are top-level importable from
  `xains` (and from `xains.metrics`). ADR 0024.
- `scored_only: bool = False` option on `render_grades`. When `True`, fields
  absent from the direction dicts are omitted (drops the 9 NarrativityGrades
  auxiliaries; no visible effect on ExtractionGrades). Default `False`
  preserves prior behaviour. README narrativity sample and notebook Step 7
  display use it. ADR 0026.
- Optional `dotenv` extra (`pip install "xains[dotenv]"`) bundling
  `python-dotenv>=1.0`, plus a tracked `.env.example` at the repo root.
  The library does not call `load_dotenv()` itself; callers opt in when
  they want it. Core deps remain pydantic-only. ADR 0027.
- Python 3.14 added to the supported / tested matrix. CI now runs across
  3.11, 3.12, 3.13, and 3.14; the `Programming Language :: Python :: 3.14`
  trove classifier is published. `requires-python = ">=3.11"` is unchanged.

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
- `perplexity` field from `ExtractionGrades` and `perplexity_provider`
  keyword argument from `grade_extraction`. Whole-text perplexity is the
  narrativity surface (`NarrativityGrades.ppl_ordered` via
  `grade_narrativity`), not the fidelity surface. Pre-1.0 hard cutover,
  no shim. ADR 0023.
- `readability` field from `ExtractionGrades`; `grade_extraction` no
  longer computes it. The `readability(extraction, narrative_text)`
  helper, the `textstat` optional extra, and the public re-exports
  remain - readability becomes opt-in, called directly. Pre-1.0 hard
  cutover, no shim. ADR 0025.

## [0.0.1] - 2026-04-23

### Added

- Initial skeleton: pydantic schema / types / config for all four modalities
  (tabular, text, image, graph).
- `LLMProvider` Protocol + `MockLLMProvider`.
- `PromptTemplate` ABC + `EchoPromptTemplate`.
- `Explainer` orchestrator with sync `explain()`.
- ADRs 0001–0004 recording scope, API style, data-model, and counterfactual-payload
  decisions.
