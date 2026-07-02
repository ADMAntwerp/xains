"""Explainer — the sync orchestrator.

Responsibilities:

1. Validate that the request's modality matches the schema's modality.
2. Validate the configured mode (``feature_importance`` /
   ``counterfactual`` / ``feature_importance_counterfactual``) against
   the request.
3. Generate the narrative via the configured ``NarrativeGenerator``.
4. Optionally run judge-based narrative extraction on the generated
   text (requires ``judge_llm``).
5. Package the result with audit metadata.

Not in v0: caching, retries, streaming, batch, eval pipeline.
Each gets its own PR and its own ADR.
"""

import warnings

from xains.config import ExplanationConfig
from xains.generation.base import NarrativeGenerator
from xains.guardrails import extract_counterfactual_claims, extract_narrative_claims
from xains.guardrails.types import (
    CounterfactualExtraction,
    GuardrailResult,
    NarrativeExtraction,
)
from xains.providers.base import LLMProvider
from xains.schema import DatasetSchema
from xains.types import (
    ExplanationMode,
    ExplanationRequest,
    ExplanationResult,
    TabularExplanationRequest,
)


def _merge_token_counts(
    a: dict[str, int] | None, b: dict[str, int] | None
) -> dict[str, int] | None:
    """Element-wise sum of two token-count dicts.

    Used by the hybrid extraction dispatch (ADR 0040) to combine the FI and
    CF judge calls' token usage. Returns ``None`` only when both inputs are
    ``None``; when only one side is ``None``, the other survives untouched
    (a shallow copy). When both are dicts, keys from the union are summed
    (missing keys treated as 0).
    """
    if a is None:
        return dict(b) if b is not None else None
    if b is None:
        return dict(a)
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}


class Explainer:
    """Generate natural-language explanations from pre-computed attributions."""

    def __init__(
        self,
        schema: DatasetSchema,
        generator: NarrativeGenerator,
        config: ExplanationConfig | None = None,
        judge_llm: LLMProvider | None = None,
    ) -> None:
        self.schema = schema
        self.generator = generator
        self.config = config if config is not None else ExplanationConfig(mode="feature_importance")
        # No fallback to a generator-held LLM; explain() validates when
        # extraction is requested.
        self.judge_llm = judge_llm

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        self._validate_modality(request)
        self._warn_if_counterfactual_does_not_flip(request)
        mode = self._validate_mode(request)

        gen = self.generator.generate(request, self.schema, self.config)

        # Defensive copy so we can append the optional judge failure without
        # mutating the GenerationResult's internal list.
        guardrails: list[GuardrailResult] | None = (
            list(gen.guardrails) if gen.guardrails is not None else None
        )
        narrative_extraction: NarrativeExtraction | None = None
        counterfactual_extraction: CounterfactualExtraction | None = None
        guardrail_tokens_used: dict[str, int] | None = None

        if (
            self.config.run_guardrails
            and self.config.extract_narrative
            and isinstance(request, TabularExplanationRequest)
        ):
            if self.judge_llm is None:
                raise ValueError(
                    "extract_narrative=True requires judge_llm to be passed "
                    "to Explainer(...). Pass an LLMProvider as judge_llm, or "
                    "set ExplanationConfig(extract_narrative=False)."
                )
            # Dispatch extraction by mode (ADR 0033, ADR 0040):
            # - "counterfactual"  -> CF extraction only
            # - "feature_importance" -> FI extraction only
            # - "feature_importance_counterfactual" -> both extractors run
            #   over the same generated text; token counts merged
            #   element-wise, each extraction's advisory failure appended
            #   independently.
            if mode == "counterfactual":
                cf_extraction, judge_response, failure = extract_counterfactual_claims(
                    gen.text, request, self.schema, self.judge_llm
                )
                guardrail_tokens_used = judge_response.tokens_used
                if cf_extraction is not None:
                    counterfactual_extraction = cf_extraction
                if failure is not None:
                    if guardrails is None:
                        guardrails = []
                    guardrails.append(failure)
            elif mode == "feature_importance_counterfactual":
                fi_extraction, fi_judge_response, fi_failure = extract_narrative_claims(
                    gen.text, request, self.schema, self.judge_llm
                )
                cf_extraction, cf_judge_response, cf_failure = extract_counterfactual_claims(
                    gen.text, request, self.schema, self.judge_llm
                )
                guardrail_tokens_used = _merge_token_counts(
                    fi_judge_response.tokens_used, cf_judge_response.tokens_used
                )
                if fi_extraction is not None:
                    narrative_extraction = fi_extraction
                if cf_extraction is not None:
                    counterfactual_extraction = cf_extraction
                if fi_failure is not None:
                    if guardrails is None:
                        guardrails = []
                    guardrails.append(fi_failure)
                if cf_failure is not None:
                    if guardrails is None:
                        guardrails = []
                    guardrails.append(cf_failure)
            else:  # "feature_importance"
                extraction, judge_response, failure = extract_narrative_claims(
                    gen.text, request, self.schema, self.judge_llm
                )
                guardrail_tokens_used = judge_response.tokens_used
                if extraction is not None:
                    narrative_extraction = extraction
                if failure is not None:
                    if guardrails is None:
                        guardrails = []
                    guardrails.append(failure)

        return ExplanationResult(
            text=gen.text,
            mode=mode,
            prompt=gen.prompt,
            raw_llm_response=gen.raw_llm_response,
            model_name=gen.model_name,
            tokens_used=gen.tokens_used,
            latency_ms=gen.latency_ms,
            guardrails=guardrails,
            narrative_extraction=narrative_extraction,
            counterfactual_extraction=counterfactual_extraction,
            guardrail_tokens_used=guardrail_tokens_used,
        )

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _validate_modality(self, request: ExplanationRequest) -> None:
        if request.modality != self.schema.modality:
            raise ValueError(
                f"Request modality {request.modality.value!r} does not match "
                f"schema modality {self.schema.modality.value!r}."
            )

    def _validate_mode(self, request: ExplanationRequest) -> ExplanationMode:
        has_cf = request.counterfactual is not None
        self._check_explicit_mode(self.config.mode, has_cf=has_cf)
        return self.config.mode

    @staticmethod
    def _check_explicit_mode(mode: ExplanationMode, *, has_cf: bool) -> None:
        if mode == "counterfactual" and not has_cf:
            raise ValueError("config.mode='counterfactual' requires request.counterfactual.")
        if mode == "feature_importance_counterfactual" and not has_cf:
            raise ValueError(
                "config.mode='feature_importance_counterfactual' requires request.counterfactual."
            )
        # "feature_importance" has no prerequisites.

    @staticmethod
    def _warn_if_counterfactual_does_not_flip(request: ExplanationRequest) -> None:
        """A CF that predicts the same class as the factual is almost always a user bug."""
        cf = request.counterfactual
        if cf is None:
            return
        factual_class = request.prediction.predicted_class
        if cf.predicted_class == factual_class:
            warnings.warn(
                f"Counterfactual predicts the same class ({factual_class!r}) "
                "as the factual. This is usually a user error — a true "
                "counterfactual flips the prediction.",
                UserWarning,
                stacklevel=3,
            )
