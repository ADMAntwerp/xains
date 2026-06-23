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
from xains.guardrails import extract_narrative_claims
from xains.guardrails.types import GuardrailResult, NarrativeExtraction
from xains.providers.base import LLMProvider
from xains.schema import DatasetSchema
from xains.types import (
    ExplanationMode,
    ExplanationRequest,
    ExplanationResult,
    TabularExplanationRequest,
)


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
        has_cf = bool(request.counterfactuals)
        self._check_explicit_mode(self.config.mode, has_cf=has_cf)
        return self.config.mode

    @staticmethod
    def _check_explicit_mode(mode: ExplanationMode, *, has_cf: bool) -> None:
        if mode == "counterfactual" and not has_cf:
            raise ValueError("config.mode='counterfactual' requires request.counterfactuals.")
        if mode == "feature_importance_counterfactual" and not has_cf:
            raise ValueError(
                "config.mode='feature_importance_counterfactual' requires request.counterfactuals."
            )
        # "feature_importance" has no prerequisites.

    @staticmethod
    def _warn_if_counterfactual_does_not_flip(request: ExplanationRequest) -> None:
        """A CF that predicts the same class as the factual is almost always a user bug."""
        if not request.counterfactuals:
            return
        factual_class = request.prediction.predicted_class
        for i, cf in enumerate(request.counterfactuals):
            if cf.predicted_class == factual_class:
                warnings.warn(
                    f"Counterfactual #{i} predicts the same class "
                    f"({factual_class!r}) as the factual. This is usually a "
                    "user error — true counterfactuals flip the prediction.",
                    UserWarning,
                    stacklevel=3,
                )
