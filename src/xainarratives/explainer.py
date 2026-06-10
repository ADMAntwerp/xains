"""Explainer — the sync orchestrator.

Responsibilities:

1. Validate that the request's modality matches the schema's modality.
2. Validate the configured mode (``feature_importance`` /
   ``counterfactual`` / ``feature_importance_counterfactual``) against
   the request.
3. Render the prompt via the configured ``PromptTemplate``.
4. Call the configured ``LLMProvider``.
5. Package the result with audit metadata.

Not in v0: caching, retries, streaming, batch, eval pipeline.
Each gets its own PR and its own ADR.
"""

import time
import warnings

from xainarratives.config import ExplanationConfig
from xainarratives.guardrails import class_name_mentioned, extract_narrative_claims
from xainarratives.guardrails.types import GuardrailResult, NarrativeExtraction
from xainarratives.prompts.base import PromptTemplate
from xainarratives.providers.base import LLMProvider
from xainarratives.schema import DatasetSchema
from xainarratives.types import (
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
        llm: LLMProvider,
        prompt_template: PromptTemplate,
        config: ExplanationConfig | None = None,
        judge_llm: LLMProvider | None = None,
    ) -> None:
        self.schema = schema
        self.llm = llm
        self.prompt_template = prompt_template
        self.config = config if config is not None else ExplanationConfig(mode="feature_importance")
        self.judge_llm = judge_llm if judge_llm is not None else self.llm

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        self._validate_modality(request)
        self._warn_if_counterfactual_does_not_flip(request)

        mode = self._validate_mode(request)
        system, user = self.prompt_template.render(request, self.schema, self.config)

        start = time.perf_counter()
        response = self.llm.generate(system, user)
        latency_ms = (time.perf_counter() - start) * 1000.0

        guardrails: list[GuardrailResult] | None = None
        narrative_extraction: NarrativeExtraction | None = None
        guardrail_tokens_used: dict[str, int] | None = None

        if self.config.run_guardrails:
            guardrails = [class_name_mentioned(response.text, self.schema, request.prediction)]
            if self.config.extract_narrative and isinstance(request, TabularExplanationRequest):
                extraction, judge_response, failure = extract_narrative_claims(
                    response.text, request, self.schema, self.judge_llm
                )
                guardrail_tokens_used = judge_response.tokens_used
                if extraction is not None:
                    narrative_extraction = extraction
                if failure is not None:
                    guardrails.append(failure)

        return ExplanationResult(
            text=response.text,
            mode=mode,
            prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
            raw_llm_response=response.text,
            model_name=response.model_name,
            tokens_used=response.tokens_used,
            latency_ms=latency_ms,
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
