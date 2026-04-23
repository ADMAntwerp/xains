"""Explainer — the sync orchestrator.

Responsibilities:

1. Validate that the request's modality matches the schema's modality.
2. Resolve ``factual`` / ``contrastive`` / ``counterfactual`` from config + request.
3. Render the prompt via the configured ``PromptTemplate``.
4. Call the configured ``LLMProvider``.
5. Package the result with audit metadata.

Not in v0: guardrails, caching, retries, streaming, batch, eval pipeline.
Each gets its own PR and its own ADR.
"""

import time
import warnings

from xainarratives.config import ExplanationConfig, ExplanationModeOrAuto
from xainarratives.prompts.base import PromptTemplate
from xainarratives.providers.base import LLMProvider
from xainarratives.schema import DatasetSchema
from xainarratives.types import ExplanationMode, ExplanationRequest, ExplanationResult


class Explainer:
    """Generate natural-language explanations from pre-computed attributions."""

    def __init__(
        self,
        schema: DatasetSchema,
        llm: LLMProvider,
        prompt_template: PromptTemplate,
        config: ExplanationConfig | None = None,
    ) -> None:
        self.schema = schema
        self.llm = llm
        self.prompt_template = prompt_template
        self.config = config or ExplanationConfig()

    def explain(self, request: ExplanationRequest) -> ExplanationResult:
        self._validate_modality(request)
        self._warn_if_counterfactual_does_not_flip(request)

        mode = self._resolve_mode(request)
        system, user = self.prompt_template.render(request, self.schema, self.config)

        start = time.perf_counter()
        response = self.llm.generate(system, user)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ExplanationResult(
            text=response.text,
            mode=mode,
            prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
            raw_llm_response=response.text,
            model_name=response.model_name,
            tokens_used=response.tokens_used,
            latency_ms=latency_ms,
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

    def _resolve_mode(self, request: ExplanationRequest) -> ExplanationMode:
        has_cf = bool(request.counterfactuals)
        has_contrast = request.contrast_class is not None

        if self.config.mode == "auto":
            if has_cf:
                return "counterfactual"
            if has_contrast:
                return "contrastive"
            return "factual"

        # Explicit mode: verify the request carries its required inputs.
        self._check_explicit_mode(self.config.mode, has_cf=has_cf, has_contrast=has_contrast)
        return self.config.mode

    @staticmethod
    def _check_explicit_mode(
        mode: ExplanationModeOrAuto, *, has_cf: bool, has_contrast: bool
    ) -> None:
        if mode == "counterfactual" and not has_cf:
            raise ValueError("config.mode='counterfactual' requires request.counterfactuals.")
        if mode == "contrastive" and not has_contrast:
            raise ValueError("config.mode='contrastive' requires request.contrast_class.")
        # "factual" has no prerequisites; "auto" is handled by the caller.

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
