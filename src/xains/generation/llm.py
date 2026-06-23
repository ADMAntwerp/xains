"""LLMNarrativeGenerator - render-prompt + LLM call + class_name_mentioned."""

import time

from xains.config import ExplanationConfig
from xains.generation.base import GenerationResult, NarrativeGenerator
from xains.guardrails import class_name_mentioned
from xains.guardrails.types import GuardrailResult
from xains.prompts.base import PromptTemplate
from xains.providers.base import LLMProvider
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest


class LLMNarrativeGenerator(NarrativeGenerator):
    """Generate by rendering a prompt and calling an LLM.

    Runs the ``class_name_mentioned`` guardrail on the LLM output (gated
    by ``config.run_guardrails``). Other post-generation activity -
    judge-based narrative extraction, grading - stays in ``Explainer``
    because it depends only on the produced text, not on how it was
    produced.
    """

    def __init__(self, *, prompt_template: PromptTemplate, llm: LLMProvider) -> None:
        self._prompt_template = prompt_template
        self._llm = llm

    @property
    def prompt_template(self) -> PromptTemplate:
        """The prompt template this generator renders (read-only)."""
        return self._prompt_template

    def generate(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> GenerationResult:
        system, user = self._prompt_template.render(request, schema, config)

        start = time.perf_counter()
        response = self._llm.generate(system, user)
        latency_ms = (time.perf_counter() - start) * 1000.0

        guardrails: list[GuardrailResult] | None = None
        if config.run_guardrails:
            guardrails = [class_name_mentioned(response.text, schema, request.prediction)]

        return GenerationResult(
            text=response.text,
            prompt=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
            model_name=response.model_name,
            raw_llm_response=response.text,
            tokens_used=response.tokens_used,
            latency_ms=latency_ms,
            guardrails=guardrails,
        )
