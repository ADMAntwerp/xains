"""NarrativeGenerator ABC + GenerationResult dataclass.

A NarrativeGenerator produces a narrative (plus audit metadata) from a
request + schema + config. Post-generation activities that don't depend
on HOW the text was produced (judge-based narrative extraction, grading)
live in ``Explainer`` and consume only the produced text.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from xain.config import ExplanationConfig
from xain.guardrails.types import GuardrailResult
from xain.schema import DatasetSchema
from xain.types import ExplanationRequest


@dataclass(frozen=True)
class GenerationResult:
    """Output of ``NarrativeGenerator.generate()``.

    The LLM path populates all fields. A future templated path leaves
    ``prompt``, ``model_name``, ``raw_llm_response``, and ``tokens_used``
    as ``None`` - those fields describe the LLM call specifically, not
    "what was generated".
    """

    text: str
    prompt: str | None
    model_name: str | None
    raw_llm_response: str | None
    tokens_used: dict[str, int] | None
    latency_ms: float | None
    guardrails: list[GuardrailResult] | None


class NarrativeGenerator(ABC):
    """Produce a narrative (plus audit metadata) from a request."""

    @abstractmethod
    def generate(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> GenerationResult:
        """Return the generated text and the audit envelope around it."""
