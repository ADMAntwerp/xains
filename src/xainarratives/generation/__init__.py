"""NarrativeGenerator abstraction + concrete generators."""

from xainarratives.generation.base import GenerationResult, NarrativeGenerator
from xainarratives.generation.llm import LLMNarrativeGenerator
from xainarratives.generation.templated import (
    DEFAULT_CLAUSE_TEMPLATE,
    DEFAULT_LEAD_TEMPLATE,
    TemplatedNarrativeGenerator,
)

__all__ = [
    "DEFAULT_CLAUSE_TEMPLATE",
    "DEFAULT_LEAD_TEMPLATE",
    "GenerationResult",
    "LLMNarrativeGenerator",
    "NarrativeGenerator",
    "TemplatedNarrativeGenerator",
]
