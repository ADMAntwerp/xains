"""NarrativeGenerator abstraction + concrete generators."""

from xain.generation.base import GenerationResult, NarrativeGenerator
from xain.generation.llm import LLMNarrativeGenerator
from xain.generation.templated import (
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
