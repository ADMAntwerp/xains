"""NarrativeGenerator abstraction + concrete generators."""

from xains.generation.base import GenerationResult, NarrativeGenerator
from xains.generation.llm import LLMNarrativeGenerator
from xains.generation.templated import (
    DEFAULT_CLAUSE_TEMPLATE,
    DEFAULT_LEAD_TEMPLATE,
    TemplatedNarrativeGenerator,
)
from xains.generation.templated_counterfactual import TemplatedCounterfactualGenerator
from xains.generation.templated_hybrid import TemplatedHybridGenerator

__all__ = [
    "DEFAULT_CLAUSE_TEMPLATE",
    "DEFAULT_LEAD_TEMPLATE",
    "GenerationResult",
    "LLMNarrativeGenerator",
    "NarrativeGenerator",
    "TemplatedCounterfactualGenerator",
    "TemplatedHybridGenerator",
    "TemplatedNarrativeGenerator",
]
