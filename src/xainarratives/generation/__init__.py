"""NarrativeGenerator abstraction + concrete generators."""

from xainarratives.generation.base import GenerationResult, NarrativeGenerator
from xainarratives.generation.llm import LLMNarrativeGenerator

__all__ = ["GenerationResult", "LLMNarrativeGenerator", "NarrativeGenerator"]
