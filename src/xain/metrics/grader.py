"""Grader: integrates all metrics into a single ExtractionGrades record."""

from pydantic import BaseModel, ConfigDict

from xain.guardrails.types import NarrativeExtraction
from xain.metrics.coverage import coverage, hallucination_count
from xain.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xain.metrics.narrativity import readability
from xain.metrics.perplexity import PerplexityProvider
from xain.schema import DatasetSchema
from xain.types import TabularExplanationRequest


class ExtractionGrades(BaseModel):
    """Aggregate of all per-extraction metrics."""

    model_config = ConfigDict(extra="forbid")

    sign_faithfulness: float | None = None
    value_faithfulness: float | None = None
    rank_correlation: float | None = None
    coverage: float
    hallucination_count: int
    readability: float | None = None
    perplexity: float | None = None
    prompt_version: str


def grade_extraction(
    extraction: NarrativeExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
    narrative_text: str,
    k: int = 10,
    perplexity_provider: PerplexityProvider | None = None,
) -> ExtractionGrades:
    """Compute all metrics for ``extraction``.

    ``perplexity_provider`` is optional; when ``None``, ``perplexity`` is
    left at ``None`` (no provider call). Callers that want a perplexity
    score supply a concrete ``PerplexityProvider``.
    """
    perplexity_value: float | None = None
    if perplexity_provider is not None:
        perplexity_value = perplexity_provider.compute(narrative_text)

    try:
        readability_value = readability(extraction, narrative_text)
    except ImportError:
        readability_value = None

    return ExtractionGrades(
        sign_faithfulness=sign_faithfulness(extraction, request),
        value_faithfulness=value_faithfulness(extraction, request),
        rank_correlation=rank_correlation(extraction, request),
        coverage=coverage(extraction, schema, k=k),
        hallucination_count=hallucination_count(extraction),
        readability=readability_value,
        perplexity=perplexity_value,
        prompt_version=extraction.prompt_version,
    )
