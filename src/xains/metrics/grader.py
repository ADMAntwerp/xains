"""Grader: integrates verbalization-fidelity metrics into ExtractionGrades."""

from pydantic import BaseModel, ConfigDict

from xains.guardrails.types import NarrativeExtraction
from xains.metrics.coverage import coverage, hallucination_count
from xains.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xains.metrics.narrativity import readability
from xains.schema import DatasetSchema
from xains.types import TabularExplanationRequest


class ExtractionGrades(BaseModel):
    """Aggregate of all per-extraction metrics."""

    model_config = ConfigDict(extra="forbid")

    sign_faithfulness: float | None = None
    value_faithfulness: float | None = None
    rank_correlation: float | None = None
    coverage: float
    hallucination_count: int
    readability: float | None = None
    prompt_version: str


def grade_extraction(
    extraction: NarrativeExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
    narrative_text: str,
    k: int = 10,
) -> ExtractionGrades:
    """Compute verbalization-fidelity metrics for ``extraction``."""
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
        prompt_version=extraction.prompt_version,
    )
