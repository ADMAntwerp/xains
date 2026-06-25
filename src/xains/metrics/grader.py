"""Grader: integrates fidelity metrics into ExtractionGrades / CounterfactualGrades."""

from pydantic import BaseModel, ConfigDict

from xains.guardrails.types import CounterfactualExtraction, NarrativeExtraction
from xains.metrics.counterfactual_fidelity import (
    cf_coverage,
    change_fidelity,
    invented_features,
)
from xains.metrics.coverage import coverage, hallucination_count
from xains.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xains.schema import DatasetSchema
from xains.types import TabularExplanationRequest


class ExtractionGrades(BaseModel):
    """Aggregate of all per-extraction metrics (feature-importance path)."""

    model_config = ConfigDict(extra="forbid")

    sign_faithfulness: float | None = None
    value_faithfulness: float | None = None
    rank_correlation: float | None = None
    coverage: float
    hallucination_count: int
    prompt_version: str


EXTRACTION_GRADE_DIRECTIONS: dict[str, str] = {
    "sign_faithfulness": "↑",
    "value_faithfulness": "↑",
    "rank_correlation": "↑",
    "coverage": "↑",
    "hallucination_count": "↓",
}


def grade_extraction(
    extraction: NarrativeExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
    narrative_text: str,
    k: int = 10,
) -> ExtractionGrades:
    """Compute verbalization-fidelity metrics for ``extraction``."""
    return ExtractionGrades(
        sign_faithfulness=sign_faithfulness(extraction, request),
        value_faithfulness=value_faithfulness(extraction, request),
        rank_correlation=rank_correlation(extraction, request),
        coverage=coverage(extraction, schema, k=k),
        hallucination_count=hallucination_count(extraction),
        prompt_version=extraction.prompt_version,
    )


# ========================================================================
# Counterfactual fidelity grader (ADR 0032)
# ========================================================================


class CounterfactualGrades(BaseModel):
    """Aggregate of all per-extraction metrics (counterfactual path).

    Flat aggregate, NOT inheriting any base. The abstract base across
    grade aggregates is deferred until a third concrete case (the hybrid
    feature-importance + counterfactual mode) lands. See ADR 0032.
    """

    model_config = ConfigDict(extra="forbid")

    change_fidelity: float | None = None
    coverage: float
    invented_features: int
    prompt_version: str


COUNTERFACTUAL_GRADE_DIRECTIONS: dict[str, str] = {
    "change_fidelity": "↑",
    "coverage": "↑",
    "invented_features": "↓",
}


def grade_counterfactual(
    extraction: CounterfactualExtraction,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> CounterfactualGrades:
    """Compute counterfactual-fidelity metrics for ``extraction``.

    Pure composition over :func:`change_fidelity`, :func:`cf_coverage`,
    :func:`invented_features`; no LLM call (extraction already happened
    upstream via :func:`extract_counterfactual_claims`).
    """
    return CounterfactualGrades(
        change_fidelity=change_fidelity(extraction, request, schema),
        coverage=cf_coverage(extraction, request, schema),
        invented_features=invented_features(extraction),
        prompt_version=extraction.prompt_version,
    )
