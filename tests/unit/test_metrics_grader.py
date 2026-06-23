"""Unit tests for grade_extraction and ExtractionGrades.

The grader is the integration point for verbalization-fidelity metrics:
pure data in, grades out. Perplexity is a narrativity concept (ADR 0008,
ADR 0023) and lives in NarrativityGrades, not here.
"""

import sys
from typing import Any

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    ExtractionGrades,
    FeatureClaim,
    FeatureSchema,
    Modality,
    NarrativeExtraction,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.metrics import grade_extraction

# ------------------------------------------------------ helpers


def _request(*contribs: tuple[str, Any, float]) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={name: val for name, val, _ in contribs},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name=name, value=val, importance=imp) for name, val, imp in contribs
        ],
    )


def _schema(*names: str) -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="test",
        description="test",
        target=TargetSchema(name="t", description="d", classes={0: "A", 1: "B"}),
        features=[FeatureSchema(name=n, dtype="numeric", description=f"feat {n}") for n in names],
    )


def _claim(name: str, *, rank: int, sign: int, value: Any = None) -> FeatureClaim:
    return FeatureClaim(
        rank=rank,
        sign=sign,
        value=value,
        narrative_name=name,
        resolved_to=name,
    )


def _extraction(
    features: dict[str, FeatureClaim],
    hallucinations: list[FeatureClaim] | None = None,
    prompt_version: str = "2",
) -> NarrativeExtraction:
    return NarrativeExtraction(
        features=features,
        hallucinations=hallucinations or [],
        prompt_version=prompt_version,
        model_name="test",
    )


# ------------------------------------------------------ tests


def test_grade_extraction_populates_all_fields_for_complete_extraction() -> None:
    request = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    schema = _schema("dti", "age")
    extraction = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1, value=0.41),
            "age": _claim("age", rank=2, sign=-1, value=29),
        }
    )
    narrative = "Higher dti pushed the applicant toward default. Younger age slightly offset this."
    grades = grade_extraction(
        extraction,
        request,
        schema,
        narrative_text=narrative,
    )
    assert isinstance(grades, ExtractionGrades)
    assert grades.sign_faithfulness == 1.0
    assert grades.value_faithfulness == 1.0
    assert grades.rank_correlation is not None
    assert grades.coverage == 1.0  # 2 of min(10, 2) = 2.
    assert grades.hallucination_count == 0
    assert grades.readability is not None


def test_grade_extraction_with_no_resolved_features_sets_fidelity_metrics_none() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={})
    grades = grade_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    # Fidelity metrics are undefined when there's nothing to compare.
    assert grades.sign_faithfulness is None
    assert grades.value_faithfulness is None
    assert grades.rank_correlation is None
    # Coverage and hallucination_count are always defined.
    assert grades.coverage == 0.0
    assert grades.hallucination_count == 0


def test_grade_extraction_forwards_prompt_version() -> None:
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(
        features={"dti": _claim("dti", rank=1, sign=1)},
        prompt_version="2",
    )
    grades = grade_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    assert grades.prompt_version == "2"


def test_extraction_grades_rejects_extra_fields() -> None:
    """ExtractionGrades has ConfigDict(extra='forbid')."""
    with pytest.raises(ValidationError):
        ExtractionGrades(
            sign_faithfulness=None,
            value_faithfulness=None,
            rank_correlation=None,
            coverage=0.0,
            hallucination_count=0,
            readability=None,
            prompt_version="2",
            unknown_extra="bogus",  # type: ignore[call-arg]
        )


def test_extraction_grades_rejects_perplexity_field() -> None:
    """Per ADR 0023, perplexity belongs on NarrativityGrades only."""
    with pytest.raises(ValidationError):
        ExtractionGrades(
            sign_faithfulness=None,
            value_faithfulness=None,
            rank_correlation=None,
            coverage=0.0,
            hallucination_count=0,
            readability=None,
            perplexity=None,  # type: ignore[call-arg]
            prompt_version="2",
        )


def test_grade_extraction_does_not_accept_perplexity_provider_kwarg() -> None:
    """Per ADR 0023, the perplexity_provider parameter is removed."""
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    with pytest.raises(TypeError):
        grade_extraction(
            extraction,
            request,
            schema,
            narrative_text="Some narrative text.",
            perplexity_provider=object(),  # type: ignore[call-arg]
        )


def test_grade_extraction_handles_missing_textstat_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When textstat is unavailable, readability=None but other metrics are populated."""
    monkeypatch.setitem(sys.modules, "textstat", None)
    request = _request(("dti", 0.41, 0.37))
    schema = _schema("dti")
    extraction = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    grades = grade_extraction(extraction, request, schema, narrative_text="Some narrative text.")
    assert grades.readability is None
    assert grades.coverage == 1.0
    assert grades.hallucination_count == 0
    assert grades.prompt_version == "2"
