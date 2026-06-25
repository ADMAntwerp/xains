"""Unit tests for grade_counterfactual + CounterfactualGrades (ADR 0032).

Mirrors test_metrics_grader.py for the counterfactual path. Pure
composition over change_fidelity / cf_coverage / invented_features; no
LLM call.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from xains import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    CounterfactualGrades,
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.metrics import COUNTERFACTUAL_GRADE_DIRECTIONS, grade_counterfactual
from xains.types import TabularCounterfactual


def _schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="Credit risk demo.",
        target=TargetSchema(
            name="default",
            description="Default outcome.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="age", dtype="numeric", unit="years", description="age"),
            FeatureSchema(name="dti", dtype="numeric", description="dti"),
        ],
    )


def _request(
    features: dict[str, Any],
    cf_features: dict[str, Any],
    cf_predicted: int = 0,
) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features=features,
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(
                name=next(iter(features)), value=next(iter(features.values())), importance=0.1
            )
        ],
        counterfactual=TabularCounterfactual(predicted_class=cf_predicted, features=cf_features),
    )


def _extraction(
    changes: dict[str, CounterfactualFeatureClaim] | None = None,
    invented: list[CounterfactualFeatureClaim] | None = None,
    prompt_version: str = "1",
) -> CounterfactualExtraction:
    return CounterfactualExtraction(
        changes=changes or {},
        invented=invented or [],
        prompt_version=prompt_version,
        model_name="test",
    )


def _claim(
    name: str, *, stated_before: Any = None, stated_after: Any = None
) -> CounterfactualFeatureClaim:
    return CounterfactualFeatureClaim(
        narrative_name=name,
        resolved_to=name,
        stated_before=stated_before,
        stated_after=stated_after,
    )


# ====================================================== grade_counterfactual


def test_grade_counterfactual_composes_three_metrics_correctly() -> None:
    """Known extraction + ground truth -> known grades on all three axes."""
    schema = _schema()
    # CF changes age AND dti.
    req = _request({"age": 29, "dti": 0.41}, {"age": 35, "dti": 0.20})
    extraction = _extraction(
        changes={
            # dti correct on both sides.
            "dti": _claim("dti", stated_before=0.41, stated_after=0.20),
            # age before wrong -> incorrect.
            "age": _claim("age", stated_before=99, stated_after=35),
        },
        invented=[
            CounterfactualFeatureClaim(narrative_name="credit_history", resolved_to=None),
        ],
        prompt_version="1",
    )
    grades = grade_counterfactual(extraction, req, schema)

    assert isinstance(grades, CounterfactualGrades)
    # 1 of 2 resolved claims correct -> 0.5
    assert grades.change_fidelity == 0.5
    # 2 of 2 changed features resolved -> 1.0
    assert grades.coverage == 1.0
    # 1 invented mention
    assert grades.invented_features == 1
    assert grades.prompt_version == "1"


def test_grade_counterfactual_change_fidelity_none_when_undefined() -> None:
    """No resolved claim about an actually-changed feature -> change_fidelity = None."""
    schema = _schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})  # only dti changes
    extraction = _extraction(
        changes={
            # Only claim is about age, which did NOT change -> ignored by change_fidelity.
            "age": _claim("age", stated_before=29, stated_after=99),
        },
    )
    grades = grade_counterfactual(extraction, req, schema)
    assert grades.change_fidelity is None
    # Coverage is 0/1 (dti not mentioned), invented_features 0.
    assert grades.coverage == 0.0
    assert grades.invented_features == 0


def test_grade_counterfactual_forwards_prompt_version() -> None:
    schema = _schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction(prompt_version="42")
    grades = grade_counterfactual(extraction, req, schema)
    assert grades.prompt_version == "42"


def test_grade_counterfactual_returns_correct_type() -> None:
    schema = _schema()
    req = _request({"age": 29, "dti": 0.41}, {"age": 29, "dti": 0.20})
    extraction = _extraction()
    grades = grade_counterfactual(extraction, req, schema)
    assert isinstance(grades, CounterfactualGrades)


# ====================================================== CounterfactualGrades shape


def test_counterfactual_grades_rejects_extra_fields() -> None:
    """CounterfactualGrades has ConfigDict(extra='forbid'), house style."""
    with pytest.raises(ValidationError):
        CounterfactualGrades(
            change_fidelity=1.0,
            coverage=1.0,
            invented_features=0,
            prompt_version="1",
            extra="bogus",  # type: ignore[call-arg]
        )


def test_counterfactual_grades_accepts_none_change_fidelity() -> None:
    """change_fidelity is float | None; coverage / invented_features are not."""
    grades = CounterfactualGrades(
        change_fidelity=None,
        coverage=0.5,
        invented_features=2,
        prompt_version="1",
    )
    assert grades.change_fidelity is None


# ====================================================== COUNTERFACTUAL_GRADE_DIRECTIONS


def test_directions_dict_has_correct_arrows() -> None:
    assert COUNTERFACTUAL_GRADE_DIRECTIONS == {
        "change_fidelity": "↑",
        "coverage": "↑",
        "invented_features": "↓",
    }


def test_directions_dict_excludes_prompt_version() -> None:
    """prompt_version is metadata, not a grade - mirror EXTRACTION_GRADE_DIRECTIONS."""
    assert "prompt_version" not in COUNTERFACTUAL_GRADE_DIRECTIONS
