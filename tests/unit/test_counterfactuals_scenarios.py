"""Unit tests for build_scenarios and CounterfactualScenario (ADR 0030 + ADR 0031).

Per ADR 0031 a request carries a single counterfactual; build_scenarios
returns a single CounterfactualScenario (not a list).
"""

from typing import Any

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.counterfactuals import (
    ChangedFeature,
    CounterfactualScenario,
    build_scenarios,
)
from xains.types import TabularCounterfactual, TextCounterfactual


def _schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="Credit risk demo.",
        target=TargetSchema(
            name="default",
            description="Whether the applicant defaulted.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="age", dtype="numeric", unit="years", description="age"),
            FeatureSchema(name="salary", dtype="numeric", unit="EUR", description="salary"),
            FeatureSchema(name="dti", dtype="numeric", description="debt-to-income"),
        ],
    )


def _cf(
    features: dict[str, Any],
    *,
    predicted_class: int = 0,
    method: str | None = None,
    changed: list[str] | None = None,
) -> TabularCounterfactual:
    return TabularCounterfactual(
        predicted_class=predicted_class,
        features=features,
        changed_features=changed,
        method=method,
    )


def _request_with(cf: TabularCounterfactual) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=cf,
    )


def test_build_scenarios_returns_single_scenario() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    scenario = build_scenarios(req, schema)
    assert isinstance(scenario, CounterfactualScenario)


def test_factual_and_cf_labels_mapped_from_schema_target_classes() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    scenario = build_scenarios(req, schema)
    assert scenario.factual_label == "Defaulted"
    assert scenario.cf_label == "Repaid"


def test_method_passthrough() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, method="DiCE"))
    assert build_scenarios(req, schema).method == "DiCE"

    req_none = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, method=None))
    assert build_scenarios(req_none, schema).method is None


def test_changes_are_changed_feature_instances() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    scenario = build_scenarios(req, schema)
    assert scenario.changes == [
        ChangedFeature(name="salary", before=52000, after=80000),
    ]


def test_counterfactual_none_raises_value_error() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=None,
    )
    with pytest.raises(ValueError, match=r"counterfactual"):
        build_scenarios(req, schema)


def test_factual_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=99),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=_cf({"age": 29, "salary": 80000, "dti": 0.41}),
    )
    with pytest.raises(ValueError, match=r"99"):
        build_scenarios(req, schema)


def test_cf_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, predicted_class=99))
    with pytest.raises(ValueError, match=r"99"):
        build_scenarios(req, schema)


def test_non_tabular_counterfactual_raises_type_error() -> None:
    schema = _schema()
    txt_cf = TextCounterfactual(predicted_class=0, text="counterfactual text")
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=txt_cf,
    )
    with pytest.raises(TypeError, match="tabular"):
        build_scenarios(req, schema)


def test_changed_feature_name_not_in_schema_raises() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41, "mystery": 1},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.41, "mystery": 7}),
    )
    with pytest.raises(ValueError, match=r"mystery"):
        build_scenarios(req, schema)


def test_identical_cf_produces_scenario_with_empty_changes() -> None:
    """A degenerate CF that doesn't actually change anything still yields a scenario."""
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 52000, "dti": 0.41}))  # identical
    scenario = build_scenarios(req, schema)
    assert scenario.changes == []
    assert scenario.factual_label == "Defaulted"
    assert scenario.cf_label == "Repaid"


def test_scenario_rejects_extra_fields() -> None:
    """CounterfactualScenario has ConfigDict(extra='forbid'), house style."""
    with pytest.raises(ValidationError):
        CounterfactualScenario(
            factual_label="A",
            cf_label="B",
            changes=[],
            method=None,
            extra="bogus",  # type: ignore[call-arg]
        )
