"""Unit tests for TemplatedCounterfactualGenerator (ADR 0030).

LLM-free CF narrative. Deterministic prose; exact-string asserts.
"""

from typing import Any

import pytest

from xains import (
    DatasetSchema,
    ExplanationConfig,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
    TemplatedCounterfactualGenerator,
)
from xains.types import TabularCounterfactual, TextExplanationRequest, TokenContribution


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


def _config() -> ExplanationConfig:
    return ExplanationConfig(mode="counterfactual")


def _cf(
    features: dict[str, Any],
    *,
    predicted_class: int = 0,
    method: str | None = None,
) -> TabularCounterfactual:
    return TabularCounterfactual(
        predicted_class=predicted_class,
        features=features,
        method=method,
    )


def _request_with(cfs: list[TabularCounterfactual]) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=cfs,
    )


# ------------------------------------------------------ prose shape


def test_single_cf_single_change_exact_prose() -> None:
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.20})])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "dti would need to change from 0.41 to 0.2."
    )


def test_single_cf_two_changes_uses_oxford_comma_and() -> None:
    req = _request_with([_cf({"age": 35, "salary": 52000, "dti": 0.20})])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "age would need to change from 29 to 35, and "
        "dti would need to change from 0.41 to 0.2."
    )


def test_single_cf_three_changes_uses_oxford_comma_and() -> None:
    req = _request_with([_cf({"age": 35, "salary": 80000, "dti": 0.20})])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "age would need to change from 29 to 35, "
        "salary would need to change from 52000 to 80000, and "
        "dti would need to change from 0.41 to 0.2."
    )


def test_multiple_cfs_subsequent_sentences_start_with_alternatively() -> None:
    cfs = [
        _cf({"age": 29, "salary": 52000, "dti": 0.20}),
        _cf({"age": 35, "salary": 52000, "dti": 0.41}),
    ]
    result = TemplatedCounterfactualGenerator().generate(_request_with(cfs), _schema(), _config())
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "dti would need to change from 0.41 to 0.2. "
        "Alternatively, age would need to change from 29 to 35."
    )


# ------------------------------------------------------ method provenance


def test_method_off_by_default_never_appears() -> None:
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.20}, method="DiCE")])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert "DiCE" not in result.text
    assert "method" not in result.text


def test_method_shown_when_include_method_true_and_cf_method_set() -> None:
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.20}, method="DiCE")])
    result = TemplatedCounterfactualGenerator(include_method=True).generate(
        req, _schema(), _config()
    )
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "dti would need to change from 0.41 to 0.2 (method: DiCE)."
    )


def test_method_omitted_when_include_method_true_but_cf_method_none() -> None:
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.20}, method=None)])
    result = TemplatedCounterfactualGenerator(include_method=True).generate(
        req, _schema(), _config()
    )
    assert "method" not in result.text


# ------------------------------------------------------ degenerate / error paths


def test_identical_cf_renders_no_change_sentence() -> None:
    """CF identical to factual: no changes to describe, but valid scenario."""
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.41})])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, no feature changes were detected."
    )


def test_non_tabular_request_raises_type_error() -> None:
    txt_req = TextExplanationRequest(
        text="some text",
        prediction=Prediction(predicted_class=1),
        contributions=[TokenContribution(token="x", span=(0, 1), importance=0.5)],
    )
    with pytest.raises(TypeError, match="TabularExplanationRequest"):
        TemplatedCounterfactualGenerator().generate(txt_req, _schema(), _config())


def test_counterfactuals_none_raises_value_error() -> None:
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactuals=None,
    )
    with pytest.raises(ValueError, match=r"counterfactuals"):
        TemplatedCounterfactualGenerator().generate(req, _schema(), _config())


# ------------------------------------------------------ GenerationResult shape


def test_generation_result_has_text_and_latency_only() -> None:
    req = _request_with([_cf({"age": 29, "salary": 52000, "dti": 0.20})])
    result = TemplatedCounterfactualGenerator().generate(req, _schema(), _config())
    assert isinstance(result.text, str) and result.text
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    # LLM-only fields are None for templated path (ADR 0019 contract).
    assert result.prompt is None
    assert result.model_name is None
    assert result.raw_llm_response is None
    assert result.tokens_used is None
    assert result.guardrails is None
