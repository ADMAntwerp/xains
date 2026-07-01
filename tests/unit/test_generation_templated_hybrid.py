"""Unit tests for TemplatedHybridGenerator (ADR 0037).

The hybrid generator composes the feature-importance and counterfactual
templated narratives into a two-section text (FI paragraph, blank line, CF
paragraph). Deterministic, no LLM. Requires both contributions and a
counterfactual on the request.
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
    TemplatedHybridGenerator,
    TemplatedNarrativeGenerator,
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
    return ExplanationConfig(mode="feature_importance_counterfactual")


def _cf(features: dict[str, Any], *, predicted_class: int = 0) -> TabularCounterfactual:
    return TabularCounterfactual(predicted_class=predicted_class, features=features)


def _request_with(cf: TabularCounterfactual) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=cf,
    )


# ------------------------------------------------------ composition


def test_hybrid_text_is_fi_then_blank_line_then_cf() -> None:
    """The hybrid text equals the FI narrative, a blank line, then the CF narrative."""
    req = _request_with(_cf({"age": 29, "salary": 52000, "dti": 0.20}))
    schema, config = _schema(), _config()

    fi_text = TemplatedNarrativeGenerator().generate(req, schema, config).text
    cf_text = TemplatedCounterfactualGenerator().generate(req, schema, config).text

    result = TemplatedHybridGenerator().generate(req, schema, config)
    assert result.text == f"{fi_text}\n\n{cf_text}"


def test_hybrid_sections_both_present() -> None:
    req = _request_with(_cf({"age": 29, "salary": 52000, "dti": 0.20}))
    result = TemplatedHybridGenerator().generate(req, _schema(), _config())
    # FI section names the prediction; CF section names the flip.
    assert "Defaulted" in result.text
    assert "To change the prediction from Defaulted to Repaid" in result.text
    assert "\n\n" in result.text


def test_hybrid_is_deterministic() -> None:
    req = _request_with(_cf({"age": 35, "salary": 52000, "dti": 0.20}))
    schema, config = _schema(), _config()
    a = TemplatedHybridGenerator().generate(req, schema, config).text
    b = TemplatedHybridGenerator().generate(req, schema, config).text
    assert a == b


def test_hybrid_result_is_llm_metadata_free() -> None:
    """Templated path leaves LLM-specific fields None."""
    req = _request_with(_cf({"age": 29, "salary": 52000, "dti": 0.20}))
    result = TemplatedHybridGenerator().generate(req, _schema(), _config())
    assert result.prompt is None
    assert result.model_name is None
    assert result.raw_llm_response is None
    assert result.tokens_used is None


# ------------------------------------------------------ guards


def test_hybrid_non_tabular_request_raises_type_error() -> None:
    req = TextExplanationRequest(
        text="some text",
        prediction=Prediction(predicted_class=1),
        contributions=[TokenContribution(token="some", importance=0.5, span=(0, 4))],
    )
    with pytest.raises(TypeError, match="TabularExplanationRequest"):
        TemplatedHybridGenerator().generate(req, _schema(), _config())


def test_hybrid_missing_counterfactual_raises_value_error() -> None:
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
    )
    with pytest.raises(ValueError, match="counterfactual"):
        TemplatedHybridGenerator().generate(req, _schema(), _config())
