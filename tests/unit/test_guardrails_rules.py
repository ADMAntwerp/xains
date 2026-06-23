"""Unit tests for the rule-based guardrails.

Only ``class_name_mentioned`` remains as a rule-based check: strict, exact
case-insensitive substring of the label string, no single-word fallback, no
fuzzy matching, modality-agnostic. (See ADR 0006 for why a rule-based
feature-invention check was dropped.)
"""

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TargetSchema,
    TextSpec,
)
from xains.guardrails import GuardrailResult, class_name_mentioned

# ------------------------------------------------------ fixtures


@pytest.fixture
def tabular_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="24-month default on personal loans.",
        target=TargetSchema(
            name="default",
            description="Whether the applicant defaulted.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="age", dtype="numeric", description="Applicant age."),
            FeatureSchema(name="dti", dtype="numeric", description="Debt-to-income ratio."),
        ],
    )


# ---------------------------------------------- class_name_mentioned


def test_class_name_mentioned_passes_on_exact_substring(tabular_schema: DatasetSchema) -> None:
    text = "This applicant Defaulted within the window."
    pred = Prediction(predicted_class=1)
    result = class_name_mentioned(text, tabular_schema, pred)
    assert result.name == "class_name_mentioned"
    assert result.severity == "failure"
    assert result.passed is True


def test_class_name_mentioned_passes_case_insensitive(tabular_schema: DatasetSchema) -> None:
    text = "This applicant defaulted within the window."
    pred = Prediction(predicted_class=1)
    result = class_name_mentioned(text, tabular_schema, pred)
    assert result.passed is True


def test_class_name_mentioned_fails_when_label_absent(tabular_schema: DatasetSchema) -> None:
    text = "The outcome is positive and the model is confident."
    pred = Prediction(predicted_class=1)  # label = "Defaulted"
    result = class_name_mentioned(text, tabular_schema, pred)
    assert result.passed is False
    assert result.severity == "failure"


def test_class_name_mentioned_strict_multiword_match() -> None:
    """Multi-word label must appear as a substring in full — no single-word fallback."""
    schema = DatasetSchema(
        modality=Modality.TABULAR,
        name="loan",
        description="desc",
        target=TargetSchema(
            name="decision",
            description="d",
            classes={0: "Denied", 1: "Approved for loan"},
        ),
        features=[FeatureSchema(name="age", dtype="numeric", description="years")],
    )
    text = "The applicant was approved."  # does NOT contain "approved for loan"
    pred = Prediction(predicted_class=1)
    result = class_name_mentioned(text, schema, pred)
    assert result.passed is False


def test_class_name_mentioned_details_include_expected_label(
    tabular_schema: DatasetSchema,
) -> None:
    text = "Nothing here."
    pred = Prediction(predicted_class=1)
    result = class_name_mentioned(text, tabular_schema, pred)
    assert result.details["expected_label"] == "Defaulted"
    assert result.details["predicted_class"] == 1


def test_class_name_mentioned_modality_agnostic() -> None:
    """Same function works for non-tabular requests — it only uses schema + prediction."""
    schema = DatasetSchema(
        modality=Modality.TEXT,
        name="sentiment",
        description="reviews",
        target=TargetSchema(
            name="polarity",
            description="p",
            classes={"pos": "Positive", "neg": "Negative"},
        ),
        text_spec=TextSpec(language="en"),
    )
    text = "The review is strongly Positive throughout."
    pred = Prediction(predicted_class="pos")
    result = class_name_mentioned(text, schema, pred)
    assert result.passed is True


# --------------------------------------------------- GuardrailResult


def test_guardrail_result_valid_model() -> None:
    r = GuardrailResult(
        name="test",
        severity="advisory",
        passed=True,
        details={"k": "v"},
    )
    assert r.name == "test"
    assert r.severity == "advisory"
    assert r.passed is True


def test_guardrail_result_rejects_unknown_severity() -> None:
    with pytest.raises(ValidationError):
        GuardrailResult(name="x", severity="warning", passed=True)
