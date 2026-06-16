"""Direct unit tests for TemplatedNarrativeGenerator.generate().

NET-NEW for the templated (LLM-free) narrative path. The 8 tests pin
down: byte-exact default output, method= injection, ordinal phrasing,
ranking by absolute importance, GenerationResult shape (LLM fields
None), Explainer integration with LLM-free generation but LLM-graded
extraction, editable clause_template, and the strict-ValueError on a
contribution naming a feature missing from request.features.
"""

import json

import pytest

from xainarratives import (
    DatasetSchema,
    Explainer,
    ExplanationConfig,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xainarratives.generation import GenerationResult, TemplatedNarrativeGenerator
from xainarratives.providers import MockLLMProvider


def _credit_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TABULAR,
        name="credit_risk",
        description="24-month default.",
        target=TargetSchema(
            name="default",
            description="Whether the applicant defaulted.",
            classes={0: "Repaid", 1: "Defaulted"},
        ),
        features=[
            FeatureSchema(name="dti", dtype="numeric", description="Debt-to-income ratio."),
            FeatureSchema(name="age", dtype="numeric", description="Applicant age."),
            FeatureSchema(name="income", dtype="numeric", description="Annual income."),
            FeatureSchema(name="employment_years", dtype="numeric", description="Years employed."),
            FeatureSchema(
                name="num_credit_lines", dtype="numeric", description="Open credit lines."
            ),
        ],
    )


def _credit_request() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={
            "dti": 0.41,
            "age": 29,
            "income": 52000,
            "employment_years": 3,
            "num_credit_lines": 5,
        },
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
            TabularContribution(name="income", value=52000, importance=0.08),
            TabularContribution(name="employment_years", value=3, importance=-0.05),
            TabularContribution(name="num_credit_lines", value=5, importance=0.03),
        ],
    )


def test_default_output_byte_exact_method_agnostic() -> None:
    """Top-5 default output: method word omitted; clauses for all 5 features."""
    generator = TemplatedNarrativeGenerator()
    config = ExplanationConfig(mode="feature_importance")
    result = generator.generate(_credit_request(), _credit_schema(), config)

    expected = (
        "The model predicts Defaulted. "
        "The most important feature is dti (0.41), with a contribution of +0.37. "
        "The second most important feature is age (29), with a contribution of -0.12. "
        "The third most important feature is income (52000), with a contribution of +0.08. "
        "The 4th most important feature is employment_years (3), with a contribution of -0.05. "
        "The 5th most important feature is num_credit_lines (5), with a contribution of +0.03."
    )
    assert result.text == expected


def test_method_shap_injects_into_each_clause() -> None:
    """method='SHAP' resolves to 'SHAP ' (trailing space) inside each clause."""
    generator = TemplatedNarrativeGenerator(method="SHAP")
    config = ExplanationConfig(mode="feature_importance")
    result = generator.generate(_credit_request(), _credit_schema(), config)

    expected = (
        "The model predicts Defaulted. "
        "The most important feature is dti (0.41), with a SHAP contribution of +0.37. "
        "The second most important feature is age (29), with a SHAP contribution of -0.12. "
        "The third most important feature is income (52000), with a SHAP contribution of +0.08. "
        "The 4th most important feature is employment_years (3), "
        "with a SHAP contribution of -0.05. "
        "The 5th most important feature is num_credit_lines (5), with a SHAP contribution of +0.03."
    )
    assert result.text == expected


def test_ordinals_words_for_1_through_3_digits_from_4_onward() -> None:
    """Ranks 1/2/3 use 'most'/'second most'/'third most'; rank 4 uses '4th most'."""
    generator = TemplatedNarrativeGenerator()
    config = ExplanationConfig(mode="feature_importance", top_k_features=4)
    text = generator.generate(_credit_request(), _credit_schema(), config).text

    assert "The most important feature is dti" in text
    assert "The second most important feature is age" in text
    assert "The third most important feature is income" in text
    assert "The 4th most important feature is employment_years" in text
    assert "num_credit_lines" not in text


def test_ranking_by_absolute_importance_not_signed() -> None:
    """|-0.13| > |+0.04|, so the negative one ranks first."""
    schema = _credit_schema()
    request = TabularExplanationRequest(
        features={"dti": 0.20, "age": 50},
        prediction=Prediction(predicted_class=0),
        contributions=[
            TabularContribution(name="age", value=50, importance=0.04),
            TabularContribution(name="dti", value=0.20, importance=-0.13),
        ],
    )
    config = ExplanationConfig(mode="feature_importance", top_k_features=2)
    text = TemplatedNarrativeGenerator().generate(request, schema, config).text

    first_clause = text.split(". ")[1]
    second_clause = text.split(". ")[2]
    assert "dti" in first_clause and "-0.13" in first_clause
    assert "age" in second_clause and "+0.04" in second_clause


def test_generation_result_shape_llm_fields_none() -> None:
    """Templated path: text + latency populated; LLM-side fields all None."""
    generator = TemplatedNarrativeGenerator()
    config = ExplanationConfig(mode="feature_importance")
    result = generator.generate(_credit_request(), _credit_schema(), config)

    assert isinstance(result, GenerationResult)
    assert isinstance(result.text, str) and result.text
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    assert result.prompt is None
    assert result.model_name is None
    assert result.raw_llm_response is None
    assert result.tokens_used is None
    assert result.guardrails is None


def _valid_extraction_json_for_dti() -> str:
    return json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "drives default",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
        }
    )


def test_explainer_integration_templated_generation_llm_grading() -> None:
    """Templated generator + judge_llm: judge runs on the templated text."""
    judge = MockLLMProvider(responses=[_valid_extraction_json_for_dti()], model_name="judge")
    explainer = Explainer(
        schema=_credit_schema(),
        generator=TemplatedNarrativeGenerator(),
        judge_llm=judge,
    )
    result = explainer.explain(_credit_request())

    assert "The model predicts" in result.text
    assert result.prompt is None
    assert result.model_name is None
    assert result.narrative_extraction is not None
    assert judge._index == 1


def test_clause_template_custom_substitutes_via_substitution_helper() -> None:
    """Custom clause_template gets {ordinal}/{name}/{value}/{importance}/{method}."""
    generator = TemplatedNarrativeGenerator(
        clause_template="rank={ordinal} feat={name} val={value} imp={importance}.",
    )
    config = ExplanationConfig(mode="feature_importance", top_k_features=1)
    text = generator.generate(_credit_request(), _credit_schema(), config).text

    assert text == "The model predicts Defaulted. rank=most feat=dti val=0.41 imp=+0.37."


def test_contribution_feature_missing_from_request_features_raises() -> None:
    """Contribution names 'mystery' but features dict doesn't have it."""
    schema = _credit_schema()
    request = TabularExplanationRequest(
        features={"dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="mystery", value=999.0, importance=0.5),
        ],
    )
    config = ExplanationConfig(mode="feature_importance", top_k_features=1)
    with pytest.raises(
        ValueError,
        match=r"Contribution names feature 'mystery' but it is not in request\.features",
    ):
        TemplatedNarrativeGenerator().generate(request, schema, config)
