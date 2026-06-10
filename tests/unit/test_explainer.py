"""Tests for xainarratives.explainer."""

import pytest

from xainarratives import (
    DatasetSchema,
    Explainer,
    ExplanationConfig,
    GraphExplanationRequest,
    ImageExplanationRequest,
    Prediction,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
    TextExplanationRequest,
)
from xainarratives.prompts import EchoPromptTemplate
from xainarratives.providers import MockLLMProvider

# ---------------------------------------------------------------- #
# End-to-end round-trip per modality                               #
# ---------------------------------------------------------------- #


def _explainer(schema: DatasetSchema, responses: list[str] | None = None) -> Explainer:
    return Explainer(
        schema=schema,
        llm=MockLLMProvider(responses=responses or ["ok"]),
        prompt_template=EchoPromptTemplate(),
    )


def test_tabular_round_trip(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    result = _explainer(tabular_schema, ["tabular-ok"]).explain(tabular_request)
    assert result.text == "tabular-ok"
    assert result.mode == "feature_importance"
    assert result.model_name == "mock-v0"
    assert "SYSTEM:" in result.prompt and "USER:" in result.prompt
    assert result.latency_ms is not None and result.latency_ms >= 0.0


def test_text_round_trip(text_schema: DatasetSchema, text_request: TextExplanationRequest) -> None:
    result = _explainer(text_schema).explain(text_request)
    assert result.mode == "feature_importance"


def test_image_round_trip(
    image_schema: DatasetSchema, image_request: ImageExplanationRequest
) -> None:
    result = _explainer(image_schema).explain(image_request)
    assert result.mode == "feature_importance"


def test_graph_round_trip(
    graph_schema: DatasetSchema, graph_request: GraphExplanationRequest
) -> None:
    result = _explainer(graph_schema).explain(graph_request)
    assert result.mode == "feature_importance"


# ---------------------------------------------------------------- #
# Modality validation                                              #
# ---------------------------------------------------------------- #


def test_modality_mismatch_raises(
    tabular_schema: DatasetSchema, text_request: TextExplanationRequest
) -> None:
    explainer = _explainer(tabular_schema)
    with pytest.raises(ValueError, match="does not match schema modality"):
        explainer.explain(text_request)


# ---------------------------------------------------------------- #
# Mode validation                                                  #
# ---------------------------------------------------------------- #


def test_explicit_counterfactual_without_cfs_raises(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    explainer = Explainer(
        schema=tabular_schema,
        llm=MockLLMProvider(),
        prompt_template=EchoPromptTemplate(),
        config=ExplanationConfig(mode="counterfactual"),
    )
    with pytest.raises(ValueError, match=r"requires request\.counterfactuals"):
        explainer.explain(tabular_request)


def test_explicit_feature_importance_counterfactual_without_counterfactuals_raises(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    explainer = Explainer(
        schema=tabular_schema,
        llm=MockLLMProvider(),
        prompt_template=EchoPromptTemplate(),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
    )
    with pytest.raises(ValueError, match=r"requires request\.counterfactuals"):
        explainer.explain(tabular_request)


def test_explicit_feature_importance_ignores_counterfactuals(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Factual mode is always safe — extra inputs are allowed but not used."""
    req = tabular_request.model_copy(
        update={
            "counterfactuals": [
                TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20})
            ]
        }
    )
    explainer = Explainer(
        schema=tabular_schema,
        llm=MockLLMProvider(),
        prompt_template=EchoPromptTemplate(),
        config=ExplanationConfig(mode="feature_importance"),
    )
    assert explainer.explain(req).mode == "feature_importance"


# ---------------------------------------------------------------- #
# CF sanity check: must flip the class                             #
# ---------------------------------------------------------------- #


def test_cf_same_class_as_factual_warns(tabular_schema: DatasetSchema) -> None:
    req = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.3)],
        counterfactuals=[
            # Same predicted class as the factual — should warn.
            TabularCounterfactual(predicted_class=1, features={"age": 29, "dti": 0.20})
        ],
    )
    with pytest.warns(UserWarning, match="predicts the same class"):
        _explainer(tabular_schema).explain(req)


def test_cf_flipping_class_does_not_warn(tabular_schema: DatasetSchema) -> None:
    req = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.3)],
        counterfactuals=[
            TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20})
        ],
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        _explainer(tabular_schema).explain(req)
