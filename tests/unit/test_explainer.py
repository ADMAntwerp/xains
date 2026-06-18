"""Tests for xain.explainer."""

import pytest

from xain import (
    DatasetSchema,
    Explainer,
    ExplanationConfig,
    GraphExplanationRequest,
    ImageExplanationRequest,
    LLMNarrativeGenerator,
    Prediction,
    TabularContribution,
    TabularCounterfactual,
    TabularExplanationRequest,
    TextExplanationRequest,
)
from xain.prompts import EchoPromptTemplate
from xain.providers import MockLLMProvider

# ---------------------------------------------------------------- #
# End-to-end round-trip per modality                               #
# ---------------------------------------------------------------- #


def _explainer(schema: DatasetSchema, responses: list[str] | None = None) -> Explainer:
    llm = MockLLMProvider(responses=responses or ["ok"])
    return Explainer(
        schema=schema,
        generator=LLMNarrativeGenerator(prompt_template=EchoPromptTemplate(), llm=llm),
        judge_llm=llm,
    )


def test_tabular_round_trip(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    result = _explainer(tabular_schema, ["tabular-ok"]).explain(tabular_request)
    assert result.text == "tabular-ok"
    assert result.mode == "feature_importance"
    assert result.model_name == "mock-v0"
    assert result.prompt is not None and "SYSTEM:" in result.prompt and "USER:" in result.prompt
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
    llm = MockLLMProvider()
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=EchoPromptTemplate(), llm=llm),
        config=ExplanationConfig(mode="counterfactual"),
        judge_llm=llm,
    )
    with pytest.raises(ValueError, match=r"requires request\.counterfactuals"):
        explainer.explain(tabular_request)


def test_explicit_feature_importance_counterfactual_without_counterfactuals_raises(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider()
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=EchoPromptTemplate(), llm=llm),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
        judge_llm=llm,
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
    llm = MockLLMProvider()
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=EchoPromptTemplate(), llm=llm),
        config=ExplanationConfig(mode="feature_importance"),
        judge_llm=llm,
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


# ---------------------------------------------------------------- #
# Phase-1 behavior change: judge_llm-None fail-fast                #
# ---------------------------------------------------------------- #


def test_explain_raises_when_extract_narrative_true_and_judge_llm_none(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """``config.extract_narrative=True`` with ``judge_llm=None`` must raise a
    clear ``ValueError`` — replacing the silent ``judge_llm = self.llm``
    fallback present in the pre-refactor Explainer.

    NOTE — TEST-MIGRATION EXCEPTION. When the NarrativeGenerator refactor
    lands, this test's construction shape migrates to the new signature:

        Explainer(
            schema=tabular_schema,
            generator=LLMNarrativeGenerator(
                prompt_template=EchoPromptTemplate(),
                llm=MockLLMProvider(responses=["the applicant Defaulted"]),
            ),
            # judge_llm deliberately omitted to provoke the new ValueError
        )

    The ``judge_llm`` omission is INTENTIONAL — it is what triggers the
    new error. The standard 17-site migration rule (always preserve the
    fallback by adding ``judge_llm=L`` explicitly) does NOT apply to this
    test.

    Pre-refactor red state: this test fails with 'DID NOT RAISE ValueError'
    because today's code silently falls back to ``self.llm`` as the judge
    and ``explain()`` completes successfully.
    """
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(
            prompt_template=EchoPromptTemplate(),
            llm=MockLLMProvider(responses=["the applicant Defaulted"]),
        ),
        # judge_llm deliberately omitted to provoke the new ValueError
    )
    with pytest.raises(ValueError, match=r"extract_narrative=True requires judge_llm"):
        explainer.explain(tabular_request)
