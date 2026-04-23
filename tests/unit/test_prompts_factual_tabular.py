"""Tests for xainarratives.prompts.factual_tabular."""

import pytest

from xainarratives import (
    DatasetSchema,
    ExplanationConfig,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
    TextExplanationRequest,
)
from xainarratives.prompts import FactualTabularPromptTemplate


def _assert_value_near_name(user: str, name: str, value: str, window: int = 40) -> None:
    idx = user.find(name)
    assert idx >= 0, f"{name} not in prompt"
    region = user[idx : idx + len(name) + window]
    assert value in region, f"{value} not within {window} chars of {name}"


def test_render_returns_non_empty_system_and_user(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig()
    system, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert isinstance(system, str) and system.strip()
    assert isinstance(user, str) and user.strip()


def test_all_top_k_features_appear_in_user_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(top_k_features=2)
    _, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "dti" in user
    assert "age" in user


def test_top_k_truncates_to_k(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(top_k_features=1)
    _, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "dti" in user
    assert "age" not in user


def test_unranked_contributions_sorted_by_abs_importance(
    tabular_schema: DatasetSchema,
) -> None:
    request = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="age", value=29, importance=-0.12),
            TabularContribution(name="dti", value=0.41, importance=0.37),
        ],
    )
    config = ExplanationConfig(top_k_features=1)
    _, user = FactualTabularPromptTemplate().render(request, tabular_schema, config)
    assert "dti" in user
    assert "age" not in user


def test_top_k_tie_breaking_widens_the_cut(
    tabular_schema: DatasetSchema,
) -> None:
    request = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="age", value=29, importance=0.30),
            TabularContribution(name="dti", value=0.41, importance=-0.30),
        ],
    )
    config = ExplanationConfig(top_k_features=1)
    _, user = FactualTabularPromptTemplate().render(request, tabular_schema, config)
    assert "age" in user
    assert "dti" in user


def test_out_of_schema_contribution_name_raises(
    tabular_schema: DatasetSchema,
) -> None:
    request = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="bogus", value=1.0, importance=0.5),
        ],
    )
    config = ExplanationConfig()
    with pytest.raises(ValueError, match="bogus"):
        FactualTabularPromptTemplate().render(request, tabular_schema, config)


def test_predicted_class_uses_human_label(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig()
    _, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "Defaulted" in user


def test_unknown_predicted_class_raises(
    tabular_schema: DatasetSchema,
) -> None:
    request = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=99),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
        ],
    )
    config = ExplanationConfig()
    with pytest.raises(ValueError, match="99"):
        FactualTabularPromptTemplate().render(request, tabular_schema, config)


def test_audience_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(audience="business")
    system, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "business" in (system + user)


def test_tone_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(tone="empathetic")
    system, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "empathetic" in (system + user)


def test_max_length_words_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(max_length_words=77)
    system, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "77" in (system + user)


def test_rejects_non_tabular_request(
    tabular_schema: DatasetSchema, text_request: TextExplanationRequest
) -> None:
    config = ExplanationConfig()
    with pytest.raises(TypeError):
        FactualTabularPromptTemplate().render(text_request, tabular_schema, config)


def test_system_prompt_mentions_target_name(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig()
    system, _ = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    assert "default" in system


def test_render_is_deterministic(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FactualTabularPromptTemplate()
    config = ExplanationConfig()
    s1, u1 = template.render(tabular_request, tabular_schema, config)
    s2, u2 = template.render(tabular_request, tabular_schema, config)
    assert s1 == s2
    assert u1 == u2


def test_top_k_feature_values_surfaced(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(top_k_features=2)
    _, user = FactualTabularPromptTemplate().render(tabular_request, tabular_schema, config)
    _assert_value_near_name(user, "dti", "0.41")
    _assert_value_near_name(user, "age", "29")


def test_string_predicted_class_rendered_as_human_label() -> None:
    schema = DatasetSchema(
        modality=Modality.TABULAR,
        name="toy",
        description="Toy schema with string class keys.",
        target=TargetSchema(
            name="label",
            description="Label.",
            classes={"a": "Alpha", "b": "Beta"},
        ),
        features=[
            FeatureSchema(name="x", dtype="numeric", description="X."),
        ],
    )
    request = TabularExplanationRequest(
        features={"x": 1.0},
        prediction=Prediction(predicted_class="a"),
        contributions=[
            TabularContribution(name="x", value=1.0, importance=0.5),
        ],
    )
    config = ExplanationConfig()
    _, user = FactualTabularPromptTemplate().render(request, schema, config)
    assert "Alpha" in user
