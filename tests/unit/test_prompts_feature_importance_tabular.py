"""Tests for xainarratives.prompts.feature_importance_tabular."""

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
from xainarratives.config import DEFAULT_NARRATIVE_RULES
from xainarratives.prompts import FeatureImportanceTabularPromptTemplate


def _assert_value_near_name(user: str, name: str, value: str, window: int = 40) -> None:
    idx = user.find(name)
    assert idx >= 0, f"{name} not in prompt"
    region = user[idx : idx + len(name) + window]
    assert value in region, f"{value} not within {window} chars of {name}"


def test_render_returns_non_empty_system_and_user(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance")
    system, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert isinstance(system, str) and system.strip()
    assert isinstance(user, str) and user.strip()


def test_all_top_k_features_appear_in_user_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", top_k_features=2)
    _, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert "dti" in user
    assert "age" in user


def test_top_k_truncates_to_k(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", top_k_features=1)
    _, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
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
    config = ExplanationConfig(mode="feature_importance", top_k_features=1)
    _, user = FeatureImportanceTabularPromptTemplate().render(request, tabular_schema, config)
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
    config = ExplanationConfig(mode="feature_importance", top_k_features=1)
    _, user = FeatureImportanceTabularPromptTemplate().render(request, tabular_schema, config)
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
    config = ExplanationConfig(mode="feature_importance")
    with pytest.raises(ValueError, match="bogus"):
        FeatureImportanceTabularPromptTemplate().render(request, tabular_schema, config)


def test_predicted_class_uses_human_label(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance")
    _, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
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
    config = ExplanationConfig(mode="feature_importance")
    with pytest.raises(ValueError, match="99"):
        FeatureImportanceTabularPromptTemplate().render(request, tabular_schema, config)


def test_audience_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", audience="business")
    system, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert "business" in (system + user)


def test_tone_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", tone="empathetic")
    system, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert "empathetic" in (system + user)


def test_max_length_words_reflected_in_prompt(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", max_length_words=77)
    system, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert "77" in (system + user)


def test_rejects_non_tabular_request(
    tabular_schema: DatasetSchema, text_request: TextExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance")
    with pytest.raises(TypeError):
        FeatureImportanceTabularPromptTemplate().render(text_request, tabular_schema, config)


def test_system_prompt_mentions_target_name(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance")
    system, _ = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert "default" in system


def test_render_is_deterministic(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate()
    config = ExplanationConfig(mode="feature_importance")
    s1, u1 = template.render(tabular_request, tabular_schema, config)
    s2, u2 = template.render(tabular_request, tabular_schema, config)
    assert s1 == s2
    assert u1 == u2


def test_top_k_feature_values_surfaced(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", top_k_features=2)
    _, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
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
    config = ExplanationConfig(mode="feature_importance")
    _, user = FeatureImportanceTabularPromptTemplate().render(request, schema, config)
    assert "Alpha" in user


def test_system_includes_default_narrative_rules(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance")
    system, _ = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    # Default rules block appears verbatim in the system message.
    assert DEFAULT_NARRATIVE_RULES in system
    # Existing header still present.
    assert "You are explaining" in system


def test_system_includes_custom_narrative_rules(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    config = ExplanationConfig(mode="feature_importance", narrative_rules="CUSTOM_NARRATIVE_RULES")
    system, _ = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    # Custom block replaces (not appends to) the default.
    assert "CUSTOM_NARRATIVE_RULES" in system
    assert "An XAI Narrative should establish a continuous structure" not in system


# ====================================================================
# Editable prompt templates (placeholder substitution)
# ====================================================================


def test_default_templates_reproduce_legacy_output_byte_for_byte(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """The critical regression guard: default templates produce byte-identical
    output to the pre-refactor render() implementation."""
    config = ExplanationConfig(mode="feature_importance")
    system, user = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    expected_system = (
        "You are explaining a model prediction for the 'default' target. "
        "Audience: end_user. Tone: neutral. "
        "Keep the explanation under 150 words."
        "\n\n"
        f"{DEFAULT_NARRATIVE_RULES}"
    )
    expected_user = (
        "Prediction: Defaulted.\n"
        "Top contributions by magnitude:\n"
        "- dti = 0.41: importance=+0.37\n"
        "- age = 29 [years]: importance=-0.12"
    )
    assert system == expected_system
    assert user == expected_user


def test_custom_system_template_substitutes_builtins(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate(
        system_template="Target: {target_name} | Audience: {audience}"
    )
    system, _ = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert system == "Target: default | Audience: end_user"


def test_custom_user_template_substitutes_builtins(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate(
        user_template="Pred: {prediction}\n{contributions}"
    )
    _, user = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert user == (
        "Pred: Defaulted\n- dti = 0.41: importance=+0.37\n- age = 29 [years]: importance=-0.12"
    )


def test_extra_placeholders_substitute_in_system_template(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate(
        system_template="Domain: {domain}. Target: {target_name}.",
        extra_placeholders={"domain": "credit risk"},
    )
    system, _ = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert system == "Domain: credit risk. Target: default."


def test_extra_placeholders_substitute_in_user_template(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate(
        user_template="Domain={domain} | Pred={prediction}",
        extra_placeholders={"domain": "credit risk"},
    )
    _, user = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert user == "Domain=credit risk | Pred=Defaulted"


def test_extra_placeholders_reserved_name_raises() -> None:
    """extra_placeholders cannot rebind built-in names; constructor-time fail-fast."""
    with pytest.raises(ValueError, match=r"target_name"):
        FeatureImportanceTabularPromptTemplate(extra_placeholders={"target_name": "spoof"})


def test_unknown_placeholder_in_system_template_raises_at_render(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """A typo in a system_template placeholder name fires at render() with both the
    typo and (some of) the valid names in the message."""
    template = FeatureImportanceTabularPromptTemplate(system_template="Hi {target_naem}")
    with pytest.raises(ValueError) as excinfo:
        template.render(
            tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
        )
    msg = str(excinfo.value)
    assert "target_naem" in msg
    assert "target_name" in msg  # the actual valid name appears in the error


def test_unknown_placeholder_in_user_template_raises_at_render(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    template = FeatureImportanceTabularPromptTemplate(
        user_template="{predicition} | {contributions}"
    )
    with pytest.raises(ValueError) as excinfo:
        template.render(
            tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
        )
    msg = str(excinfo.value)
    assert "predicition" in msg
    assert "prediction" in msg


def test_omitted_builtin_placeholder_is_allowed(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Templates need not reference every built-in; omitting one is the user's choice."""
    template = FeatureImportanceTabularPromptTemplate(
        user_template="Just the prediction: {prediction}"
    )
    _, user = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert user == "Just the prediction: Defaulted"


def test_literal_json_braces_left_alone(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """JSON-like `{"key": "val"}` is not a placeholder (first char after { is `"`)."""
    template = FeatureImportanceTabularPromptTemplate(
        system_template='JSON: {"key": "val"}. Target: {target_name}.'
    )
    system, _ = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert system == 'JSON: {"key": "val"}. Target: default.'


def test_literal_non_identifier_braces_left_alone(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Empty, spaced, and digit-starting brace forms are not placeholders."""
    template = FeatureImportanceTabularPromptTemplate(
        system_template="Empty: {} | Spaced: { foo } | Digits: {123} | Target: {target_name}"
    )
    system, _ = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert system == "Empty: {} | Spaced: { foo } | Digits: {123} | Target: default"


def test_bare_unmatched_braces_left_alone(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Standalone braces / non-pattern brace pairs are not parsed as placeholders."""
    template = FeatureImportanceTabularPromptTemplate(
        system_template="Open: { suffix } close. Target: {target_name}"
    )
    system, _ = template.render(
        tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
    )
    assert system == "Open: { suffix } close. Target: default"


def test_narrative_rules_with_braces_substitutes_clean_no_re_parse(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """One-pass guarantee: the SUBSTITUTED value of {narrative_rules} is not
    re-scanned for placeholders. Even an identifier-shaped {appendix} inside the
    custom narrative_rules survives verbatim — if substitution were recursive,
    this would raise on the unknown {appendix} token."""
    custom_rules = "See the {appendix} for full instructions."
    config = ExplanationConfig(mode="feature_importance", narrative_rules=custom_rules)
    system, _ = FeatureImportanceTabularPromptTemplate().render(
        tabular_request, tabular_schema, config
    )
    assert system.endswith("See the {appendix} for full instructions.")


def test_unknown_identifier_braces_raise(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Literal {identifier} text in a TEMPLATE is not supported in v1. An unknown
    identifier-shaped token is a hard error, not a pass-through. Documents the
    deferred-decision: escaping ({{ }} or otherwise) is intentionally not
    implemented; users must either declare the name via extra_placeholders or
    not use that pattern in the template."""
    template = FeatureImportanceTabularPromptTemplate(
        system_template="See the {appendix} for details. Target: {target_name}"
    )
    with pytest.raises(ValueError) as excinfo:
        template.render(
            tabular_request, tabular_schema, ExplanationConfig(mode="feature_importance")
        )
    assert "appendix" in str(excinfo.value)
