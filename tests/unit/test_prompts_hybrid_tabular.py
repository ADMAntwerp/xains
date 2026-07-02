"""Unit tests for HybridTabularPromptTemplate (ADR 0039).

Mirrors the FI + CF template test styles. Verifies the hybrid composes
both body blocks in one prompt, keeps FI-half guards, forwards the CF
`include_method` flag, and preserves the ADR 0029 editable-template
contract.
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
    TabularCounterfactual,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.prompts.hybrid_tabular import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    HybridTabularPromptTemplate,
)


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
            FeatureSchema(name="salary", dtype="numeric", unit="EUR", description="salary"),
            FeatureSchema(name="dti", dtype="numeric", description="debt-to-income"),
        ],
    )


def _config() -> ExplanationConfig:
    return ExplanationConfig(
        mode="feature_importance_counterfactual",
        audience="end_user",
    )


def _request_with(
    *,
    contributions: list[TabularContribution] | None = None,
    counterfactual: TabularCounterfactual | None = None,
    factual_predicted_class: int = 1,
) -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(
            predicted_class=factual_predicted_class,
            probabilities={0: 0.2, 1: 0.8},
        ),
        contributions=contributions
        or [
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
        ],
        counterfactual=counterfactual
        or TabularCounterfactual(
            predicted_class=0,
            features={"age": 29, "salary": 52000, "dti": 0.20},
        ),
    )


def _cf(
    features: dict[str, Any],
    *,
    predicted_class: int = 0,
    method: str | None = None,
) -> TabularCounterfactual:
    return TabularCounterfactual(predicted_class=predicted_class, features=features, method=method)


# ------------------------------------------------------ module exports


def test_module_exports_default_templates() -> None:
    assert isinstance(DEFAULT_SYSTEM_TEMPLATE, str)
    assert isinstance(DEFAULT_USER_TEMPLATE, str)
    assert "{contributions}" in DEFAULT_USER_TEMPLATE
    assert "{counterfactual}" in DEFAULT_USER_TEMPLATE


# ------------------------------------------------------ render composes both sections


def test_render_returns_both_contribution_block_and_counterfactual_block() -> None:
    schema = _schema()
    req = _request_with()
    _, user = HybridTabularPromptTemplate().render(req, schema, _config())
    # FI-side block content
    assert "- dti = 0.41: importance=+0.37" in user
    assert "- age = 29 [years]: importance=-0.12" in user
    # CF-side block content
    assert "To change the prediction from Defaulted to Repaid:" in user
    assert "  - dti: 0.41 -> 0.2" in user


def test_render_system_prompt_contains_two_part_and_before_after_instructions() -> None:
    schema = _schema()
    req = _request_with()
    system, _ = HybridTabularPromptTemplate().render(req, schema, _config())
    assert "two parts" in system
    assert "counterfactual" in system
    assert "value it changes from" in system
    assert "value it changes to" in system
    assert "in the counterfactual" in system  # scoping to the CF half


# ------------------------------------------------------ {prediction} is the factual label


def test_prediction_placeholder_is_the_factual_label_not_the_cf_label() -> None:
    """{prediction} must resolve to schema.target.classes[factual predicted_class] (matching FI)."""
    schema = _schema()
    req = _request_with()  # factual=1 (Defaulted), CF flips to 0 (Repaid)
    _, user = HybridTabularPromptTemplate().render(req, schema, _config())
    assert user.startswith("Prediction: Defaulted.")
    # The CF label still appears inside the CF block's flip lead:
    assert "to Repaid:" in user


# ------------------------------------------------------ include_method


def test_include_method_off_by_default_hides_method_even_when_set() -> None:
    schema = _schema()
    req = _request_with(
        counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.20}, method="DiCE")
    )
    _, user = HybridTabularPromptTemplate().render(req, schema, _config())
    assert "DiCE" not in user
    assert "method" not in user


def test_include_method_true_with_cf_method_shows_suffix_on_cf_flip_line() -> None:
    schema = _schema()
    req = _request_with(
        counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.20}, method="DiCE")
    )
    _, user = HybridTabularPromptTemplate(include_method=True).render(req, schema, _config())
    assert "(method: DiCE)" in user
    # The suffix belongs to the CF block, not the FI block.
    fi_block, cf_block = user.split("Counterfactual scenario")
    assert "(method: DiCE)" not in fi_block
    assert "(method: DiCE)" in cf_block


def test_include_method_true_but_cf_method_none_hides_suffix() -> None:
    schema = _schema()
    req = _request_with(counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.20}, method=None))
    _, user = HybridTabularPromptTemplate(include_method=True).render(req, schema, _config())
    assert "method" not in user


# ------------------------------------------------------ guards


def test_non_tabular_request_raises_type_error() -> None:
    schema = _schema()
    with pytest.raises(TypeError, match="TabularExplanationRequest"):
        HybridTabularPromptTemplate().render(
            object(),  # type: ignore[arg-type]
            schema,
            _config(),
        )


def test_missing_counterfactual_raises_value_error() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=None,
    )
    with pytest.raises(ValueError, match=r"counterfactual"):
        HybridTabularPromptTemplate().render(req, schema, _config())


def test_unknown_feature_in_contributions_raises_value_error() -> None:
    schema = _schema()
    req = _request_with(
        contributions=[
            TabularContribution(name="mystery", value=1, importance=0.5),
        ],
    )
    with pytest.raises(ValueError, match=r"mystery"):
        HybridTabularPromptTemplate().render(req, schema, _config())


def test_factual_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=99),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.20}),
    )
    with pytest.raises(ValueError, match=r"99"):
        HybridTabularPromptTemplate().render(req, schema, _config())


# ------------------------------------------------------ editable templates (ADR 0017)


def test_custom_templates_render_with_known_placeholders() -> None:
    schema = _schema()
    req = _request_with()
    custom_sys = "Custom for {target_name}/{audience}."
    custom_user = "P:{prediction}\nFI:{contributions}\nCF:{counterfactual}"
    sys, user = HybridTabularPromptTemplate(
        system_template=custom_sys,
        user_template=custom_user,
    ).render(req, schema, _config())
    assert sys == "Custom for default/end_user."
    assert user.startswith("P:Defaulted\nFI:")
    assert "\nCF:To change the prediction from Defaulted to Repaid:" in user


def test_unknown_token_in_template_raises_value_error() -> None:
    schema = _schema()
    req = _request_with()
    with pytest.raises(ValueError, match=r"bogus"):
        HybridTabularPromptTemplate(
            user_template="P:{prediction}\nC:{contributions}\nCF:{counterfactual}\nExtra:{bogus}",
        ).render(req, schema, _config())


def test_extra_placeholders_conflicting_with_builtin_raises_in_init() -> None:
    with pytest.raises(ValueError, match=r"contributions"):
        HybridTabularPromptTemplate(extra_placeholders={"contributions": "x"})


def test_extra_placeholders_conflicting_with_counterfactual_builtin_raises() -> None:
    with pytest.raises(ValueError, match=r"counterfactual"):
        HybridTabularPromptTemplate(extra_placeholders={"counterfactual": "x"})


def test_extra_placeholders_substituted_into_custom_template() -> None:
    schema = _schema()
    req = _request_with()
    _, user = HybridTabularPromptTemplate(
        extra_placeholders={"region": "EU"},
        user_template="{contributions}\n{counterfactual}\nRegion: {region}",
    ).render(req, schema, _config())
    assert "Region: EU" in user
