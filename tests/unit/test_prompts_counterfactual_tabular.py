"""Unit tests for CounterfactualTabularPromptTemplate (ADR 0029 + ADR 0031).

Per ADR 0031 a request carries a single counterfactual; the template
renders one flip block (no numbering, no alternatives).
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
)
from xains.prompts.counterfactual_tabular import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    CounterfactualTabularPromptTemplate,
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


def _config() -> ExplanationConfig:
    return ExplanationConfig(mode="counterfactual", audience="end_user")


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


# ------------------------------------------------------ defaults present


def test_module_exports_default_templates() -> None:
    assert isinstance(DEFAULT_SYSTEM_TEMPLATE, str)
    assert isinstance(DEFAULT_USER_TEMPLATE, str)
    assert "{counterfactual}" in DEFAULT_USER_TEMPLATE


# ------------------------------------------------------ render shape


def test_single_change_renders_flip_lead_and_indented_change_line() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    assert "To change the prediction from Defaulted to Repaid:" in user
    assert "  - salary: 52000 -> 80000 [EUR]" in user
    # No numbering exists anymore.
    assert "Scenario" not in user


def test_multiple_changes_in_one_cf_render_one_block_with_multiple_lines() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 35, "salary": 80000, "dti": 0.20}))
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    # All three change lines appear under the same single flip lead.
    assert user.count("To change the prediction from Defaulted to Repaid:") == 1
    assert "  - age: 29 -> 35 [years]" in user
    assert "  - salary: 52000 -> 80000 [EUR]" in user
    assert "  - dti: 0.41 -> 0.2" in user


# ------------------------------------------------------ method provenance


def test_method_off_by_default_never_appears() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, method="DiCE"))
    _, user = CounterfactualTabularPromptTemplate().render(req, schema, _config())
    assert "DiCE" not in user
    assert "method" not in user


def test_method_shown_when_include_method_true_and_cf_method_set() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, method="DiCE"))
    _, user = CounterfactualTabularPromptTemplate(include_method=True).render(
        req, schema, _config()
    )
    assert "(method: DiCE)" in user


def test_method_omitted_when_include_method_true_but_cf_method_none() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, method=None))
    _, user = CounterfactualTabularPromptTemplate(include_method=True).render(
        req, schema, _config()
    )
    assert "method" not in user


# ------------------------------------------------------ error paths


def test_non_tabular_request_raises_type_error() -> None:
    schema = _schema()
    with pytest.raises(TypeError, match="TabularExplanationRequest"):
        CounterfactualTabularPromptTemplate().render(
            object(),  # type: ignore[arg-type]
            schema,
            _config(),
        )


def test_counterfactual_none_raises_value_error() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=None,
    )
    with pytest.raises(ValueError, match=r"counterfactual"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


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
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_cf_changes_feature_absent_from_schema_raises_value_error() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41, "mystery": 1},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=_cf({"age": 29, "salary": 52000, "dti": 0.41, "mystery": 7}),
    )
    with pytest.raises(ValueError, match=r"mystery"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_factual_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=99),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=_cf({"age": 29, "salary": 80000, "dti": 0.41}),
    )
    with pytest.raises(ValueError, match=r"99"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


def test_cf_predicted_class_not_in_schema_raises() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}, predicted_class=99))
    with pytest.raises(ValueError, match=r"99"):
        CounterfactualTabularPromptTemplate().render(req, schema, _config())


# ------------------------------------------------------ editable templates (ADR 0017)


def test_extra_placeholders_conflicting_with_builtin_raises_in_init() -> None:
    with pytest.raises(ValueError, match=r"counterfactual"):
        CounterfactualTabularPromptTemplate(extra_placeholders={"counterfactual": "x"})


def test_custom_templates_render_with_known_placeholders() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    custom_sys = "Custom for {target_name}/{audience}."
    custom_user = "P:{prediction}\nC:{counterfactual}"
    sys, user = CounterfactualTabularPromptTemplate(
        system_template=custom_sys,
        user_template=custom_user,
    ).render(req, schema, _config())
    assert sys == "Custom for default/end_user."
    assert user.startswith("P:Defaulted\nC:")


def test_unknown_token_in_template_raises_value_error() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    with pytest.raises(ValueError, match=r"bogus"):
        CounterfactualTabularPromptTemplate(
            user_template="P:{prediction}\nC:{counterfactual}\nExtra:{bogus}",
        ).render(req, schema, _config())


def test_extra_placeholders_substituted_into_custom_template() -> None:
    schema = _schema()
    req = _request_with(_cf({"age": 29, "salary": 80000, "dti": 0.41}))
    _, user = CounterfactualTabularPromptTemplate(
        extra_placeholders={"region": "EU"},
        user_template="{counterfactual}\nRegion: {region}",
    ).render(req, schema, _config())
    assert "Region: EU" in user
