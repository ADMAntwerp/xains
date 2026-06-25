"""Unit tests for extract_counterfactual_claims + CF extraction models (ADR 0031).

Mirrors test_guardrails_extraction.py for the counterfactual path.
"""

import json

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.guardrails import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    extract_counterfactual_claims,
)
from xains.providers import MockLLMProvider
from xains.types import TabularCounterfactual


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


def _request() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "salary": 52000, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=TabularCounterfactual(
            predicted_class=0,
            features={"age": 29, "salary": 52000, "dti": 0.20},
        ),
    )


def _judge(canned: str) -> MockLLMProvider:
    return MockLLMProvider(responses=[canned])


# ------------------------------------------------------ model contracts


def test_resolved_changes_key_must_match_resolved_to() -> None:
    with pytest.raises(ValidationError, match=r"resolved_to"):
        CounterfactualExtraction(
            changes={
                "dti": CounterfactualFeatureClaim(
                    narrative_name="DTI",
                    resolved_to="other",  # mismatch
                    stated_before=0.41,
                    stated_after=0.20,
                ),
            },
            invented=[],
            prompt_version="1",
            model_name="m",
        )


def test_invented_claims_must_have_resolved_to_none() -> None:
    with pytest.raises(ValidationError, match=r"resolved_to"):
        CounterfactualExtraction(
            changes={},
            invented=[
                CounterfactualFeatureClaim(
                    narrative_name="X",
                    resolved_to="dti",  # must be None for invented
                ),
            ],
            prompt_version="1",
            model_name="m",
        )


def test_claim_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CounterfactualFeatureClaim(
            narrative_name="X",
            extra="bogus",  # type: ignore[call-arg]
        )


def test_extraction_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CounterfactualExtraction(
            changes={},
            invented=[],
            prompt_version="1",
            model_name="m",
            extra="bogus",  # type: ignore[call-arg]
        )


# ------------------------------------------------------ extract_counterfactual_claims happy path


def test_happy_path_parses_canned_json_into_extraction() -> None:
    payload = json.dumps(
        {
            "changes": {
                "dti": {
                    "narrative_name": "debt-to-income ratio",
                    "stated_before": 0.41,
                    "stated_after": 0.20,
                    "stated_direction": "decreased",
                },
            },
            "invented": [],
        }
    )
    extraction, response, failure = extract_counterfactual_claims(
        "narrative...", _request(), _schema(), _judge(payload)
    )
    assert failure is None
    assert extraction is not None
    assert "dti" in extraction.changes
    claim = extraction.changes["dti"]
    assert claim.resolved_to == "dti"
    assert claim.narrative_name == "debt-to-income ratio"
    assert claim.stated_before == 0.41
    assert claim.stated_after == 0.20
    assert claim.stated_direction == "decreased"
    assert extraction.invented == []
    assert extraction.prompt_version  # non-empty
    assert response.text == payload


def test_invented_channel_captures_unresolved_mentions() -> None:
    payload = json.dumps(
        {
            "changes": {},
            "invented": [
                {
                    "narrative_name": "credit history",
                    "stated_before": "fair",
                    "stated_after": "excellent",
                    "stated_direction": "improved",
                }
            ],
        }
    )
    extraction, _, failure = extract_counterfactual_claims(
        "narrative...", _request(), _schema(), _judge(payload)
    )
    assert failure is None
    assert extraction is not None
    assert len(extraction.invented) == 1
    inv = extraction.invented[0]
    assert inv.narrative_name == "credit history"
    assert inv.resolved_to is None
    assert inv.stated_before == "fair"
    assert inv.stated_after == "excellent"


def test_mixed_resolved_and_invented() -> None:
    payload = json.dumps(
        {
            "changes": {
                "dti": {
                    "narrative_name": "DTI",
                    "stated_before": 0.41,
                    "stated_after": 0.20,
                    "stated_direction": "decreased",
                },
            },
            "invented": [
                {
                    "narrative_name": "credit score",
                    "stated_before": None,
                    "stated_after": None,
                    "stated_direction": "improved",
                },
            ],
        }
    )
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert failure is None
    assert extraction is not None
    assert set(extraction.changes) == {"dti"}
    assert len(extraction.invented) == 1


def test_empty_extraction_when_narrative_lists_no_changes() -> None:
    payload = json.dumps({"changes": {}, "invented": []})
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert failure is None
    assert extraction is not None
    assert extraction.changes == {}
    assert extraction.invented == []


def test_fenced_json_is_stripped() -> None:
    payload = "```json\n" + json.dumps({"changes": {}, "invented": []}) + "\n```"
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert failure is None
    assert extraction is not None


# ------------------------------------------------------ failure paths


def test_malformed_json_returns_advisory_failure() -> None:
    extraction, response, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge("not json")
    )
    assert extraction is None
    assert failure is not None
    assert failure.name == "extract_counterfactual_claims"
    assert failure.severity == "advisory"
    assert failure.passed is False
    # Response is still returned so token spend is accounted for.
    assert response.text == "not json"


def test_changes_key_not_in_schema_returns_advisory_failure() -> None:
    payload = json.dumps(
        {
            "changes": {
                "mystery": {
                    "narrative_name": "?",
                    "stated_before": None,
                    "stated_after": None,
                    "stated_direction": None,
                },
            },
            "invented": [],
        }
    )
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert extraction is None
    assert failure is not None
    assert failure.severity == "advisory"


def test_extra_fields_in_wire_payload_returns_advisory_failure() -> None:
    payload = json.dumps(
        {
            "changes": {
                "dti": {
                    "narrative_name": "DTI",
                    "stated_before": 0.41,
                    "stated_after": 0.20,
                    "stated_direction": "decreased",
                    "unexpected_field": "x",
                },
            },
            "invented": [],
        }
    )
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert extraction is None
    assert failure is not None


def test_wrong_top_level_shape_returns_advisory_failure() -> None:
    # Missing required `invented` key; pydantic default kicks in only if the
    # whole `changes` dict shape is also valid - let's break with an array.
    payload = json.dumps([{"changes": {}, "invented": []}])
    extraction, _, failure = extract_counterfactual_claims(
        "...", _request(), _schema(), _judge(payload)
    )
    assert extraction is None
    assert failure is not None
