"""Unit tests for extract_narrative_claims.

Extraction is a single LLM call that produces a structured JSON claim per
feature the narrative mentions. These tests script MockLLMProvider to return
known JSON shapes and assert the parsing / validation / failure behaviour.
No real LLM is called.
"""

import json

import pytest

from xainarratives import (
    DatasetSchema,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xainarratives.guardrails import (
    FeatureClaim,
    GuardrailResult,
    NarrativeExtraction,
    extract_narrative_claims,
)
from xainarratives.providers import MockLLMProvider

# ------------------------------------------------------ fixtures


@pytest.fixture
def schema() -> DatasetSchema:
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
            FeatureSchema(name="age", dtype="numeric", description="Applicant age."),
            FeatureSchema(name="dti", dtype="numeric", description="Debt-to-income ratio."),
        ],
    )


@pytest.fixture
def request_obj() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
        ],
    )


def _ok_payload() -> str:
    return json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "pushes toward default",
                },
                "age": {
                    "rank": 2,
                    "sign": -1,
                    "value": 29,
                    "assumption": "younger, mildly offsets",
                },
            }
        }
    )


# ----------------------------------------- happy-path parsing


def test_extraction_parses_well_formed_json(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[_ok_payload()], model_name="mock-judge")
    extraction, response, failure = extract_narrative_claims(
        text="Explanation text.", request=request_obj, schema=schema, judge_llm=llm
    )
    assert isinstance(extraction, NarrativeExtraction)
    assert failure is None
    assert response.text == _ok_payload()
    assert set(extraction.features.keys()) == {"dti", "age"}
    assert all(isinstance(c, FeatureClaim) for c in extraction.features.values())


def test_extraction_records_prompt_version(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[_ok_payload()], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.prompt_version == "1"


def test_extraction_records_model_name_from_response(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[_ok_payload()], model_name="judge-xyz")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.model_name == "judge-xyz"


# ----------------------------------------- narrative-name preservation


def test_extraction_preserves_narrative_feature_name_keys(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Narrative uses synonym 'debt-to-income'; key must NOT be normalized to 'dti'."""
    payload = json.dumps(
        {
            "features": {
                "debt-to-income": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "pushes toward default",
                }
            }
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert "debt-to-income" in extraction.features
    assert "dti" not in extraction.features


# ----------------------------------------- value polymorphism


def test_extraction_accepts_string_value(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {"features": {"dti": {"rank": 1, "sign": 1, "value": "high", "assumption": "elevated"}}}
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.features["dti"].value == "high"


def test_extraction_accepts_null_value(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {"features": {"dti": {"rank": 1, "sign": 1, "value": None, "assumption": "unnamed value"}}}
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.features["dti"].value is None


def test_extraction_accepts_sign_zero(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {
            "features": {
                "dti": {"rank": 1, "sign": 0, "value": 0.41, "assumption": "mentioned neutrally"}
            }
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.features["dti"].sign == 0


# ----------------------------------------- validation failures


def test_extraction_rejects_sign_outside_minus_one_zero_plus_one(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {"features": {"dti": {"rank": 1, "sign": 2, "value": 0.41, "assumption": ""}}}
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None
    assert failure.passed is False
    assert failure.severity == "advisory"


def test_extraction_rejects_rank_less_than_one(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {"features": {"dti": {"rank": 0, "sign": 1, "value": 0.41, "assumption": ""}}}
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None


def test_extraction_rejects_rank_not_permutation_of_1_to_n(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Ranks must be a dense 1..N permutation. Covers duplicates AND gaps."""
    # Case A: duplicate ranks {1, 1}
    payload_dup = json.dumps(
        {
            "features": {
                "dti": {"rank": 1, "sign": 1, "value": 0.41, "assumption": ""},
                "age": {"rank": 1, "sign": -1, "value": 29, "assumption": ""},
            }
        }
    )
    llm = MockLLMProvider(responses=[payload_dup], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None

    # Case B: gapped ranks {1, 42}
    payload_gap = json.dumps(
        {
            "features": {
                "dti": {"rank": 1, "sign": 1, "value": 0.41, "assumption": ""},
                "age": {"rank": 42, "sign": -1, "value": 29, "assumption": ""},
            }
        }
    )
    llm = MockLLMProvider(responses=[payload_gap], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None


# ----------------------------------------- fence stripping


def test_extraction_strips_markdown_fences(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    fenced = "```json\n" + _ok_payload() + "\n```"
    llm = MockLLMProvider(responses=[fenced], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert failure is None


# ----------------------------------------- parse-failure paths


def test_extraction_malformed_json_returns_none_and_failure_result(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=["this is not json at all"], model_name="mock-judge")
    extraction, response, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None
    assert isinstance(failure, GuardrailResult)
    assert failure.name == "extract_narrative_claims"
    assert failure.severity == "advisory"
    assert failure.passed is False
    assert failure.details["reason"] == "could_not_parse_extraction_output"
    assert failure.details["prompt_version"] == "1"
    assert failure.details["model_name"] == "mock-judge"
    assert "raw" in failure.details
    # Response still available so caller can account for tokens spent.
    assert response.text == "this is not json at all"


def test_extraction_schema_violation_returns_none_and_failure_result(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Valid JSON, wrong shape: missing 'features' top-level key."""
    payload = json.dumps({"not_features": {}})
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None
    assert failure.severity == "advisory"
    assert failure.details["reason"] == "could_not_parse_extraction_output"
    assert failure.details["model_name"] == "mock-judge"


# ----------------------------------------- empty / minimal cases


def test_extraction_empty_features_dict_is_valid(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """A narrative that names no features is valid — empty dict, no error."""
    payload = json.dumps({"features": {}})
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.features == {}
    assert failure is None


def test_feature_claim_valid_model() -> None:
    c = FeatureClaim(rank=1, sign=1, value=0.41, assumption="drives decision")
    assert c.rank == 1
    assert c.sign == 1
    assert c.value == 0.41
    assert c.assumption == "drives decision"
