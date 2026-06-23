"""Unit tests for extract_narrative_claims (prompt v2: resolution-at-extraction).

The LLM resolves narrative mentions to schema feature names at extraction
time. ``NarrativeExtraction.features`` is keyed by schema name; unresolved
mentions go into ``NarrativeExtraction.hallucinations``. See ADR 0007.
"""

import json

import pytest
from pydantic import ValidationError

from xains import (
    DatasetSchema,
    FeatureClaim,
    FeatureSchema,
    Modality,
    NarrativeExtraction,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
)
from xains.guardrails import GuardrailResult, extract_narrative_claims
from xains.providers import MockLLMProvider

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
                    "narrative_name": "dti",
                },
                "age": {
                    "rank": 2,
                    "sign": -1,
                    "value": 29,
                    "assumption": "younger, mildly offsets",
                    "narrative_name": "age",
                },
            },
            "hallucinations": [],
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
    assert extraction.hallucinations == []


def test_extraction_records_prompt_version_2(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[_ok_payload()], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.prompt_version == "2"


def test_extraction_records_model_name_from_response(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[_ok_payload()], model_name="judge-xyz")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.model_name == "judge-xyz"


# ----------------------------------------- resolution + hallucinations


def test_extraction_resolved_features_keyed_by_schema_name(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Narrative says 'debt-to-income'; LLM resolves to schema name 'dti'.

    The features dict is keyed by 'dti'. The original mention is preserved
    in claim.narrative_name; claim.resolved_to equals the schema key.
    """
    payload = json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "pushes toward default",
                    "narrative_name": "debt-to-income",
                }
            },
            "hallucinations": [],
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert "dti" in extraction.features
    assert "debt-to-income" not in extraction.features
    claim = extraction.features["dti"]
    assert claim.narrative_name == "debt-to-income"
    assert claim.resolved_to == "dti"


def test_extraction_hallucinations_recorded_separately(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Unresolved mentions go into hallucinations with resolved_to=None."""
    payload = json.dumps(
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
            "hallucinations": [
                {
                    "rank": 2,
                    "sign": -1,
                    "value": None,
                    "assumption": "stabilizing factor",
                    "narrative_name": "borrowing_pressure",
                }
            ],
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, _ = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert len(extraction.hallucinations) == 1
    halluc = extraction.hallucinations[0]
    assert halluc.narrative_name == "borrowing_pressure"
    assert halluc.resolved_to is None
    assert halluc.rank == 2


def test_extraction_rejects_non_schema_feature_key_in_features_dict(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """features keys must be in schema.features. Non-schema names → failure."""
    payload = json.dumps(
        {
            "features": {
                "bogus": {
                    "rank": 1,
                    "sign": 1,
                    "value": None,
                    "assumption": "made up",
                    "narrative_name": "bogus",
                }
            },
            "hallucinations": [],
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None
    assert failure.severity == "advisory"
    assert failure.passed is False


def test_extraction_rejects_resolved_to_mismatch_with_key() -> None:
    """NarrativeExtraction validator: features[name].resolved_to must equal name."""
    with pytest.raises(ValidationError):
        NarrativeExtraction(
            features={
                "dti": FeatureClaim(rank=1, sign=1, narrative_name="debt", resolved_to="age"),
            },
            hallucinations=[],
            prompt_version="2",
            model_name="x",
        )


# ----------------------------------------- value polymorphism (preserved)


def test_extraction_accepts_string_value(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    payload = json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": "high",
                    "assumption": "elevated",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
        }
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
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": None,
                    "assumption": "unnamed value",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
        }
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
                "dti": {
                    "rank": 1,
                    "sign": 0,
                    "value": 0.41,
                    "assumption": "mentioned neutrally",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
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
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 2,
                    "value": 0.41,
                    "assumption": "",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
        }
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
        {
            "features": {
                "dti": {
                    "rank": 0,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [],
        }
    )
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None


def test_extraction_rejects_rank_not_permutation_of_1_to_n_over_union(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Ranks across features and hallucinations must form a 1..N dense permutation."""
    # Gapped ranks: features {dti:rank=1} + hallucinations [{rank=3}] → [1, 3]
    payload_gap = json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [
                {
                    "rank": 3,
                    "sign": -1,
                    "value": None,
                    "assumption": "",
                    "narrative_name": "ghost",
                }
            ],
        }
    )
    llm = MockLLMProvider(responses=[payload_gap], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is None
    assert failure is not None

    # Duplicate ranks: features {dti:rank=1} + hallucinations [{rank=1}]
    payload_dup = json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "",
                    "narrative_name": "dti",
                }
            },
            "hallucinations": [
                {
                    "rank": 1,
                    "sign": -1,
                    "value": None,
                    "assumption": "",
                    "narrative_name": "ghost",
                }
            ],
        }
    )
    llm = MockLLMProvider(responses=[payload_dup], model_name="mock-judge")
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
    assert failure.details["prompt_version"] == "2"
    assert failure.details["model_name"] == "mock-judge"
    assert "raw" in failure.details
    assert response.text == "this is not json at all"


def test_extraction_schema_violation_returns_none_and_failure_result(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """Valid JSON, wrong shape: missing top-level 'features' key."""
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


# ----------------------------------------- empty / minimal


def test_extraction_empty_features_and_empty_hallucinations_is_valid(
    schema: DatasetSchema, request_obj: TabularExplanationRequest
) -> None:
    """A narrative that names no features at all is valid."""
    payload = json.dumps({"features": {}, "hallucinations": []})
    llm = MockLLMProvider(responses=[payload], model_name="mock-judge")
    extraction, _, failure = extract_narrative_claims(
        text="x", request=request_obj, schema=schema, judge_llm=llm
    )
    assert extraction is not None
    assert extraction.features == {}
    assert extraction.hallucinations == []
    assert failure is None


# ----------------------------------------- FeatureClaim direct


def test_feature_claim_valid_model() -> None:
    c = FeatureClaim(
        rank=1,
        sign=1,
        value=0.41,
        assumption="drives decision",
        narrative_name="debt-to-income",
        resolved_to="dti",
    )
    assert c.rank == 1
    assert c.sign == 1
    assert c.narrative_name == "debt-to-income"
    assert c.resolved_to == "dti"


def test_feature_claim_resolved_to_none_implies_hallucination_role() -> None:
    """``resolved_to=None`` is a valid FeatureClaim shape (used for hallucinations).

    NarrativeExtraction enforces:
      * a claim in ``features`` dict has ``resolved_to == key``,
      * a claim in ``hallucinations`` has ``resolved_to is None``.
    """
    halluc_shape = FeatureClaim(rank=1, sign=1, narrative_name="ghost", resolved_to=None)
    assert halluc_shape.resolved_to is None

    # Hallucination entry with resolved_to set → rejected.
    with pytest.raises(ValidationError):
        NarrativeExtraction(
            features={},
            hallucinations=[
                FeatureClaim(rank=1, sign=1, narrative_name="x", resolved_to="dti"),
            ],
            prompt_version="2",
            model_name="x",
        )

    # Resolved feature with resolved_to=None → rejected.
    with pytest.raises(ValidationError):
        NarrativeExtraction(
            features={
                "dti": FeatureClaim(rank=1, sign=1, narrative_name="dti", resolved_to=None),
            },
            hallucinations=[],
            prompt_version="2",
            model_name="x",
        )
