"""Explainer integration tests for guardrails + narrative extraction.

Covers gating (run_guardrails, extract_narrative), judge_llm routing,
non-tabular modality skips, and the token-accounting split between
``tokens_used`` (generator) and ``guardrail_tokens_used`` (extraction).

MockLLMProvider does not populate ``tokens_used``; a small structural
subclass (``_TokenMock``) is defined locally for the token-accounting tests.
MockLLMProvider is deliberately NOT modified in this PR.
"""

import json

import pytest

from xainarratives import (
    DatasetSchema,
    Explainer,
    ExplanationConfig,
    ExplanationResult,
    FeatureSchema,
    Modality,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
    TargetSchema,
    TextExplanationRequest,
    TextSpec,
    TokenContribution,
)
from xainarratives.guardrails import NarrativeExtraction
from xainarratives.prompts.base import PromptTemplate
from xainarratives.providers import LLMResponse, MockLLMProvider

# ------------------------------------------------------ helpers


class _FakePromptTemplate(PromptTemplate):
    """Minimal concrete PromptTemplate for Explainer tests."""

    def render(self, request, schema, config):
        return ("sys", "user")


def _valid_extraction_json() -> str:
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


class _TokenMock:
    """Structural LLMProvider returning scripted text + tokens_used per call."""

    def __init__(
        self,
        scripted: list[tuple[str, dict[str, int]]],
        model_name: str = "tok-mock",
    ) -> None:
        self._scripted = scripted
        self._index = 0
        self._model_name = model_name

    def generate(self, system: str, user: str) -> LLMResponse:
        text, tokens = self._scripted[self._index]
        self._index += 1
        return LLMResponse(text=text, model_name=self._model_name, tokens_used=tokens)


NARRATIVE = "Higher dti pushed the applicant toward Defaulted."


# ------------------------------------------------------ fixtures


@pytest.fixture
def tab_schema() -> DatasetSchema:
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
def tab_request() -> TabularExplanationRequest:
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
        ],
    )


@pytest.fixture
def text_schema() -> DatasetSchema:
    return DatasetSchema(
        modality=Modality.TEXT,
        name="sentiment",
        description="Reviews.",
        target=TargetSchema(
            name="polarity",
            description="Sentiment polarity.",
            classes={"pos": "Positive", "neg": "Negative"},
        ),
        text_spec=TextSpec(language="en"),
    )


@pytest.fixture
def text_request() -> TextExplanationRequest:
    return TextExplanationRequest(
        text="great product",
        prediction=Prediction(predicted_class="pos"),
        contributions=[TokenContribution(token="great", span=(0, 5), importance=0.5)],
    )


# ------------------------------------------------------ gating


def test_guardrails_disabled_sets_all_three_fields_none(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[NARRATIVE], model_name="gen")
    config = ExplanationConfig(run_guardrails=False)
    expl = Explainer(
        schema=tab_schema,
        llm=llm,
        prompt_template=_FakePromptTemplate(),
        config=config,
    )
    result = expl.explain(tab_request)
    assert isinstance(result, ExplanationResult)
    assert result.guardrails is None
    assert result.narrative_extraction is None
    assert result.guardrail_tokens_used is None


def test_rule_based_run_by_default(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[NARRATIVE], model_name="gen")
    config = ExplanationConfig(extract_narrative=False)
    expl = Explainer(
        schema=tab_schema,
        llm=llm,
        prompt_template=_FakePromptTemplate(),
        config=config,
    )
    result = expl.explain(tab_request)
    assert result.guardrails is not None
    assert len(result.guardrails) == 1
    assert result.guardrails[0].name == "class_name_mentioned"
    assert result.narrative_extraction is None
    assert result.guardrail_tokens_used is None


def test_extraction_runs_by_default_for_tabular(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[NARRATIVE, _valid_extraction_json()], model_name="gen")
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert isinstance(result.narrative_extraction, NarrativeExtraction)
    assert "dti" in result.narrative_extraction.features


def test_extraction_disabled_runs_only_rule_based(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[NARRATIVE], model_name="gen")
    config = ExplanationConfig(extract_narrative=False)
    expl = Explainer(
        schema=tab_schema,
        llm=llm,
        prompt_template=_FakePromptTemplate(),
        config=config,
    )
    expl.explain(tab_request)
    assert llm._index == 1


# ------------------------------------------------------ judge_llm routing


def test_judge_llm_defaults_to_generator_llm_when_not_supplied(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=[NARRATIVE, _valid_extraction_json()], model_name="gen")
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert llm._index == 2
    assert result.narrative_extraction is not None


def test_judge_llm_uses_supplied_provider_when_set(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    gen = MockLLMProvider(responses=[NARRATIVE], model_name="gen")
    judge = MockLLMProvider(responses=[_valid_extraction_json()], model_name="judge")
    expl = Explainer(
        schema=tab_schema,
        llm=gen,
        prompt_template=_FakePromptTemplate(),
        judge_llm=judge,
    )
    result = expl.explain(tab_request)
    assert gen._index == 1
    assert judge._index == 1
    assert result.narrative_extraction is not None


# ------------------------------------------------------ modality


def test_non_tabular_request_skips_rule1_and_extraction_runs_rule2(
    text_schema: DatasetSchema, text_request: TextExplanationRequest
) -> None:
    llm = MockLLMProvider(responses=["This review is Positive overall."], model_name="gen")
    expl = Explainer(schema=text_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(text_request)
    assert result.guardrails is not None
    assert len(result.guardrails) == 1
    assert result.guardrails[0].name == "class_name_mentioned"
    assert result.narrative_extraction is None
    assert result.guardrail_tokens_used is None


# ------------------------------------------------------ token accounting


def test_generator_tokens_in_tokens_used_only(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    gen_tokens = {"input": 10, "output": 20, "total": 30}
    judge_tokens = {"input": 100, "output": 200, "total": 300}
    llm = _TokenMock([(NARRATIVE, gen_tokens), (_valid_extraction_json(), judge_tokens)])
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert result.tokens_used == gen_tokens
    assert result.tokens_used != judge_tokens


def test_extraction_tokens_in_guardrail_tokens_used_only(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    gen_tokens = {"input": 10, "output": 20, "total": 30}
    judge_tokens = {"input": 100, "output": 200, "total": 300}
    llm = _TokenMock([(NARRATIVE, gen_tokens), (_valid_extraction_json(), judge_tokens)])
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert result.guardrail_tokens_used == judge_tokens
    assert result.guardrail_tokens_used != gen_tokens


def test_extraction_parse_failure_still_records_tokens(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    gen_tokens = {"input": 10, "output": 20, "total": 30}
    judge_tokens = {"input": 100, "output": 200, "total": 300}
    llm = _TokenMock([(NARRATIVE, gen_tokens), ("not json at all", judge_tokens)])
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert result.narrative_extraction is None
    assert result.guardrail_tokens_used == judge_tokens
    assert result.guardrails is not None
    failures = [
        g for g in result.guardrails if g.name == "extract_narrative_claims" and not g.passed
    ]
    assert len(failures) == 1


def test_tokens_used_schema_conforms_to_adr_0005(
    tab_schema: DatasetSchema, tab_request: TabularExplanationRequest
) -> None:
    gen_tokens = {"input": 10, "output": 20, "total": 30}
    judge_tokens = {"input": 100, "output": 200, "total": 300}
    llm = _TokenMock([(NARRATIVE, gen_tokens), (_valid_extraction_json(), judge_tokens)])
    expl = Explainer(schema=tab_schema, llm=llm, prompt_template=_FakePromptTemplate())
    result = expl.explain(tab_request)
    assert result.tokens_used is not None
    assert set(result.tokens_used.keys()) == {"input", "output", "total"}
    assert result.guardrail_tokens_used is not None
    assert set(result.guardrail_tokens_used.keys()) == {"input", "output", "total"}
