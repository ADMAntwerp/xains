"""Direct unit tests for LLMNarrativeGenerator.generate().

These are NET-NEW tests for the generator abstraction. The behavior-
preservation contract for the Explainer LLM path is covered by the
existing 283 tests (whose Explainer construction sites get migrated
to the new signature; their assertions stay byte-identical).
"""

from xainarratives import (
    DatasetSchema,
    ExplanationConfig,
    TabularExplanationRequest,
)
from xainarratives.generation import GenerationResult, LLMNarrativeGenerator
from xainarratives.prompts import EchoPromptTemplate
from xainarratives.providers import MockLLMProvider


def test_llm_generator_returns_populated_generation_result(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """All audit fields populate on the LLM path; class_name_mentioned runs."""
    generator = LLMNarrativeGenerator(
        prompt_template=EchoPromptTemplate(),
        llm=MockLLMProvider(responses=["the applicant Defaulted on the loan"]),
    )
    config = ExplanationConfig(mode="feature_importance")

    result = generator.generate(tabular_request, tabular_schema, config)

    assert isinstance(result, GenerationResult)
    assert result.text == "the applicant Defaulted on the loan"
    assert result.prompt is not None
    assert "SYSTEM:" in result.prompt and "USER:" in result.prompt
    assert result.model_name == "mock-v0"
    assert result.raw_llm_response == "the applicant Defaulted on the loan"
    assert result.latency_ms is not None and result.latency_ms >= 0.0
    # run_guardrails defaults to True → class_name_mentioned fires.
    assert result.guardrails is not None
    assert len(result.guardrails) == 1
    assert result.guardrails[0].name == "class_name_mentioned"


def test_llm_generator_run_guardrails_false_omits_guardrails(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """``config.run_guardrails=False`` → guardrails is None; other fields still populate."""
    generator = LLMNarrativeGenerator(
        prompt_template=EchoPromptTemplate(),
        llm=MockLLMProvider(responses=["any text"]),
    )
    config = ExplanationConfig(mode="feature_importance", run_guardrails=False)

    result = generator.generate(tabular_request, tabular_schema, config)

    assert result.guardrails is None
    assert result.text == "any text"
    assert result.prompt is not None
    assert result.latency_ms is not None


def test_llm_generator_passes_request_prediction_to_class_name_mentioned(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """class_name_mentioned receives the request's predicted class.

    tabular_request.predicted_class=1 → label='Defaulted' in tabular_schema.
    LLM output that mentions 'Defaulted' → guardrail passes; output that
    doesn't → guardrail fails. Pins down that request.prediction reaches
    the guardrail correctly through the generator boundary.
    """
    config = ExplanationConfig(mode="feature_importance")

    gen_pass = LLMNarrativeGenerator(
        prompt_template=EchoPromptTemplate(),
        llm=MockLLMProvider(responses=["The applicant Defaulted within 24 months."]),
    )
    res_pass = gen_pass.generate(tabular_request, tabular_schema, config)
    assert res_pass.guardrails is not None
    assert res_pass.guardrails[0].name == "class_name_mentioned"
    assert res_pass.guardrails[0].passed is True

    gen_fail = LLMNarrativeGenerator(
        prompt_template=EchoPromptTemplate(),
        llm=MockLLMProvider(responses=["The application was reviewed positively."]),
    )
    res_fail = gen_fail.generate(tabular_request, tabular_schema, config)
    assert res_fail.guardrails is not None
    assert res_fail.guardrails[0].name == "class_name_mentioned"
    assert res_fail.guardrails[0].passed is False
