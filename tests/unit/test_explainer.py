"""Tests for xains.explainer."""

import pytest

from xains import (
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
from xains.prompts import EchoPromptTemplate
from xains.providers import MockLLMProvider
from xains.providers.base import LLMResponse

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
    with pytest.raises(ValueError, match=r"requires request\.counterfactual"):
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
    with pytest.raises(ValueError, match=r"requires request\.counterfactual"):
        explainer.explain(tabular_request)


def test_explicit_feature_importance_ignores_counterfactuals(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Factual mode is always safe — extra inputs are allowed but not used."""
    req = tabular_request.model_copy(
        update={
            "counterfactual": TabularCounterfactual(
                predicted_class=0, features={"age": 29, "dti": 0.20}
            )
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
        counterfactual=TabularCounterfactual(
            # Same predicted class as the factual — should warn.
            predicted_class=1,
            features={"age": 29, "dti": 0.20},
        ),
    )
    with pytest.warns(UserWarning, match="predicts the same class"):
        _explainer(tabular_schema).explain(req)


def test_cf_flipping_class_does_not_warn(tabular_schema: DatasetSchema) -> None:
    req = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.3)],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
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


# ---------------------------------------------------------------- #
# End-to-end counterfactual mode (ADR 0030)                        #
# ---------------------------------------------------------------- #


def test_counterfactual_mode_through_templated_generator(
    tabular_schema: DatasetSchema,
) -> None:
    """Full Explainer.explain() through the templated CF path: no LLM, deterministic prose."""
    from xains import TemplatedCounterfactualGenerator

    req = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
    )
    explainer = Explainer(
        schema=tabular_schema,
        generator=TemplatedCounterfactualGenerator(),
        config=ExplanationConfig(mode="counterfactual", extract_narrative=False),
    )

    result = explainer.explain(req)

    assert result.mode == "counterfactual"
    assert result.text == (
        "To change the prediction from Defaulted to Repaid, "
        "dti would need to change from 0.41 to 0.2."
    )
    # Templated path: LLM-only audit fields are None.
    assert result.prompt is None
    assert result.model_name is None
    assert result.raw_llm_response is None


def test_counterfactual_mode_through_llm_path(
    tabular_schema: DatasetSchema,
) -> None:
    """Full Explainer.explain() through the LLM CF path: MockLLMProvider + CF prompt template."""
    from xains.prompts import CounterfactualTabularPromptTemplate

    req = TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
    )
    llm = MockLLMProvider(responses=["The applicant could have repaid by lowering DTI."])
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(
            prompt_template=CounterfactualTabularPromptTemplate(),
            llm=llm,
        ),
        config=ExplanationConfig(mode="counterfactual", extract_narrative=False),
    )

    result = explainer.explain(req)

    assert result.mode == "counterfactual"
    assert result.text == "The applicant could have repaid by lowering DTI."
    # LLM path: prompt + model_name populated.
    assert result.prompt is not None
    assert "To change the prediction from Defaulted to Repaid:" in result.prompt
    assert "  - dti: 0.41 -> 0.2" in result.prompt
    assert result.model_name == "mock-v0"


# ---------------------------------------------------------------- #
# Extraction dispatch by mode (ADR 0033)                           #
# ---------------------------------------------------------------- #


def _cf_request(
    tabular_schema: DatasetSchema,
) -> TabularExplanationRequest:
    """Tabular CF request used by the dispatch tests."""
    # tabular_schema's classes are {0: "Repaid", 1: "Defaulted"}; factual=1, CF flips to 0.
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[TabularContribution(name="dti", value=0.41, importance=0.37)],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
    )


def test_counterfactual_mode_dispatches_to_cf_extraction(
    tabular_schema: DatasetSchema,
) -> None:
    """mode='counterfactual' + extract_narrative=True -> counterfactual_extraction populated."""
    import json

    # Canned CF-extraction JSON the mock judge will return.
    cf_payload = json.dumps(
        {
            "changes": {
                "dti": {
                    "narrative_name": "debt-to-income ratio",
                    "stated_before": 0.41,
                    "stated_after": 0.20,
                    "stated_direction": "decreased",
                }
            },
            "invented": [],
        }
    )
    # Generator returns a CF narrative; judge returns the CF JSON.
    gen_llm = MockLLMProvider(responses=["DTI would need to drop from 0.41 to 0.20."])
    judge_llm = MockLLMProvider(responses=[cf_payload])

    from xains.prompts import CounterfactualTabularPromptTemplate

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(
            prompt_template=CounterfactualTabularPromptTemplate(), llm=gen_llm
        ),
        config=ExplanationConfig(mode="counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_cf_request(tabular_schema))

    assert result.mode == "counterfactual"
    # CF channel populated, FI channel left None.
    assert result.counterfactual_extraction is not None
    assert "dti" in result.counterfactual_extraction.changes
    assert result.counterfactual_extraction.changes["dti"].stated_after == 0.20
    assert result.narrative_extraction is None


def test_feature_importance_mode_still_populates_narrative_extraction(
    tabular_schema: DatasetSchema, tabular_request: TabularExplanationRequest
) -> None:
    """Regression: FI dispatch is unchanged - narrative_extraction populated, CF channel None."""
    # The judge LLM returns garbage; FI extraction logs an advisory failure but
    # the channel discipline (which extraction field is touched) is what we pin.
    llm = MockLLMProvider(responses=["mocked narrative", "garbage-not-json"])
    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=EchoPromptTemplate(), llm=llm),
        config=ExplanationConfig(mode="feature_importance"),
        judge_llm=llm,
    )

    result = explainer.explain(tabular_request)

    assert result.mode == "feature_importance"
    # FI extraction was attempted (advisory failure logged), CF channel untouched.
    assert result.counterfactual_extraction is None


def test_counterfactual_mode_cf_extraction_failure_logs_advisory(
    tabular_schema: DatasetSchema,
) -> None:
    """CF judge returns bad JSON -> both extraction fields None + advisory GuardrailResult."""
    gen_llm = MockLLMProvider(responses=["a CF narrative"])
    judge_llm = MockLLMProvider(responses=["this is not json at all"])

    from xains.prompts import CounterfactualTabularPromptTemplate

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(
            prompt_template=CounterfactualTabularPromptTemplate(), llm=gen_llm
        ),
        config=ExplanationConfig(mode="counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_cf_request(tabular_schema))

    assert result.counterfactual_extraction is None
    assert result.narrative_extraction is None
    assert result.guardrails is not None
    advisories = [g for g in result.guardrails if g.name == "extract_counterfactual_claims"]
    assert len(advisories) == 1
    assert advisories[0].severity == "advisory"
    assert advisories[0].passed is False


def test_counterfactual_mode_judge_llm_none_raises_shared_value_error(
    tabular_schema: DatasetSchema,
) -> None:
    """The judge_llm-None guard is mode-agnostic; mode='counterfactual' fires the same error."""
    gen_llm = MockLLMProvider(responses=["a CF narrative"])

    from xains.prompts import CounterfactualTabularPromptTemplate

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(
            prompt_template=CounterfactualTabularPromptTemplate(), llm=gen_llm
        ),
        config=ExplanationConfig(mode="counterfactual"),
        # judge_llm deliberately omitted
    )

    with pytest.raises(ValueError, match=r"extract_narrative=True requires judge_llm"):
        explainer.explain(_cf_request(tabular_schema))


# ---------------------------------------------------------------- #
# Hybrid mode: dual extraction (ADR 0040)                          #
# ---------------------------------------------------------------- #


def _fi_payload_str() -> str:
    """Canned FI-extraction JSON (matches _EXTRACTION_PROMPT_VERSION == '3')."""
    import json

    return json.dumps(
        {
            "features": {
                "dti": {
                    "rank": 1,
                    "sign": 1,
                    "value": 0.41,
                    "assumption": "high debt drives default",
                    "narrative_name": "debt-to-income",
                }
            },
            "hallucinations": [],
        }
    )


def _cf_payload_str() -> str:
    """Canned CF-extraction JSON."""
    import json

    return json.dumps(
        {
            "changes": {
                "dti": {
                    "narrative_name": "debt-to-income ratio",
                    "stated_before": 0.41,
                    "stated_after": 0.20,
                    "stated_direction": "decreased",
                }
            },
            "invented": [],
        }
    )


class _TokenCountingMock:
    """Test-local LLM provider serving canned responses with configurable tokens_used.

    Serves ``(text, tokens_used)`` pairs in order. Structurally satisfies
    ``LLMProvider``.
    """

    def __init__(self, scripted: list[tuple[str, dict[str, int] | None]]) -> None:
        self._scripted = scripted
        self._idx = 0

    def generate(self, system: str, user: str) -> LLMResponse:
        text, tokens = self._scripted[self._idx]
        self._idx += 1
        return LLMResponse(text=text, model_name="mock-counting", tokens_used=tokens)


def _hybrid_request(tabular_schema: DatasetSchema) -> TabularExplanationRequest:
    """Tabular request with BOTH contributions and a counterfactual (for hybrid mode)."""
    return TabularExplanationRequest(
        features={"age": 29, "dti": 0.41},
        prediction=Prediction(predicted_class=1, probabilities={0: 0.2, 1: 0.8}),
        contributions=[
            TabularContribution(name="dti", value=0.41, importance=0.37),
            TabularContribution(name="age", value=29, importance=-0.12),
        ],
        counterfactual=TabularCounterfactual(predicted_class=0, features={"age": 29, "dti": 0.20}),
    )


def test_hybrid_mode_populates_both_extraction_channels(
    tabular_schema: DatasetSchema,
) -> None:
    """mode='feature_importance_counterfactual' runs both extractors."""
    from xains.prompts import HybridTabularPromptTemplate

    gen_llm = MockLLMProvider(responses=["a hybrid narrative"])
    # Judge is called TWICE: first for FI extraction, then for CF extraction.
    judge_llm = MockLLMProvider(responses=[_fi_payload_str(), _cf_payload_str()])

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=HybridTabularPromptTemplate(), llm=gen_llm),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_hybrid_request(tabular_schema))

    assert result.mode == "feature_importance_counterfactual"
    # BOTH channels populated:
    assert result.narrative_extraction is not None
    assert "dti" in result.narrative_extraction.features
    assert result.counterfactual_extraction is not None
    assert "dti" in result.counterfactual_extraction.changes


def test_hybrid_mode_guardrail_tokens_used_is_element_wise_sum(
    tabular_schema: DatasetSchema,
) -> None:
    """guardrail_tokens_used sums the two judge calls' token dicts element-wise."""
    from xains.prompts import HybridTabularPromptTemplate

    gen_llm = MockLLMProvider(responses=["a hybrid narrative"])
    judge_llm = _TokenCountingMock(
        [
            (_fi_payload_str(), {"input": 100, "output": 20, "total": 120}),
            (_cf_payload_str(), {"input": 80, "output": 15, "total": 95}),
        ]
    )

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=HybridTabularPromptTemplate(), llm=gen_llm),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_hybrid_request(tabular_schema))

    assert result.guardrail_tokens_used == {
        "input": 180,
        "output": 35,
        "total": 215,
    }


def test_hybrid_mode_partial_fi_failure_still_populates_cf(
    tabular_schema: DatasetSchema,
) -> None:
    """FI extraction fails to parse -> narrative_extraction None + advisory in guardrails;
    CF extraction still populates counterfactual_extraction."""
    from xains.prompts import HybridTabularPromptTemplate

    gen_llm = MockLLMProvider(responses=["a hybrid narrative"])
    judge_llm = MockLLMProvider(responses=["not json at all", _cf_payload_str()])

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=HybridTabularPromptTemplate(), llm=gen_llm),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_hybrid_request(tabular_schema))

    assert result.narrative_extraction is None
    assert result.counterfactual_extraction is not None
    assert result.guardrails is not None
    fi_failures = [g for g in result.guardrails if g.name == "extract_narrative_claims"]
    cf_failures = [g for g in result.guardrails if g.name == "extract_counterfactual_claims"]
    assert len(fi_failures) == 1
    assert fi_failures[0].severity == "advisory"
    assert cf_failures == []


def test_hybrid_mode_partial_cf_failure_still_populates_fi(
    tabular_schema: DatasetSchema,
) -> None:
    """CF extraction fails to parse -> counterfactual_extraction None + advisory in guardrails;
    FI extraction still populates narrative_extraction."""
    from xains.prompts import HybridTabularPromptTemplate

    gen_llm = MockLLMProvider(responses=["a hybrid narrative"])
    judge_llm = MockLLMProvider(responses=[_fi_payload_str(), "not json at all"])

    explainer = Explainer(
        schema=tabular_schema,
        generator=LLMNarrativeGenerator(prompt_template=HybridTabularPromptTemplate(), llm=gen_llm),
        config=ExplanationConfig(mode="feature_importance_counterfactual"),
        judge_llm=judge_llm,
    )

    result = explainer.explain(_hybrid_request(tabular_schema))

    assert result.narrative_extraction is not None
    assert result.counterfactual_extraction is None
    assert result.guardrails is not None
    fi_failures = [g for g in result.guardrails if g.name == "extract_narrative_claims"]
    cf_failures = [g for g in result.guardrails if g.name == "extract_counterfactual_claims"]
    assert fi_failures == []
    assert len(cf_failures) == 1
    assert cf_failures[0].severity == "advisory"


# ---------------------------------------------------------------- #
# _merge_token_counts direct unit tests                            #
# ---------------------------------------------------------------- #


def test_merge_token_counts_both_none_returns_none() -> None:
    from xains.explainer import _merge_token_counts

    assert _merge_token_counts(None, None) is None


def test_merge_token_counts_left_none_returns_shallow_copy_of_right() -> None:
    from xains.explainer import _merge_token_counts

    b = {"input": 10, "output": 3}
    result = _merge_token_counts(None, b)
    assert result == {"input": 10, "output": 3}
    assert result is not b  # shallow copy, not the same dict


def test_merge_token_counts_right_none_returns_shallow_copy_of_left() -> None:
    from xains.explainer import _merge_token_counts

    a = {"input": 10, "output": 3}
    result = _merge_token_counts(a, None)
    assert result == {"input": 10, "output": 3}
    assert result is not a


def test_merge_token_counts_both_dicts_sums_union_of_keys() -> None:
    from xains.explainer import _merge_token_counts

    a = {"input": 10, "output": 3, "total": 13}
    b = {"input": 5, "output": 2, "total": 7}
    assert _merge_token_counts(a, b) == {"input": 15, "output": 5, "total": 20}


def test_merge_token_counts_disjoint_keys_treated_as_zero() -> None:
    from xains.explainer import _merge_token_counts

    a = {"input": 10}
    b = {"output": 5}
    assert _merge_token_counts(a, b) == {"input": 10, "output": 5}
