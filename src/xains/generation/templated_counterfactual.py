"""TemplatedCounterfactualGenerator - LLM-free counterfactual narrative.

Consumes :func:`xains.counterfactuals.build_scenarios` to derive structured
per-CF data, then renders one sentence per scenario. The first sentence
leads with the prediction flip (factual to CF class); subsequent sentences
start with ``"Alternatively, "`` (the flip is implicit). Mirrors
:class:`xains.generation.templated.TemplatedNarrativeGenerator`'s shape (ADR 0019)
for the counterfactual path. See ADR 0030.

GenerationResult: ``text`` and ``latency_ms`` populated; ``prompt``,
``model_name``, ``raw_llm_response``, ``tokens_used``, and ``guardrails`` are
all None (templated text has no LLM call to audit).

Tabular only.
"""

import time

from xains.config import ExplanationConfig
from xains.counterfactuals import CounterfactualScenario, build_scenarios
from xains.generation.base import GenerationResult, NarrativeGenerator
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest


class TemplatedCounterfactualGenerator(NarrativeGenerator):
    """Generate a counterfactual narrative from pre-computed CFs, no LLM.

    ``include_method=True`` appends ``" (method: <cf.method>)"`` to a
    scenario's sentence when the CF carries a non-``None`` ``method`` field.
    Default ``False`` keeps narratives method-agnostic, matching
    :class:`xains.prompts.counterfactual_tabular.CounterfactualTabularPromptTemplate`.
    """

    def __init__(self, *, include_method: bool = False) -> None:
        self._include_method = include_method

    def generate(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> GenerationResult:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"TemplatedCounterfactualGenerator requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )

        start = time.perf_counter()
        scenarios = build_scenarios(request, schema)
        sentences = [self._render_scenario(sc, is_first=(i == 0)) for i, sc in enumerate(scenarios)]
        text = " ".join(sentences)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return GenerationResult(
            text=text,
            prompt=None,
            model_name=None,
            raw_llm_response=None,
            tokens_used=None,
            latency_ms=latency_ms,
            guardrails=None,
        )

    def _render_scenario(self, sc: CounterfactualScenario, *, is_first: bool) -> str:
        method_suffix = (
            f" (method: {sc.method})" if self._include_method and sc.method is not None else ""
        )

        if not sc.changes:
            # Degenerate: no changed features (e.g. CF identical to factual).
            if is_first:
                return (
                    f"To change the prediction from {sc.factual_label} to "
                    f"{sc.cf_label}, no feature changes were detected{method_suffix}."
                )
            return f"Alternatively, no feature changes were detected{method_suffix}."

        clauses = [
            f"{chg.name} would need to change from {chg.before} to {chg.after}"
            for chg in sc.changes
        ]
        if len(clauses) == 1:
            joined = clauses[0]
        else:
            joined = ", ".join(clauses[:-1]) + ", and " + clauses[-1]

        if is_first:
            return (
                f"To change the prediction from {sc.factual_label} to "
                f"{sc.cf_label}, {joined}{method_suffix}."
            )
        return f"Alternatively, {joined}{method_suffix}."
