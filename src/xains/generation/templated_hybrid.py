"""TemplatedHybridGenerator - compose FI + CF templated narratives, no LLM.

Per ADR 0037, the ``feature_importance_counterfactual`` mode produces a
two-section narrative: a feature-importance paragraph (why the model
predicted the outcome) followed by a counterfactual paragraph (what would
flip it). This generator composes ``TemplatedNarrativeGenerator`` and
``TemplatedCounterfactualGenerator``, joining their outputs with a blank
line. Tabular only.
"""

import time

from xains.config import ExplanationConfig
from xains.generation.base import GenerationResult, NarrativeGenerator
from xains.generation.templated import TemplatedNarrativeGenerator
from xains.generation.templated_counterfactual import TemplatedCounterfactualGenerator
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest


class TemplatedHybridGenerator(NarrativeGenerator):
    """Compose a feature-importance and a counterfactual templated narrative.

    Produces two sections separated by a blank line: the feature-importance
    narrative (from ``TemplatedNarrativeGenerator``) then the counterfactual
    narrative (from ``TemplatedCounterfactualGenerator``). Deterministic, no
    LLM. Requires a request carrying both contributions and a counterfactual.

    ``method`` is passed to the feature-importance sub-generator and
    ``include_method`` to the counterfactual sub-generator, mirroring their
    own constructors.
    """

    def __init__(self, *, method: str | None = None, include_method: bool = False) -> None:
        self._fi = TemplatedNarrativeGenerator(method=method)
        self._cf = TemplatedCounterfactualGenerator(include_method=include_method)

    def generate(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> GenerationResult:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"TemplatedHybridGenerator requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )
        if request.counterfactual is None:
            raise ValueError(
                "TemplatedHybridGenerator requires request.counterfactual (hybrid "
                "narratives explain both the prediction and its counterfactual)."
            )

        start = time.perf_counter()
        fi_result = self._fi.generate(request, schema, config)
        cf_result = self._cf.generate(request, schema, config)
        text = f"{fi_result.text}\n\n{cf_result.text}"
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
