"""TemplatedNarrativeGenerator - LLM-free narrative from feature-importance contributions.

Reads raw per-feature values from ``request.features`` (not the contribution's
own ``.value`` field), ranks by ``abs(importance)`` descending, and renders one
clause per top-k contribution joined by single spaces. Editable: ``lead_template``
and ``clause_template`` are constructor args, substituted via the same one-pass
primitive used by ``FeatureImportanceTabularPromptTemplate`` (see ADR 0017).

The ``method`` constructor param (None by default) injects ``"{method} "``
(trailing space) before "contribution" in the default clause template; the
library does NOT hardcode any attribution method - pass ``method="SHAP"`` to
reproduce Cedro 2026's paper wording.

GenerationResult: ``text`` and ``latency_ms`` populated; ``prompt``,
``model_name``, ``raw_llm_response``, ``tokens_used``, and ``guardrails`` are
all None (templated text has no LLM call to audit and cannot hallucinate the
class-name guardrail's concern).
"""

import time

from xains._substitution import substitute
from xains.config import ExplanationConfig
from xains.generation.base import GenerationResult, NarrativeGenerator
from xains.schema import DatasetSchema
from xains.types import ExplanationRequest, TabularExplanationRequest

DEFAULT_LEAD_TEMPLATE = "The model predicts {prediction}."

DEFAULT_CLAUSE_TEMPLATE = (
    "The {ordinal} important feature is {name} ({value}), "
    "with a {method}contribution of {importance}."
)


def _ordinal(rank: int) -> str:
    """1 -> 'most', 2 -> 'second most', 3 -> 'third most', n>=4 -> '{n}th most'."""
    if rank == 1:
        return "most"
    if rank == 2:
        return "second most"
    if rank == 3:
        return "third most"
    return f"{rank}th most"


class TemplatedNarrativeGenerator(NarrativeGenerator):
    """Generate a narrative from feature-importance contributions, no LLM."""

    def __init__(
        self,
        *,
        lead_template: str = DEFAULT_LEAD_TEMPLATE,
        clause_template: str = DEFAULT_CLAUSE_TEMPLATE,
        method: str | None = None,
    ) -> None:
        self._lead_template = lead_template
        self._clause_template = clause_template
        self._method = method

    def generate(
        self,
        request: ExplanationRequest,
        schema: DatasetSchema,
        config: ExplanationConfig,
    ) -> GenerationResult:
        if not isinstance(request, TabularExplanationRequest):
            raise TypeError(
                f"TemplatedNarrativeGenerator requires a TabularExplanationRequest, "
                f"got {type(request).__name__}."
            )

        start = time.perf_counter()

        prediction_label = schema.target.classes[request.prediction.predicted_class]
        lead = substitute(self._lead_template, {"prediction": str(prediction_label)})

        ranked = sorted(request.contributions, key=lambda c: abs(c.importance), reverse=True)[
            : config.top_k_features
        ]
        method_prefix = f"{self._method} " if self._method else ""

        clauses: list[str] = []
        for rank, c in enumerate(ranked, start=1):
            if c.name not in request.features:
                raise ValueError(
                    f"Contribution names feature {c.name!r} but it is not in "
                    "request.features; templated explanations require the raw "
                    "value for every contributed feature."
                )
            value = str(request.features[c.name])
            sign = "+" if c.importance >= 0 else "-"
            importance = f"{sign}{abs(c.importance):g}"
            clauses.append(
                substitute(
                    self._clause_template,
                    {
                        "ordinal": _ordinal(rank),
                        "name": c.name,
                        "value": value,
                        "importance": importance,
                        "method": method_prefix,
                    },
                )
            )

        text = " ".join([lead, *clauses])
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
