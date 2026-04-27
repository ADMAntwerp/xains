"""Narrative-claim extraction.

A single LLM call inspects the generator's narrative and returns a structured
JSON object listing each feature the narrative mentions, with the ``rank``,
``sign``, ``value``, and ``assumption`` the text claims. Scoring is downstream
and deterministic (PR 5).

The schema follows Ichmoukhamedov et al. 2024 ("How good is my story? Towards
quantitative metrics for evaluating LLM-generated XAI narratives",
arXiv:2412.10220), Fig. 3. Field names match the paper verbatim.

Bumping ``_EXTRACTION_PROMPT_VERSION`` whenever the prompt text or the output
schema changes lets PR 5 distinguish extractions produced under different
contracts.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xainarratives.guardrails.types import (
    FeatureClaim,
    GuardrailResult,
    NarrativeExtraction,
)
from xainarratives.providers.base import LLMProvider, LLMResponse
from xainarratives.schema import DatasetSchema
from xainarratives.types import TabularExplanationRequest

_EXTRACTION_PROMPT_VERSION = "1"

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class _ExtractionFeatureClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    sign: Literal[-1, 0, 1]
    value: Any = None
    assumption: str = ""


class _ExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, _ExtractionFeatureClaim] = Field(default_factory=dict)


_SYSTEM_PROMPT = f"""\
You extract structured claims from model-explanation narratives.

Output exactly one JSON object matching the schema below. No prose, no
markdown, no comments.

Schema:
{{
  "features": {{
    "<feature_name_as_in_narrative>": {{
      "rank": <int >= 1>,
      "sign": -1 | 0 | 1,
      "value": <number | string | null>,
      "assumption": <string>
    }}
  }}
}}

Definitions:
- ``rank``: 1-indexed position in the narrative's order of importance — the
  order the text describes, not the model's importance ranking. Ranks across
  features must form a dense permutation of 1..N.
- ``sign``: direction the narrative claims for this feature (+1 increases
  the predicted class, -1 decreases it, 0 unspecified).
- ``value``: the feature value as cited in the text, or null if unnamed.
- ``assumption``: the narrative's reason / clause about this feature.

Use the feature name as it appears in the narrative (synonyms preserved).
A narrative that names no features yields {{"features": {{}}}}.

Prompt version: {_EXTRACTION_PROMPT_VERSION}
"""


def _build_user_prompt(text: str, request: TabularExplanationRequest, schema: DatasetSchema) -> str:
    feature_lines: list[str] = []
    if schema.features:
        for f in schema.features:
            unit = f" [{f.unit}]" if f.unit else ""
            feature_lines.append(f"- {f.name} ({f.dtype}){unit}: {f.description}")
    schema_block = "\n".join(feature_lines) if feature_lines else "(none)"

    label = schema.target.classes[request.prediction.predicted_class]
    pred_lines = [f"- predicted_class: {request.prediction.predicted_class} ({label})"]
    if request.prediction.probabilities is not None:
        pred_lines.append(f"- probabilities: {request.prediction.probabilities}")

    contrib_lines = [
        f"- {c.name}: value={c.value}, importance={c.importance}" for c in request.contributions
    ]

    return (
        "## Schema features\n"
        f"{schema_block}\n\n"
        "## Prediction\n" + "\n".join(pred_lines) + "\n\n"
        "## Contributions\n" + "\n".join(contrib_lines) + "\n\n"
        "## Explanation text under review\n"
        "```\n"
        f"{text}\n"
        "```\n\n"
        "## Task\n"
        "Extract one entry per feature the narrative mentions. Return exactly "
        "one JSON object matching the schema in the system prompt."
    )


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    m = _FENCE_RE.match(s)
    return m.group(1) if m else s


def _failure_result(response: LLMResponse, raw: str) -> GuardrailResult:
    return GuardrailResult(
        name="extract_narrative_claims",
        severity="advisory",
        passed=False,
        details={
            "reason": "could_not_parse_extraction_output",
            "raw": raw[:2000],
            "prompt_version": _EXTRACTION_PROMPT_VERSION,
            "model_name": response.model_name,
        },
    )


def extract_narrative_claims(
    text: str,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
    judge_llm: LLMProvider,
) -> tuple[NarrativeExtraction | None, LLMResponse, GuardrailResult | None]:
    """Single LLM call → structured per-feature claims.

    Returns ``(extraction, response, failure)``:

    * On success: ``(NarrativeExtraction, response, None)``.
    * On parse / validation failure: ``(None, response, GuardrailResult)``;
      the caller still sees ``response`` so token spend can be accounted for.
    """
    user = _build_user_prompt(text, request, schema)
    response = judge_llm.generate(_SYSTEM_PROMPT, user)

    cleaned = _strip_fences(response.text)
    try:
        raw_obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, response, _failure_result(response, response.text)

    try:
        parsed = _ExtractionSchema.model_validate(raw_obj)
    except ValidationError:
        return None, response, _failure_result(response, response.text)

    try:
        extraction = NarrativeExtraction(
            features={
                name: FeatureClaim(
                    rank=claim.rank,
                    sign=claim.sign,
                    value=claim.value,
                    assumption=claim.assumption,
                )
                for name, claim in parsed.features.items()
            },
            prompt_version=_EXTRACTION_PROMPT_VERSION,
            model_name=response.model_name,
        )
    except ValidationError:
        return None, response, _failure_result(response, response.text)

    return extraction, response, None
