"""Narrative-claim extraction (prompt v2: resolution-at-extraction).

A single LLM call inspects the generator's narrative and returns structured
per-feature claims. The LLM resolves each narrative mention to a schema
feature name (emitted under ``features[<schema_name>]``) or marks it as a
hallucination (emitted in ``hallucinations[]``). See ADR 0007.

Schema follows Ichmoukhamedov et al. 2024 ("How good is my story? Towards
quantitative metrics for evaluating LLM-generated XAI narratives",
arXiv:2412.10220), Fig. 3, with the resolution channel split added in
``_EXTRACTION_PROMPT_VERSION = "2"``.
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xains.guardrails.types import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    FeatureClaim,
    GuardrailResult,
    NarrativeExtraction,
)
from xains.providers.base import LLMProvider, LLMResponse
from xains.schema import DatasetSchema
from xains.types import TabularExplanationRequest

_EXTRACTION_PROMPT_VERSION = "2"

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class _ExtractionClaim(BaseModel):
    """Wire-format claim from the LLM (no ``resolved_to`` — synthesized)."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    sign: Literal[-1, 0, 1]
    value: Any = None
    assumption: str = ""
    narrative_name: str = Field(min_length=1)


class _ExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, _ExtractionClaim] = Field(default_factory=dict)
    hallucinations: list[_ExtractionClaim] = Field(default_factory=list)


_SYSTEM_PROMPT = f"""\
You extract structured claims from model-explanation narratives and resolve
each claim to a schema feature name.

Output exactly one JSON object matching the schema below. No prose, no
markdown, no comments.

Schema:
{{
  "features": {{
    "<schema_feature_name>": {{
      "rank": <int >= 1>,
      "sign": -1 | 0 | 1,
      "value": <number | string | null>,
      "assumption": <string>,
      "narrative_name": <string, the original mention from the narrative>
    }}
  }},
  "hallucinations": [
    {{
      "rank": <int >= 1>,
      "sign": -1 | 0 | 1,
      "value": <number | string | null>,
      "assumption": <string>,
      "narrative_name": <string, the original mention from the narrative>
    }}
  ]
}}

Definitions:
- ``rank``: 1-indexed position in the narrative's order of importance — the
  order the text describes, not the model's importance ranking. Ranks across
  features AND hallucinations together must form a dense permutation of
  1..N where N = total mentions.
- ``sign``: direction the narrative claims for this feature (+1 increases
  the predicted class, -1 decreases it, 0 unspecified).
- ``value``: the feature value as cited in the text, or null if unnamed.
- ``assumption``: the narrative's reason / clause about this feature.
- ``narrative_name``: the feature name as it appears in the narrative
  (synonyms preserved verbatim).

Resolution rule: for each feature mention in the narrative, map it to a
schema feature from the resolution vocabulary in the user prompt. Use the
exact schema name as the dict key under ``features``. If you cannot
confidently identify the schema feature for a mention, place the entry in
``hallucinations`` instead. **When in doubt, hallucinate** — false-positive
hallucinations are visible in scoring; silent misattributions are not.

A narrative that names no features yields
{{"features": {{}}, "hallucinations": []}}.

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
        "## Resolution vocabulary (schema features)\n"
        "Use these exact names as keys under `features`. Any mention you "
        "cannot confidently map to one of these names goes under "
        "`hallucinations` instead.\n\n"
        f"{schema_block}\n\n"
        "## Prediction\n" + "\n".join(pred_lines) + "\n\n"
        "## Contributions\n" + "\n".join(contrib_lines) + "\n\n"
        "## Explanation text under review\n"
        "```\n"
        f"{text}\n"
        "```\n\n"
        "## Task\n"
        "Extract one entry per feature the narrative mentions. Resolve each "
        "to a schema name from the vocabulary above, or mark as a "
        "hallucination. Return exactly one JSON object matching the schema "
        "in the system prompt."
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
    """Single LLM call → structured per-feature claims with schema resolution.

    Returns ``(extraction, response, failure)``:

    * On success: ``(NarrativeExtraction, response, None)``.
    * On parse / validation failure: ``(None, response, GuardrailResult)``;
      the caller still sees ``response`` so token spend can be accounted for.

    Failure paths:
    * Malformed JSON.
    * JSON shape that does not match the wire schema.
    * A ``features`` key that is not in the schema's resolution vocabulary.
    * Pydantic-level validation failure on the final ``NarrativeExtraction``
      (rank-permutation violation, etc.).
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

    schema_vocab = {f.name for f in schema.features} if schema.features else set()
    if any(name not in schema_vocab for name in parsed.features):
        return None, response, _failure_result(response, response.text)

    try:
        extraction = NarrativeExtraction(
            features={
                schema_name: FeatureClaim(
                    rank=claim.rank,
                    sign=claim.sign,
                    value=claim.value,
                    assumption=claim.assumption,
                    narrative_name=claim.narrative_name,
                    resolved_to=schema_name,
                )
                for schema_name, claim in parsed.features.items()
            },
            hallucinations=[
                FeatureClaim(
                    rank=h.rank,
                    sign=h.sign,
                    value=h.value,
                    assumption=h.assumption,
                    narrative_name=h.narrative_name,
                    resolved_to=None,
                )
                for h in parsed.hallucinations
            ],
            prompt_version=_EXTRACTION_PROMPT_VERSION,
            model_name=response.model_name,
        )
    except ValidationError:
        return None, response, _failure_result(response, response.text)

    return extraction, response, None


# ========================================================================
# Counterfactual extraction (ADR 0031)
# ========================================================================

_CF_EXTRACTION_PROMPT_VERSION = "1"


class _CounterfactualClaim(BaseModel):
    """Wire-format CF claim from the LLM (no resolved_to — synthesized post-LLM)."""

    model_config = ConfigDict(extra="forbid")

    narrative_name: str = Field(min_length=1)
    stated_before: Any = None
    stated_after: Any = None
    stated_direction: str | None = None


class _CFExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: dict[str, _CounterfactualClaim] = Field(default_factory=dict)
    invented: list[_CounterfactualClaim] = Field(default_factory=list)


_CF_SYSTEM_PROMPT = f"""\
You extract structured change-claims from counterfactual-explanation
narratives and resolve each claim to a schema feature name.

A counterfactual narrative describes hypothetical changes to a model's
input that would flip its prediction. Each "change-claim" is a clause
stating that a feature would change (synonyms: "X must increase",
"lower X to ...", "X would need to change from ... to ...").

Output exactly one JSON object matching the schema below. No prose, no
markdown, no comments.

Schema:
{{
  "changes": {{
    "<schema_feature_name>": {{
      "narrative_name": <string, the original mention from the narrative>,
      "stated_before": <number | string | null, the value the narrative attributes BEFORE the change, or null if not stated>,
      "stated_after": <number | string | null, the value the narrative attributes AFTER the change, or null if not stated>,
      "stated_direction": <string | null, e.g. "increased", "decreased", "changed", or null if not stated>
    }}
  }},
  "invented": [
    {{
      "narrative_name": <string>,
      "stated_before": <number | string | null>,
      "stated_after": <number | string | null>,
      "stated_direction": <string | null>
    }}
  ]
}}

Definitions:
- ``narrative_name``: the feature name as it appears in the narrative,
  preserved verbatim (synonyms included).
- ``stated_before`` / ``stated_after``: the values the narrative cites
  for the change. Use ``null`` when the narrative does not state a value
  for that side. Extract literally from the narrative; do not infer
  numbers from context.
- ``stated_direction``: the direction word the narrative uses
  ("increased", "decreased", "changed", "lowered", ...), or ``null``
  when no direction word is used.

Resolution rule: for each change-claim in the narrative, map the
mentioned feature to a schema feature from the resolution vocabulary in
the user prompt. Use the exact schema name as the dict key under
``changes``. If you cannot confidently identify the schema feature for
a mention, place the entry in ``invented`` instead. **When in doubt,
mark as invented** - false-positive inventions are visible in scoring;
silent misattributions are not.

A narrative that names no counterfactual changes yields
{{"changes": {{}}, "invented": []}}.

Prompt version: {_CF_EXTRACTION_PROMPT_VERSION}
"""


def _build_cf_user_prompt(
    text: str,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
) -> str:
    feature_lines: list[str] = []
    if schema.features:
        for f in schema.features:
            unit = f" [{f.unit}]" if f.unit else ""
            feature_lines.append(f"- {f.name} ({f.dtype}){unit}: {f.description}")
    schema_block = "\n".join(feature_lines) if feature_lines else "(none)"

    label = schema.target.classes[request.prediction.predicted_class]
    pred_line = f"- predicted_class: {request.prediction.predicted_class} ({label})"

    return (
        "## Resolution vocabulary (schema features)\n"
        "Use these exact names as keys under `changes`. Any mention you "
        "cannot confidently map to one of these names goes under "
        "`invented` instead.\n\n"
        f"{schema_block}\n\n"
        "## Factual prediction\n"
        f"{pred_line}\n\n"
        "## Explanation text under review\n"
        "```\n"
        f"{text}\n"
        "```\n\n"
        "## Task\n"
        "Extract one entry per counterfactual change the narrative "
        "mentions. For each, capture the feature's stated `before` and "
        "`after` values (or null if the narrative does not state them) "
        'and any stated direction word (e.g. "increased", "decreased", '
        '"changed"). Resolve each mention to a schema name from the '
        "vocabulary above, or place it under `invented`. Return exactly "
        "one JSON object matching the schema in the system prompt."
    )


def _cf_failure_result(response: LLMResponse, raw: str) -> GuardrailResult:
    return GuardrailResult(
        name="extract_counterfactual_claims",
        severity="advisory",
        passed=False,
        details={
            "reason": "could_not_parse_extraction_output",
            "raw": raw[:2000],
            "prompt_version": _CF_EXTRACTION_PROMPT_VERSION,
            "model_name": response.model_name,
        },
    )


def extract_counterfactual_claims(
    text: str,
    request: TabularExplanationRequest,
    schema: DatasetSchema,
    judge_llm: LLMProvider,
) -> tuple[CounterfactualExtraction | None, LLMResponse, GuardrailResult | None]:
    """Single LLM call → structured per-change claims with schema resolution.

    Mirrors :func:`extract_narrative_claims` shape. Returns
    ``(extraction, response, failure)``:

    * On success: ``(CounterfactualExtraction, response, None)``.
    * On parse / validation failure: ``(None, response, GuardrailResult)``;
      caller still sees ``response`` so token spend can be accounted for.

    Failure paths: malformed JSON, JSON shape that does not match the
    wire schema, a ``changes`` key not in the schema's resolution
    vocabulary, or pydantic-level validation failure on the final
    ``CounterfactualExtraction``. See ADR 0031.
    """
    user = _build_cf_user_prompt(text, request, schema)
    response = judge_llm.generate(_CF_SYSTEM_PROMPT, user)

    cleaned = _strip_fences(response.text)
    try:
        raw_obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, response, _cf_failure_result(response, response.text)

    try:
        parsed = _CFExtractionSchema.model_validate(raw_obj)
    except ValidationError:
        return None, response, _cf_failure_result(response, response.text)

    schema_vocab = {f.name for f in schema.features} if schema.features else set()
    if any(name not in schema_vocab for name in parsed.changes):
        return None, response, _cf_failure_result(response, response.text)

    try:
        extraction = CounterfactualExtraction(
            changes={
                schema_name: CounterfactualFeatureClaim(
                    narrative_name=claim.narrative_name,
                    resolved_to=schema_name,
                    stated_before=claim.stated_before,
                    stated_after=claim.stated_after,
                    stated_direction=claim.stated_direction,
                )
                for schema_name, claim in parsed.changes.items()
            },
            invented=[
                CounterfactualFeatureClaim(
                    narrative_name=h.narrative_name,
                    resolved_to=None,
                    stated_before=h.stated_before,
                    stated_after=h.stated_after,
                    stated_direction=h.stated_direction,
                )
                for h in parsed.invented
            ],
            prompt_version=_CF_EXTRACTION_PROMPT_VERSION,
            model_name=response.model_name,
        )
    except ValidationError:
        return None, response, _cf_failure_result(response, response.text)

    return extraction, response, None
