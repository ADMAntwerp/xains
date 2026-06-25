"""Guardrails + narrative-extraction layer."""

from xains.guardrails.extraction import (
    extract_counterfactual_claims,
    extract_narrative_claims,
)
from xains.guardrails.rules import class_name_mentioned
from xains.guardrails.types import (
    CounterfactualExtraction,
    CounterfactualFeatureClaim,
    FeatureClaim,
    GuardrailResult,
    NarrativeExtraction,
)

__all__ = [
    "CounterfactualExtraction",
    "CounterfactualFeatureClaim",
    "FeatureClaim",
    "GuardrailResult",
    "NarrativeExtraction",
    "class_name_mentioned",
    "extract_counterfactual_claims",
    "extract_narrative_claims",
]
