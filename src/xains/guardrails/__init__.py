"""Guardrails + narrative-extraction layer."""

from xains.guardrails.extraction import extract_narrative_claims
from xains.guardrails.rules import class_name_mentioned
from xains.guardrails.types import (
    FeatureClaim,
    GuardrailResult,
    NarrativeExtraction,
)

__all__ = [
    "FeatureClaim",
    "GuardrailResult",
    "NarrativeExtraction",
    "class_name_mentioned",
    "extract_narrative_claims",
]
