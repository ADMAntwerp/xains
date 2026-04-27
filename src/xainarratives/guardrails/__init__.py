"""Guardrails + narrative-extraction layer."""

from xainarratives.guardrails.extraction import extract_narrative_claims
from xainarratives.guardrails.rules import class_name_mentioned
from xainarratives.guardrails.types import (
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
