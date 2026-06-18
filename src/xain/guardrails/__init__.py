"""Guardrails + narrative-extraction layer."""

from xain.guardrails.extraction import extract_narrative_claims
from xain.guardrails.rules import class_name_mentioned
from xain.guardrails.types import (
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
