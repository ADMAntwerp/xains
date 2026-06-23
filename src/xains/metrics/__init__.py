"""Metrics: deterministic scoring on top of NarrativeExtraction.

All metric functions are pure (extraction + request/schema → score|None).
The single I/O exception is the ``PerplexityProvider`` Protocol, gated
behind explicit caller-supplied implementations.
"""

from xains.metrics.coverage import coverage, hallucination_count
from xains.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xains.metrics.grader import ExtractionGrades, grade_extraction
from xains.metrics.narrativity import (
    NarrativityGrades,
    ccpr,
    cecpr,
    csr,
    dcpr,
    fdr,
    grade_narrativity,
    readability,
    ttcpr,
    vcpr,
)
from xains.metrics.perplexity import (
    DisabledProvider,
    PerplexityProvider,
)
from xains.metrics.perplexity_api import OpenAICompatibleEchoProvider
from xains.metrics.perplexity_hf import HuggingFacePerplexityProvider

__all__ = [
    "DisabledProvider",
    "ExtractionGrades",
    "HuggingFacePerplexityProvider",
    "NarrativityGrades",
    "OpenAICompatibleEchoProvider",
    "PerplexityProvider",
    "ccpr",
    "cecpr",
    "coverage",
    "csr",
    "dcpr",
    "fdr",
    "grade_extraction",
    "grade_narrativity",
    "hallucination_count",
    "rank_correlation",
    "readability",
    "sign_faithfulness",
    "ttcpr",
    "value_faithfulness",
    "vcpr",
]
