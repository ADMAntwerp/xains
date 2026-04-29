"""Metrics: deterministic scoring on top of NarrativeExtraction.

All metric functions are pure (extraction + request/schema → score|None).
The single I/O exception is the ``PerplexityProvider`` Protocol, gated
behind explicit caller-supplied implementations.
"""

from xainarratives.metrics.coverage import coverage, hallucination_count
from xainarratives.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xainarratives.metrics.narrativity import readability
from xainarratives.metrics.perplexity import (
    APIPerplexityProvider,
    DisabledProvider,
    PerplexityProvider,
)
from xainarratives.metrics.scorer import ExtractionScores, score_extraction

__all__ = [
    "APIPerplexityProvider",
    "DisabledProvider",
    "ExtractionScores",
    "PerplexityProvider",
    "coverage",
    "hallucination_count",
    "rank_correlation",
    "readability",
    "score_extraction",
    "sign_faithfulness",
    "value_faithfulness",
]
