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
from xainarratives.metrics.narrativity import (
    NarrativityScores,
    ccpr,
    cecpr,
    csr,
    dcpr,
    fdr,
    readability,
    score_narrativity,
    ttcpr,
    vcpr,
)
from xainarratives.metrics.perplexity import (
    DisabledProvider,
    PerplexityProvider,
)
from xainarratives.metrics.perplexity_api import OpenAICompatibleEchoProvider
from xainarratives.metrics.perplexity_hf import HuggingFacePerplexityProvider
from xainarratives.metrics.scorer import ExtractionScores, score_extraction

__all__ = [
    "DisabledProvider",
    "ExtractionScores",
    "HuggingFacePerplexityProvider",
    "NarrativityScores",
    "OpenAICompatibleEchoProvider",
    "PerplexityProvider",
    "ccpr",
    "cecpr",
    "coverage",
    "csr",
    "dcpr",
    "fdr",
    "hallucination_count",
    "rank_correlation",
    "readability",
    "score_extraction",
    "score_narrativity",
    "sign_faithfulness",
    "ttcpr",
    "value_faithfulness",
    "vcpr",
]
