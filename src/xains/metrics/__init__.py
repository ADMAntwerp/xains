"""Metrics: deterministic scoring on top of NarrativeExtraction.

All metric functions are pure (extraction + request/schema → score|None).
The single I/O exception is the ``PerplexityProvider`` Protocol, gated
behind explicit caller-supplied implementations.
"""

from xains.metrics.counterfactual_fidelity import (
    cf_coverage,
    change_fidelity,
    invented_features,
)
from xains.metrics.coverage import coverage, hallucination_count
from xains.metrics.fidelity import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)
from xains.metrics.grader import EXTRACTION_GRADE_DIRECTIONS, ExtractionGrades, grade_extraction
from xains.metrics.narrativity import (
    NARRATIVITY_GRADE_DIRECTIONS,
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
from xains.metrics.render import render_grades

__all__ = [
    "EXTRACTION_GRADE_DIRECTIONS",
    "NARRATIVITY_GRADE_DIRECTIONS",
    "DisabledProvider",
    "ExtractionGrades",
    "HuggingFacePerplexityProvider",
    "NarrativityGrades",
    "OpenAICompatibleEchoProvider",
    "PerplexityProvider",
    "ccpr",
    "cecpr",
    "cf_coverage",
    "change_fidelity",
    "coverage",
    "csr",
    "dcpr",
    "fdr",
    "grade_extraction",
    "grade_narrativity",
    "hallucination_count",
    "invented_features",
    "rank_correlation",
    "readability",
    "render_grades",
    "sign_faithfulness",
    "ttcpr",
    "value_faithfulness",
    "vcpr",
]
