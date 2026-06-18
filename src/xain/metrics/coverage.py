"""Coverage and hallucination-count metrics."""

from xain.guardrails.types import NarrativeExtraction
from xain.schema import DatasetSchema


def coverage(extraction: NarrativeExtraction, schema: DatasetSchema, k: int) -> float:
    """Fraction of (top-k of) schema features that the narrative resolved.

    Denominator is ``min(k, n_schema_features)`` so a narrative can hit 1.0
    by discussing the top-k features rather than all of them. Numerator is
    the count of distinct schema features present in ``extraction.features``.

    Always defined: returns 0.0 when the schema has no features, when no
    features were resolved, or both. Raises ``ValueError`` when ``k <= 0``
    (caller bug, not degenerate input).
    """
    if k <= 0:
        raise ValueError(f"coverage: k must be >= 1; got {k}.")
    n_schema = len(schema.features) if schema.features else 0
    denominator = min(k, n_schema)
    if denominator == 0:
        return 0.0
    schema_names = {f.name for f in schema.features or []}
    resolved = sum(1 for name in extraction.features if name in schema_names)
    return resolved / denominator


def hallucination_count(extraction: NarrativeExtraction) -> int:
    """Number of unresolved feature mentions in the extraction."""
    return len(extraction.hallucinations)
