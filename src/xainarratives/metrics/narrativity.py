"""Narrativity metrics.

Currently: ``readability`` (Flesch reading-ease) over the original narrative
text. ``textstat`` is an optional dependency, imported lazily on first use.
"""

import math

from xainarratives.guardrails.types import NarrativeExtraction

_MISSING_TEXTSTAT_MESSAGE = (
    "The 'textstat' package is required for readability(). "
    'Install with: pip install "xainarratives[textstat]"'
)


def readability(
    extraction: NarrativeExtraction,
    narrative_text: str,
) -> float | None:
    """Flesch reading-ease score over the original narrative text.

    The ``extraction`` argument is unused in this implementation; it is kept
    in the signature for symmetry with the other metrics and to leave room
    for future per-claim readability work.

    Returns ``None`` for empty or whitespace-only input, or when
    ``textstat.flesch_reading_ease`` returns NaN (e.g. extremely short
    input where syllable / sentence counts collapse).

    Raises ``ImportError`` when ``textstat`` is not installed; the message
    points the caller at the right pip extra.
    """
    if not narrative_text.strip():
        return None
    try:
        import textstat
    except ImportError as exc:
        raise ImportError(_MISSING_TEXTSTAT_MESSAGE) from exc
    score = textstat.flesch_reading_ease(narrative_text)
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return float(score)
