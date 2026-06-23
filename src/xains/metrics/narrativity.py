"""Narrativity metrics.

Primary entry: ``grade_narrativity(text, provider) -> NarrativityGrades``.
The seven paper metrics (CSR, DCPR, CCPR, CECPR, FDR, TTCPR, VCPR) are also
exposed as standalone functions.

Paper: Cedro & Martens 2026, "On the Importance and Evaluation of
Narrativity in Natural Language AI Explanations" (arXiv:2604.18311).
``readability`` (Flesch reading-ease) ships from PR 5 and is unchanged.

NLTK + scipy are optional dependencies (``pip install
"xains[narrativity]"``). Standalone metric functions raise
``ImportError`` when the required packages are missing; ``grade_narrativity``
catches ``ImportError`` and degrades the affected fields to ``None``.
"""

import math
import random
from typing import Any

from pydantic import BaseModel, ConfigDict

from xains.guardrails.types import NarrativeExtraction
from xains.metrics._internal.curve_fit import fit_decay_constant
from xains.metrics._internal.lexicons import (
    count_phrase_occurrences,
    load_cause_effect_markers,
    load_connectives,
)
from xains.metrics._internal.perplexity_utils import cumulative_perplexity
from xains.metrics._internal.tokenize import (
    bigrams,
    count_verbs,
    pos_tags,
    split_sentences,
    word_tokens,
)
from xains.metrics.perplexity import PerplexityProvider

_SHUFFLE_SEED = 42

_MISSING_TEXTSTAT_MESSAGE = (
    "The 'textstat' package is required for readability(). "
    'Install with: pip install "xains[textstat]"'
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


# ====================================================================
# Internal helpers (not part of the public surface).
# ====================================================================


def _shuffled_text(sentences: list[str]) -> str:
    """Deterministic shuffle (seed 42) of a sentence list, joined by spaces."""
    rng = random.Random(_SHUFFLE_SEED)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def _dist2(text: str) -> float | None:
    """Distinct-bigram ratio: |unique bigrams| / |total bigrams|."""
    bgs = bigrams(word_tokens(text))
    if not bgs:
        return None
    return len(set(bgs)) / len(bgs)


def _ttr(text: str) -> float | None:
    """Type-token ratio: |unique words| / |total words|."""
    tokens = word_tokens(text)
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def _vr(text: str) -> float | None:
    """Verb ratio: |verbs| / |total words|. Requires NLTK (POS tagging)."""
    tokens = word_tokens(text)
    if not tokens:
        return None
    verbs = count_verbs(pos_tags(tokens))
    return verbs / len(tokens)


def _cr(text: str) -> float | None:
    """Connective ratio: |connective hits| / |total words|."""
    tokens = word_tokens(text)
    if not tokens:
        return None
    return count_phrase_occurrences(text, load_connectives()) / len(tokens)


def _cer(text: str) -> float | None:
    """Cause-effect-marker ratio: |cer hits| / |total words|."""
    tokens = word_tokens(text)
    if not tokens:
        return None
    return count_phrase_occurrences(text, load_cause_effect_markers()) / len(tokens)


def _decay_constant_from_text(text: str, provider: PerplexityProvider) -> float | None:
    """Compose ``cumulative_perplexity`` + ``fit_decay_constant``.

    Returns the fitted ``b`` (decay constant), or ``None`` if anything along
    the chain degrades.
    """
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return None
    cum_ppl = cumulative_perplexity(sentences, provider)
    if cum_ppl is None:
        return None
    return fit_decay_constant(cum_ppl)


# ====================================================================
# Seven paper metrics (Cedro & Martens 2026).
# ====================================================================


def csr(text: str, provider: PerplexityProvider) -> float | None:
    """Continuous Structure Rate (Eq. 1, ↑).

    ``CSR = (PPL_shuffled - PPL_ordered) / PPL_ordered``.

    Returns ``None`` if the text has fewer than 2 sentences, if either PPL
    call returns ``None``, or if ``PPL_ordered`` is 0.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    ppl_o = provider.compute(text)
    if ppl_o is None or ppl_o == 0:
        return None
    ppl_s = provider.compute(_shuffled_text(sentences))
    if ppl_s is None:
        return None
    return (ppl_s - ppl_o) / ppl_o


def dcpr(text: str, provider: PerplexityProvider) -> float | None:
    """Diversity-adjusted Context Progression Rate (Eq. 3, ↓).

    ``DCPR = r / Dist2^2``.
    """
    r = _decay_constant_from_text(text, provider)
    if r is None:
        return None
    d2 = _dist2(text)
    if d2 is None or d2 == 0:
        return None
    return r / (d2**2)


def ccpr(text: str, provider: PerplexityProvider) -> float | None:
    """Connectives-adjusted Context Progression Rate (Eq. 4, ↓).

    ``CCPR = r / CR^2``.
    """
    r = _decay_constant_from_text(text, provider)
    if r is None:
        return None
    cr_val = _cr(text)
    if cr_val is None or cr_val == 0:
        return None
    return r / (cr_val**2)


def cecpr(text: str, provider: PerplexityProvider) -> float | None:
    """Cause-Effect-adjusted Context Progression Rate (Eq. 5, ↓).

    ``CECPR = r / CER^2``. Returns ``None`` when no cause-effect markers
    are present (per the locked PR 6 decision).
    """
    r = _decay_constant_from_text(text, provider)
    if r is None:
        return None
    cer_val = _cer(text)
    if cer_val is None or cer_val == 0:
        return None
    return r / (cer_val**2)


def fdr(text: str, provider: PerplexityProvider) -> float | None:
    """Fluency-Diversity Rate (Eq. 6, ↑).

    ``FDR = Dist2^2 / ln(PPL)``. Sentence-independent.

    Returns ``None`` if PPL <= 1 (ln <= 0), if the provider returns
    ``None``, or if there are no bigrams.
    """
    ppl = provider.compute(text)
    if ppl is None or ppl <= 1:
        return None
    d2 = _dist2(text)
    if d2 is None:
        return None
    return (d2**2) / math.log(ppl)


def ttcpr(text: str, provider: PerplexityProvider) -> float | None:
    """Type-Token-adjusted Context Progression Rate (Eq. 7, ↓).

    ``TTCPR = r / TTR^2``.
    """
    r = _decay_constant_from_text(text, provider)
    if r is None:
        return None
    ttr_val = _ttr(text)
    if ttr_val is None or ttr_val == 0:
        return None
    return r / (ttr_val**2)


def vcpr(text: str, provider: PerplexityProvider) -> float | None:
    """Verb-adjusted Context Progression Rate (Eq. 8, ↓).

    ``VCPR = r / VR^2``.
    """
    r = _decay_constant_from_text(text, provider)
    if r is None:
        return None
    vr_val = _vr(text)
    if vr_val is None or vr_val == 0:
        return None
    return r / (vr_val**2)


# ====================================================================
# Aggregate scoring.
# ====================================================================


class NarrativityGrades(BaseModel):
    """Aggregate of the seven paper narrativity metrics + auxiliary primitives."""

    model_config = ConfigDict(extra="forbid")

    # 7 derived metrics.
    csr: float | None = None
    dcpr: float | None = None
    ccpr: float | None = None
    cecpr: float | None = None
    fdr: float | None = None
    ttcpr: float | None = None
    vcpr: float | None = None

    # 9 auxiliary raw values (cheap to capture, expensive to recompute).
    ppl_ordered: float | None = None
    ppl_shuffled: float | None = None
    decay_constant: float | None = None
    dist2: float | None = None
    ttr: float | None = None
    vr: float | None = None
    cr: float | None = None
    cer: float | None = None
    n_sentences: int | None = None


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``; return ``None`` on ``ImportError`` (missing optional dep)."""
    try:
        return fn(*args, **kwargs)
    except ImportError:
        return None


def grade_narrativity(
    narrative_text: str, perplexity_provider: PerplexityProvider
) -> NarrativityGrades:
    """Compute all seven paper narrativity metrics + auxiliary primitives.

    Each metric and primitive degrades to ``None`` independently. Missing
    optional dependencies (NLTK, scipy) cause the affected fields to be
    ``None`` rather than raising.
    """
    # ---------------- text-only primitives (no NLTK except vr) ----------
    ttr_val = _safe_call(_ttr, narrative_text)
    dist2_val = _safe_call(_dist2, narrative_text)
    cr_val = _safe_call(_cr, narrative_text)
    cer_val = _safe_call(_cer, narrative_text)
    vr_val = _safe_call(_vr, narrative_text)

    # ---------------- sentence-dependent primitives ---------------------
    sentences = _safe_call(split_sentences, narrative_text)
    n_sentences = len(sentences) if sentences is not None else None

    # ---------------- perplexity-dependent primitives -------------------
    cum_ppl: list[float] | None = None
    if sentences is not None and len(sentences) >= 1:
        cum_ppl = _safe_call(cumulative_perplexity, sentences, perplexity_provider)
    ppl_ordered = cum_ppl[-1] if cum_ppl else None

    decay_constant: float | None = (
        _safe_call(fit_decay_constant, cum_ppl) if cum_ppl is not None else None
    )

    ppl_shuffled: float | None = None
    if sentences is not None and len(sentences) >= 2:
        try:
            ppl_shuffled = perplexity_provider.compute(_shuffled_text(sentences))
        except ImportError:
            ppl_shuffled = None

    # ---------------- derived metrics -----------------------------------
    csr_val: float | None = None
    if ppl_ordered is not None and ppl_ordered != 0 and ppl_shuffled is not None:
        csr_val = (ppl_shuffled - ppl_ordered) / ppl_ordered

    def _r_over_x_squared(x: float | None) -> float | None:
        if decay_constant is None or x is None or x == 0:
            return None
        return decay_constant / (x**2)

    dcpr_val = _r_over_x_squared(dist2_val)
    ccpr_val = _r_over_x_squared(cr_val)
    cecpr_val = _r_over_x_squared(cer_val)
    ttcpr_val = _r_over_x_squared(ttr_val)
    vcpr_val = _r_over_x_squared(vr_val)

    fdr_val: float | None = None
    if ppl_ordered is not None and ppl_ordered > 1 and dist2_val is not None:
        fdr_val = (dist2_val**2) / math.log(ppl_ordered)

    return NarrativityGrades(
        csr=csr_val,
        dcpr=dcpr_val,
        ccpr=ccpr_val,
        cecpr=cecpr_val,
        fdr=fdr_val,
        ttcpr=ttcpr_val,
        vcpr=vcpr_val,
        ppl_ordered=ppl_ordered,
        ppl_shuffled=ppl_shuffled,
        decay_constant=decay_constant,
        dist2=dist2_val,
        ttr=ttr_val,
        vr=vr_val,
        cr=cr_val,
        cer=cer_val,
        n_sentences=n_sentences,
    )
