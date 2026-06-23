"""Fidelity metrics: sign / value / rank.

All three are pure: ``(extraction, request) -> float | None``. Returning
``None`` means "metric undefined for this input" (no comparable features,
constant ground-truth ranks, etc.) — never an exception.
"""

import math
from collections.abc import Sequence

from xains.guardrails.types import NarrativeExtraction
from xains.types import TabularContribution, TabularExplanationRequest


def _sign_of(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _is_numeric(x: object) -> bool:
    """Numeric iff int or float, excluding bool (which subclasses int)."""
    return isinstance(x, int | float) and not isinstance(x, bool)


def _contribution_index(
    request: TabularExplanationRequest,
) -> dict[str, TabularContribution]:
    return {c.name: c for c in request.contributions}


def sign_faithfulness(
    extraction: NarrativeExtraction, request: TabularExplanationRequest
) -> float | None:
    """Fraction of resolved-feature claims whose narrative sign matches the
    sign of the contribution's importance.

    Hallucinations are ignored. ``None`` when no resolved features are also
    present in the request's contributions.
    """
    contribs = _contribution_index(request)
    correct = 0
    total = 0
    for name, claim in extraction.features.items():
        c = contribs.get(name)
        if c is None:
            continue
        total += 1
        if claim.sign == _sign_of(c.importance):
            correct += 1
    if total == 0:
        return None
    return correct / total


def value_faithfulness(
    extraction: NarrativeExtraction,
    request: TabularExplanationRequest,
    atol: float = 1e-6,
) -> float | None:
    """Fraction of resolved-feature claims whose narrative value matches the
    contribution's value within ``atol``.

    Scores only numeric value comparisons. A high score over few comparable
    pairs is not the same as a high score over the full claim set; callers
    should also look at how many features actually contributed.

    ``bool`` is excluded from "numeric" (it subclasses ``int`` in Python but
    a True/False answer is not a feature-value comparison we want to score).
    String, None, dict, or list values on either side are skipped.

    ``None`` when no comparable numeric pairs exist.
    """
    contribs = _contribution_index(request)
    correct = 0
    total = 0
    for name, claim in extraction.features.items():
        c = contribs.get(name)
        if c is None:
            continue
        if not _is_numeric(claim.value) or not _is_numeric(c.value):
            continue
        total += 1
        if math.isclose(float(claim.value), float(c.value), abs_tol=atol, rel_tol=0.0):
            correct += 1
    if total == 0:
        return None
    return correct / total


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Assign 1-indexed ranks; ties get the average of the positions they occupy.

    Higher value -> smaller rank (rank 1 is the largest).
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: -values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average 1-indexed position
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0.0 or sy2 == 0.0:
        return None
    return num / math.sqrt(sx2 * sy2)


def rank_correlation(
    extraction: NarrativeExtraction, request: TabularExplanationRequest
) -> float | None:
    """Spearman rank correlation between narrative rank and ground-truth rank
    over resolved features common to extraction and request.

    Ground-truth rank ranks contributions by descending ``abs(importance)``
    with average-rank ties, computed over the full contribution list.

    ``None`` if fewer than two common features, or if either rank series has
    zero variance (e.g. all contributions have equal magnitude).
    """
    contribs = _contribution_index(request)
    common = [(name, claim) for name, claim in extraction.features.items() if name in contribs]
    if len(common) < 2:
        return None

    all_names = [c.name for c in request.contributions]
    all_abs = [abs(c.importance) for c in request.contributions]
    all_gt_ranks = _average_ranks(all_abs)
    gt_rank_by_name = dict(zip(all_names, all_gt_ranks, strict=True))

    narrative_ranks = [float(claim.rank) for _, claim in common]
    gt_ranks = [gt_rank_by_name[name] for name, _ in common]

    return _pearson(narrative_ranks, gt_ranks)
