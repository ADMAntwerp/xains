"""Cumulative perplexity helper."""

from xainarratives.metrics.perplexity import PerplexityProvider


def cumulative_perplexity(sentences: list[str], provider: PerplexityProvider) -> list[float] | None:
    """Return ``[PPL(s1), PPL(s1+s2), ..., PPL(s1+...+sN)]``.

    Short-circuits and returns ``None`` if any provider call returns ``None``
    (subsequent calls are not made). Empty input yields an empty list and
    no provider calls.
    """
    result: list[float] = []
    for i in range(1, len(sentences) + 1):
        prefix = " ".join(sentences[:i])
        ppl = provider.compute(prefix)
        if ppl is None:
            return None
        result.append(ppl)
    return result
