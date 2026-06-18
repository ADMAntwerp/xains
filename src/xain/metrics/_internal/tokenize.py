"""Sentence/word/bigram/POS helpers.

NLTK is an optional dependency, imported lazily on first use. The first
call to ``split_sentences`` or ``pos_tags`` may trigger ``nltk.download``
silently for the required corpora (``punkt`` / ``punkt_tab`` for the
sentence splitter, ``averaged_perceptron_tagger`` /
``averaged_perceptron_tagger_eng`` for the POS tagger).
"""

import itertools
import re
from typing import Any

_MISSING_NLTK_MESSAGE = (
    "The 'nltk' package is required for split_sentences / pos_tags. "
    'Install with: pip install "xain[narrativity]"'
)

_WORD_RE = re.compile(r"[a-z]+")


def _ensure_nltk() -> Any:
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(_MISSING_NLTK_MESSAGE) from exc
    return nltk


def split_sentences(text: str) -> list[str]:
    """NLTK sent_tokenize. Returns ``[]`` for empty or whitespace-only input."""
    if not text.strip():
        return []
    nltk = _ensure_nltk()
    try:
        return list(nltk.sent_tokenize(text))
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        return list(nltk.sent_tokenize(text))


def word_tokens(text: str) -> list[str]:
    """Lowercase alphabetic tokens. Pure regex; no NLTK."""
    return _WORD_RE.findall(text.lower())


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    """Consecutive pairs. Empty if fewer than 2 tokens."""
    return list(itertools.pairwise(tokens))


def pos_tags(tokens: list[str]) -> list[tuple[str, str]]:
    """NLTK pos_tag returning Penn Treebank tags."""
    nltk = _ensure_nltk()
    try:
        return list(nltk.pos_tag(tokens))
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        return list(nltk.pos_tag(tokens))


def count_verbs(tagged: list[tuple[str, str]]) -> int:
    """Count entries whose Penn Treebank tag starts with 'VB' (VB, VBD, VBG, VBN, VBP, VBZ)."""
    return sum(1 for _, tag in tagged if tag.startswith("VB"))
