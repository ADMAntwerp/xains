"""Lexicon loaders and phrase-occurrence counter.

Both lexicon JSON files are vendored under ``data/`` and loaded once per
call. Greedy longest-first, case-insensitive, word-bounded matching matches
the spec in the PR 6 plan and the locked decisions.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str) -> frozenset[str]:
    """Load a lexicon JSON file and return the lowercased phrase set."""
    with (_DATA_DIR / filename).open(encoding="utf-8") as f:
        payload = json.load(f)
    return frozenset(p.lower().strip() for p in payload["phrases"])


def load_connectives() -> frozenset[str]:
    """Return the 142-entry connectives lexicon (Das et al. 2018)."""
    entries = _load("connectives.json")
    assert len(entries) == 142, (
        f"connectives lexicon corrupt: expected 142 entries, got {len(entries)}"
    )
    return entries


def load_cause_effect_markers() -> frozenset[str]:
    """Return the 19-entry cause-effect marker lexicon (Cedro & Martens 2026)."""
    entries = _load("cause_effect_markers.json")
    assert len(entries) == 19, (
        f"cause-effect lexicon corrupt: expected 19 entries, got {len(entries)}"
    )
    return entries


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def count_phrase_occurrences(text: str, phrases: frozenset[str]) -> int:
    """Count non-overlapping, case-insensitive, word-bounded occurrences of any
    phrase in ``phrases`` within ``text``.

    Greedy longest-first: at each position, the longest matching phrase wins
    and the cursor advances past the match, so prefixes inside a longer match
    are not double-counted.
    """
    if not text or not phrases:
        return 0
    text_lower = text.lower()
    n = len(text_lower)
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    count = 0
    pos = 0
    while pos < n:
        matched = False
        for phrase in sorted_phrases:
            end = pos + len(phrase)
            if end > n or text_lower[pos:end] != phrase:
                continue
            before_ok = pos == 0 or not _is_word_char(text_lower[pos - 1])
            after_ok = end == n or not _is_word_char(text_lower[end])
            if before_ok and after_ok:
                count += 1
                pos = end
                matched = True
                break
        if not matched:
            pos += 1
    return count
