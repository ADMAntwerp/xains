"""Unit tests for internal lexicon helpers."""

import pytest  # noqa: F401  (imported for pytest discovery only)

from xain.metrics._internal.lexicons import (
    count_phrase_occurrences,
    load_cause_effect_markers,
    load_connectives,
)

# ------------------------------------------------------ load functions


def test_load_connectives_returns_142_entries() -> None:
    entries = load_connectives()
    assert len(entries) == 142


def test_load_connectives_returns_frozenset() -> None:
    assert isinstance(load_connectives(), frozenset)


def test_load_connectives_all_lowercase() -> None:
    entries = load_connectives()
    assert all(p == p.lower() for p in entries)


def test_load_cause_effect_markers_returns_19_entries() -> None:
    entries = load_cause_effect_markers()
    assert len(entries) == 19


def test_load_cause_effect_markers_returns_frozenset() -> None:
    assert isinstance(load_cause_effect_markers(), frozenset)


def test_load_cause_effect_markers_contains_paper_terms() -> None:
    """Spot-check a few representative entries from the paper's Appendix A list."""
    entries = load_cause_effect_markers()
    for term in ("because", "since", "consequently", "therefore", "thus"):
        assert term in entries


# ------------------------------------------------------ count_phrase_occurrences


def test_count_phrase_occurrences_empty_text_returns_zero() -> None:
    assert count_phrase_occurrences("", frozenset({"and", "but"})) == 0


def test_count_phrase_occurrences_empty_phrases_returns_zero() -> None:
    assert count_phrase_occurrences("Some text with words.", frozenset()) == 0


def test_count_phrase_occurrences_case_insensitive() -> None:
    text = "HOWEVER, the result was however unclear."
    assert count_phrase_occurrences(text, frozenset({"however"})) == 2


def test_count_phrase_occurrences_once_per_occurrence() -> None:
    text = "However, however, however."
    assert count_phrase_occurrences(text, frozenset({"however"})) == 3


def test_count_phrase_occurrences_greedy_longest_first() -> None:
    """A longer phrase consumes the position; shorter prefixes do not double-count."""
    text = "as a result of the change, as a result of the policy"
    phrases = frozenset({"as", "as a result", "as a result of"})
    # Only the longest matches at each position; "as" / "as a result" inside
    # "as a result of" are not counted.
    assert count_phrase_occurrences(text, phrases) == 2


def test_count_phrase_occurrences_word_boundary_rejects_partial() -> None:
    """'since' must not match inside 'sincerely'."""
    assert count_phrase_occurrences("sincerely yours", frozenset({"since"})) == 0


def test_count_phrase_occurrences_word_boundary_with_punctuation() -> None:
    """Punctuation forms a word boundary; matches abut commas, periods, etc."""
    text = "So, since 1990, things changed."
    assert count_phrase_occurrences(text, frozenset({"so", "since"})) == 2
