"""Unit tests for the internal tokenize utilities."""

import sys

import pytest

from xainarratives.metrics._internal.tokenize import (
    bigrams,
    count_verbs,
    pos_tags,
    split_sentences,
    word_tokens,
)

# ------------------------------------------------------ split_sentences (NLTK)


def test_split_sentences_basic() -> None:
    text = "First sentence. Second one. And a third!"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_empty_returns_empty_list() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


# ------------------------------------------------------ word_tokens (pure)


def test_word_tokens_lowercase_alphabetic_only() -> None:
    text = "The DTI was 0.41, which is HIGH!"
    tokens = word_tokens(text)
    # Strips digits, punctuation; lowercases.
    assert tokens == ["the", "dti", "was", "which", "is", "high"]


def test_word_tokens_empty_text_returns_empty_list() -> None:
    assert word_tokens("") == []
    assert word_tokens("123 4.56 ,.") == []


# ------------------------------------------------------ bigrams (pure)


def test_bigrams_pairwise() -> None:
    assert bigrams(["a", "b", "c", "d"]) == [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
    ]


def test_bigrams_singleton_returns_empty() -> None:
    assert bigrams(["a"]) == []
    assert bigrams([]) == []


# ------------------------------------------------------ pos_tags + count_verbs


def test_pos_tags_returns_penn_treebank_style() -> None:
    tokens = ["the", "applicant", "defaulted"]
    tagged = pos_tags(tokens)
    assert len(tagged) == 3
    for token, tag in tagged:
        assert isinstance(token, str)
        assert isinstance(tag, str)
        assert len(tag) > 0
    # "defaulted" must be tagged as a verb (VB*) by NLTK.
    assert any(tag.startswith("VB") for _, tag in tagged)


def test_count_verbs_recognizes_VB_prefix() -> None:
    tagged = [
        ("the", "DT"),
        ("dog", "NN"),
        ("ran", "VBD"),
        ("and", "CC"),
        ("jumped", "VBD"),
        ("quickly", "RB"),
    ]
    assert count_verbs(tagged) == 2


def test_count_verbs_zero_when_no_verbs() -> None:
    assert count_verbs([("the", "DT"), ("dog", "NN")]) == 0


def test_count_verbs_handles_all_VB_subtags() -> None:
    """All Penn Treebank VB* subtags (VB, VBD, VBG, VBN, VBP, VBZ) count as verbs."""
    tagged = [
        ("be", "VB"),
        ("was", "VBD"),
        ("being", "VBG"),
        ("been", "VBN"),
        ("am", "VBP"),
        ("is", "VBZ"),
    ]
    assert count_verbs(tagged) == 6


# ------------------------------------------------------ NLTK missing


def test_split_sentences_raises_when_nltk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import nltk`` to fail yields a clear ImportError pointing at the extra."""
    monkeypatch.setitem(sys.modules, "nltk", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[narrativity\]"'):
        split_sentences("Some text. Some more.")


def test_pos_tags_raises_when_nltk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "nltk", None)
    with pytest.raises(ImportError, match=r'pip install "xainarratives\[narrativity\]"'):
        pos_tags(["the", "dog"])
