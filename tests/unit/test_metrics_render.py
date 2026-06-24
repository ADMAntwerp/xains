"""Unit tests for render_grades.

Arrows mark scored metrics only. Auxiliary primitives on NarrativityGrades
render without arrows. See ADR 0024.
"""

from xains import ExtractionGrades, NarrativityGrades, render_grades

# ------------------------------------------------------ helpers


def _ext_full() -> ExtractionGrades:
    return ExtractionGrades(
        sign_faithfulness=1.0,
        value_faithfulness=1.0,
        rank_correlation=1.0,
        coverage=1.0,
        hallucination_count=0,
        prompt_version="2",
    )


def _narr_full() -> NarrativityGrades:
    return NarrativityGrades(
        csr=0.27,
        dcpr=1.30,
        ccpr=203.14,
        cecpr=29252.68,
        fdr=0.29,
        ttcpr=2.29,
        vcpr=50.79,
        ppl_ordered=26.6,
        ppl_shuffled=33.8,
        decay_constant=1.25,
        dist2=0.98,
        ttr=0.74,
        vr=0.16,
        cr=0.08,
        cer=0.007,
        n_sentences=9,
    )


# ------------------------------------------------------ tests


def test_render_extraction_arrows_match_each_scored_field() -> None:
    out = render_grades(extraction=_ext_full())
    assert "sign_faithfulness ↑: 1.00" in out
    assert "value_faithfulness ↑: 1.00" in out
    assert "rank_correlation ↑: 1.00" in out
    assert "coverage ↑: 1.00" in out
    assert "hallucination_count ↓: 0" in out


def test_render_narrativity_arrows_match_each_scored_field() -> None:
    out = render_grades(narrativity=_narr_full())
    assert "csr ↑: 0.27" in out
    assert "fdr ↑: 0.29" in out
    assert "dcpr ↓: 1.30" in out
    assert "ccpr ↓: 203.14" in out
    assert "cecpr ↓: 29252.68" in out
    assert "ttcpr ↓: 2.29" in out
    assert "vcpr ↓: 50.79" in out


def test_render_narrativity_auxiliaries_have_no_arrow() -> None:
    out = render_grades(narrativity=_narr_full())
    for aux in (
        "ppl_ordered",
        "ppl_shuffled",
        "decay_constant",
        "dist2",
        "ttr",
        "vr",
        "cr",
        "cer",
        "n_sentences",
    ):
        assert aux in out, f"{aux} missing from render output"
        assert f"{aux} ↑:" not in out, f"{aux} should not carry an arrow"
        assert f"{aux} ↓:" not in out, f"{aux} should not carry an arrow"


def test_render_handles_none_values() -> None:
    extraction = ExtractionGrades(
        sign_faithfulness=None,
        value_faithfulness=None,
        rank_correlation=None,
        coverage=0.0,
        hallucination_count=0,
        prompt_version="2",
    )
    out = render_grades(extraction=extraction)
    assert "sign_faithfulness ↑: None" in out


def test_render_groups_under_two_headers() -> None:
    out = render_grades(extraction=_ext_full(), narrativity=_narr_full())
    assert "Verbalization fidelity" in out
    assert "Narrativity" in out
    assert out.index("Verbalization fidelity") < out.index("Narrativity")


def test_render_omits_prompt_version() -> None:
    out = render_grades(extraction=_ext_full())
    assert "prompt_version" not in out


def test_render_returns_empty_string_when_neither_given() -> None:
    assert render_grades() == ""


def test_render_extraction_only_does_not_emit_narrativity_header() -> None:
    out = render_grades(extraction=_ext_full())
    assert "Narrativity" not in out


def test_render_narrativity_only_does_not_emit_fidelity_header() -> None:
    out = render_grades(narrativity=_narr_full())
    assert "Verbalization fidelity" not in out


# ------------------------------------------------------ scored_only (ADR 0026)


def test_scored_only_hides_narrativity_auxiliaries() -> None:
    out = render_grades(narrativity=_narr_full(), scored_only=True)
    for aux in (
        "ppl_ordered",
        "ppl_shuffled",
        "decay_constant",
        "dist2",
        "ttr",
        "vr",
        "cr",
        "cer",
        "n_sentences",
    ):
        assert aux not in out, f"{aux} should be hidden when scored_only=True"


def test_scored_only_keeps_scored_narrativity_metrics_with_arrows() -> None:
    out = render_grades(narrativity=_narr_full(), scored_only=True)
    assert "csr ↑: 0.27" in out
    assert "fdr ↑: 0.29" in out
    assert "dcpr ↓: 1.30" in out
    assert "ccpr ↓: 203.14" in out
    assert "cecpr ↓: 29252.68" in out
    assert "ttcpr ↓: 2.29" in out
    assert "vcpr ↓: 50.79" in out


def test_scored_only_default_false_still_renders_auxiliaries() -> None:
    out = render_grades(narrativity=_narr_full())
    assert "ppl_ordered" in out
    assert "n_sentences" in out


def test_scored_only_on_extraction_is_a_noop_visually() -> None:
    plain = render_grades(extraction=_ext_full())
    filtered = render_grades(extraction=_ext_full(), scored_only=True)
    assert plain == filtered


# ------------------------------------------------------ value formatting (2dp)


def test_int_metric_renders_as_int_not_as_float() -> None:
    """hallucination_count and n_sentences are ints; rendering must not add decimals."""
    ext_out = render_grades(extraction=_ext_full())
    assert "hallucination_count ↓: 0" in ext_out
    assert "hallucination_count ↓: 0.00" not in ext_out

    narr_out = render_grades(narrativity=_narr_full())
    assert "n_sentences: 9" in narr_out
    assert "n_sentences: 9.00" not in narr_out


def test_float_metric_renders_to_exactly_two_decimals() -> None:
    """Floats round to 2 decimals: 0.007 -> 0.01, 26.6 -> 26.60, 1.30 -> 1.30."""
    out = render_grades(narrativity=_narr_full())
    assert "cer: 0.01" in out
    assert "ppl_ordered: 26.60" in out
    assert "dcpr ↓: 1.30" in out
