"""Unit tests for fidelity metrics: sign / value / rank.

All metrics are pure: extraction + request → score | None. No I/O.
"""

import math
from typing import Any

from xain import (
    FeatureClaim,
    NarrativeExtraction,
    Prediction,
    TabularContribution,
    TabularExplanationRequest,
)
from xain.metrics import (
    rank_correlation,
    sign_faithfulness,
    value_faithfulness,
)

# ------------------------------------------------------ helpers


def _request(*contribs: tuple[str, Any, float]) -> TabularExplanationRequest:
    """Build a request from (name, value, importance) tuples."""
    return TabularExplanationRequest(
        features={name: val for name, val, _ in contribs},
        prediction=Prediction(predicted_class=1),
        contributions=[
            TabularContribution(name=name, value=val, importance=imp) for name, val, imp in contribs
        ],
    )


def _claim(name: str, *, rank: int, sign: int, value: Any = None) -> FeatureClaim:
    return FeatureClaim(
        rank=rank,
        sign=sign,
        value=value,
        narrative_name=name,
        resolved_to=name,
    )


def _hallucination(name: str, *, rank: int, sign: int) -> FeatureClaim:
    return FeatureClaim(
        rank=rank,
        sign=sign,
        narrative_name=name,
        resolved_to=None,
    )


def _extraction(
    features: dict[str, FeatureClaim],
    hallucinations: list[FeatureClaim] | None = None,
) -> NarrativeExtraction:
    return NarrativeExtraction(
        features=features,
        hallucinations=hallucinations or [],
        prompt_version="2",
        model_name="test",
    )


# ------------------------------------------------------ sign_faithfulness


def test_sign_faithfulness_all_correct() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1),
            "age": _claim("age", rank=2, sign=-1),
        }
    )
    assert sign_faithfulness(ext, req) == 1.0


def test_sign_faithfulness_all_wrong() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=-1),
            "age": _claim("age", rank=2, sign=1),
        }
    )
    assert sign_faithfulness(ext, req) == 0.0


def test_sign_faithfulness_partial() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1),  # correct
            "age": _claim("age", rank=2, sign=1),  # wrong
        }
    )
    assert sign_faithfulness(ext, req) == 0.5


def test_sign_faithfulness_zero_importance_matches_zero_sign() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, 0.0))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1),
            "age": _claim("age", rank=2, sign=0),
        }
    )
    assert sign_faithfulness(ext, req) == 1.0


def test_sign_faithfulness_no_resolved_features_returns_none() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(features={})
    assert sign_faithfulness(ext, req) is None


def test_sign_faithfulness_ignores_hallucinations() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(
        features={"dti": _claim("dti", rank=1, sign=1)},
        hallucinations=[_hallucination("ghost", rank=2, sign=-1)],
    )
    # Only dti is comparable; hallucination is ignored. 1/1 = 1.0.
    assert sign_faithfulness(ext, req) == 1.0


# ------------------------------------------------------ value_faithfulness


def test_value_faithfulness_exact_match() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1, value=0.41)})
    assert value_faithfulness(ext, req) == 1.0


def test_value_faithfulness_within_tolerance() -> None:
    req = _request(("dti", 0.41, 0.37))
    # Difference 5e-7 < default abs_tol 1e-6.
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1, value=0.4100005)})
    assert value_faithfulness(ext, req) == 1.0


def test_value_faithfulness_outside_tolerance() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1, value=0.42)})
    assert value_faithfulness(ext, req) == 0.0


def test_value_faithfulness_string_value_skipped() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1, value="high")})
    # Non-numeric on extraction side → skipped. No comparable pairs → None.
    assert value_faithfulness(ext, req) is None


def test_value_faithfulness_no_numeric_pairs_returns_none() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1, value=None),
            "age": _claim("age", rank=2, sign=-1, value=None),
        }
    )
    assert value_faithfulness(ext, req) is None


def test_value_faithfulness_custom_atol() -> None:
    req = _request(("dti", 0.41, 0.37))
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1, value=0.45)})
    # 0.04 difference: outside default 1e-6, inside 0.05.
    assert value_faithfulness(ext, req) == 0.0
    assert value_faithfulness(ext, req, atol=0.05) == 1.0


# ------------------------------------------------------ rank_correlation


def test_rank_correlation_perfect_agreement_returns_one() -> None:
    # gt ranks (sorted by |importance|): dti=1, age=2.
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=1, sign=1),
            "age": _claim("age", rank=2, sign=-1),
        }
    )
    rho = rank_correlation(ext, req)
    assert rho is not None
    assert math.isclose(rho, 1.0, abs_tol=1e-9)


def test_rank_correlation_reverse_agreement_returns_minus_one() -> None:
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(
        features={
            "dti": _claim("dti", rank=2, sign=1),
            "age": _claim("age", rank=1, sign=-1),
        }
    )
    rho = rank_correlation(ext, req)
    assert rho is not None
    assert math.isclose(rho, -1.0, abs_tol=1e-9)


def test_rank_correlation_independent_returns_around_zero() -> None:
    # 4 features. gt ranks: a=1, b=2, c=3, d=4.
    # narrative ranks: a=3, b=1, c=4, d=2 → Spearman rho = 0 exactly.
    req = _request(
        ("a", 1, 0.40),
        ("b", 2, 0.30),
        ("c", 3, 0.20),
        ("d", 4, 0.10),
    )
    ext = _extraction(
        features={
            "a": _claim("a", rank=3, sign=1),
            "b": _claim("b", rank=1, sign=1),
            "c": _claim("c", rank=4, sign=1),
            "d": _claim("d", rank=2, sign=1),
        }
    )
    rho = rank_correlation(ext, req)
    assert rho is not None
    assert math.isclose(rho, 0.0, abs_tol=1e-9)


def test_rank_correlation_fewer_than_two_common_features_returns_none() -> None:
    # Only one resolved feature → fewer than 2 pairs → None.
    req = _request(("dti", 0.41, 0.37), ("age", 29, -0.12))
    ext = _extraction(features={"dti": _claim("dti", rank=1, sign=1)})
    assert rank_correlation(ext, req) is None


def test_rank_correlation_all_ground_truth_tied_returns_none() -> None:
    # All abs importances equal → average-rank scheme assigns same rank to all.
    # Variance of gt ranks is zero → rho undefined → None.
    req = _request(
        ("a", 1, 0.5),
        ("b", 2, 0.5),
        ("c", 3, 0.5),
    )
    ext = _extraction(
        features={
            "a": _claim("a", rank=1, sign=1),
            "b": _claim("b", rank=2, sign=1),
            "c": _claim("c", rank=3, sign=1),
        }
    )
    assert rank_correlation(ext, req) is None
