"""Unit tests for the exponential decay-constant fitter."""

import sys

import pytest

from xain.metrics._internal.curve_fit import fit_decay_constant


def test_fit_decay_constant_monotonic_decreasing_returns_positive_b() -> None:
    """y(x) = 10 * exp(-0.5*x) + 5 evaluated at x = 1..5."""
    perplexities = [11.065, 8.679, 7.231, 6.353, 5.821]
    b = fit_decay_constant(perplexities)
    assert b is not None
    assert b > 0


def test_fit_decay_constant_exactly_three_points_works() -> None:
    """Three points is the documented minimum."""
    perplexities = [10.0, 7.0, 5.0]
    b = fit_decay_constant(perplexities)
    assert b is not None
    assert b > 0


def test_fit_decay_constant_constant_series_returns_none() -> None:
    """Constant series has no decay → b ≤ 0 effectively → None."""
    perplexities = [5.0, 5.0, 5.0, 5.0, 5.0]
    assert fit_decay_constant(perplexities) is None


def test_fit_decay_constant_ascending_returns_none() -> None:
    """Ascending series fits b < 0 (growth, not decay) → None."""
    perplexities = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert fit_decay_constant(perplexities) is None


def test_fit_decay_constant_fewer_than_three_returns_none() -> None:
    assert fit_decay_constant([]) is None
    assert fit_decay_constant([10.0]) is None
    assert fit_decay_constant([10.0, 5.0]) is None


def test_fit_decay_constant_raises_when_scipy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``import scipy`` to fail yields a clear ImportError pointing at the extra."""
    # Set both the parent and the submodule to None — the lazy import is
    # ``from scipy.optimize import curve_fit`` which touches both.
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    with pytest.raises(ImportError, match=r'pip install "xain\[narrativity\]"'):
        fit_decay_constant([11.0, 8.0, 7.0, 6.0, 5.5])
