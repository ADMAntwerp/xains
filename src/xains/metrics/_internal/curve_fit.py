"""Exponential decay-constant fitter (scipy.optimize.curve_fit).

Lazy scipy import. Returns ``None`` on degenerate inputs, fit failure, or
non-decay (b ≤ 0).
"""

from typing import Any

_MISSING_SCIPY_MESSAGE = (
    "The 'scipy' package is required for fit_decay_constant(). "
    'Install with: pip install "xains[narrativity]"'
)


def fit_decay_constant(perplexities: list[float]) -> float | None:
    """Fit ``y(x) = A * exp(-b * x) + C`` and return ``b`` if it is positive.

    Returns ``None`` when:
    * ``len(perplexities) < 3`` (insufficient points),
    * the input is a constant series (no decay possible),
    * scipy fit raises (no convergence, bad bounds, etc.),
    * fitted ``b <= 0`` (non-decay / growth).
    """
    n = len(perplexities)
    if n < 3:
        return None

    y_min = min(perplexities)
    y_max = max(perplexities)
    if y_min == y_max:
        return None

    # Reject non-decreasing inputs before fitting: if the series doesn't end
    # below where it started, it doesn't represent decay. Cheap pre-check that
    # catches ascending / flat-ish inputs that would otherwise produce
    # bound-saturated degenerate fits.
    if perplexities[-1] >= perplexities[0]:
        return None

    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise ImportError(_MISSING_SCIPY_MESSAGE) from exc

    def _model(x: Any, a: float, b: float, c: float) -> Any:
        return a * np.exp(-b * x) + c

    x_data = np.arange(1, n + 1, dtype=float)
    y_data = np.array(perplexities, dtype=float)
    p0 = (y_max - y_min, 0.5, y_min)
    # Tighter b bound: real decay rates from the paper are O(0.1)-O(2);
    # widening to +/-10 lets the optimizer saturate at the boundary on
    # degenerate inputs.
    bounds = ([0.0, -5.0, -np.inf], [np.inf, 5.0, np.inf])

    try:
        popt, _ = curve_fit(_model, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        return None

    b = float(popt[1])
    if b <= 0:
        return None
    return b
