"""
Power analysis utilities for the A/B testing platform.

Functions for computing required sample sizes, minimum detectable effects,
statistical power, power curves, and experiment duration estimates.

Supports both binomial (conversion rate) and continuous metrics.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy import optimize, stats

from .base import PowerCurvePoint, PowerResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binomial",
    variance: Optional[float] = None,
) -> PowerResult:
    """Compute the per-variant sample size needed to detect an effect.

    Parameters
    ----------
    baseline_rate : float
        The expected value of the metric under control.  For binomial
        metrics this is the control conversion rate (e.g. 0.10 for 10%).
        For continuous metrics this is the control mean.
    minimum_detectable_effect : float
        The smallest **absolute** difference (treatment - control) you
        want to be able to detect.  Must be positive.
    alpha : float, optional
        Significance level (two-sided).  Default 0.05.
    power : float, optional
        Desired statistical power (1 - beta).  Default 0.80.
    metric_type : str, optional
        ``"binomial"`` for conversion-rate metrics or ``"continuous"`` for
        revenue / time-on-page style metrics.
    variance : float or None, optional
        Population variance of the metric.  **Required** when
        ``metric_type="continuous"``.  Ignored for binomial metrics (the
        variance is derived from the proportion).

    Returns
    -------
    PowerResult

    Raises
    ------
    ValueError
        On invalid inputs (e.g. MDE <= 0, unknown metric type, missing
        variance for continuous metrics).
    """
    _validate_common(alpha, power, minimum_detectable_effect, metric_type, variance)

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    delta = minimum_detectable_effect

    if metric_type == "binomial":
        p1 = baseline_rate
        p2 = baseline_rate + delta
        # Variance under each hypothesis arm
        var_sum = p1 * (1.0 - p1) + p2 * (1.0 - p2)
        n = math.ceil((z_alpha + z_beta) ** 2 * var_sum / (delta ** 2))
    else:
        # continuous
        assert variance is not None  # guaranteed by validation
        n = math.ceil((z_alpha + z_beta) ** 2 * 2.0 * variance / (delta ** 2))

    return PowerResult(
        required_sample_size_per_variant=n,
        total_sample_size=n * 2,
        power=power,
        alpha=alpha,
        minimum_detectable_effect=delta,
        baseline_rate=baseline_rate,
    )


def minimum_detectable_effect_func(
    sample_size_per_variant: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binomial",
    variance: Optional[float] = None,
) -> float:
    """Compute the smallest effect detectable with a given sample size.

    This is the inverse of :func:`required_sample_size`: given a fixed
    per-variant sample size, it finds the minimum absolute effect that can
    be detected at the requested power and significance level.

    The solution is found numerically via Brent's method
    (:func:`scipy.optimize.brentq`).

    Parameters
    ----------
    sample_size_per_variant : int
        Number of observations per arm.
    baseline_rate : float
        Control metric value (proportion for binomial, mean for continuous).
    alpha : float, optional
        Significance level (two-sided).  Default 0.05.
    power : float, optional
        Desired statistical power.  Default 0.80.
    metric_type : str, optional
        ``"binomial"`` or ``"continuous"``.
    variance : float or None, optional
        Population variance (required for continuous metrics).

    Returns
    -------
    float
        The minimum detectable absolute effect.

    Raises
    ------
    ValueError
        If the sample size is non-positive or other inputs are invalid.
    """
    if sample_size_per_variant <= 0:
        raise ValueError(
            f"sample_size_per_variant must be positive, got {sample_size_per_variant}."
        )
    if metric_type not in ("binomial", "continuous"):
        raise ValueError(f"metric_type must be 'binomial' or 'continuous', got '{metric_type}'.")
    if metric_type == "continuous" and variance is None:
        raise ValueError("variance is required when metric_type='continuous'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}.")

    # Define the function whose root we seek:
    #   f(delta) = compute_power(..., delta) - desired_power
    def _objective(delta: float) -> float:
        pw = compute_power(
            sample_size_per_variant=sample_size_per_variant,
            baseline_rate=baseline_rate,
            effect_size=delta,
            alpha=alpha,
            metric_type=metric_type,
            variance=variance,
        )
        return pw - power

    # Bracket: at delta near 0 power -> alpha (< desired), at large delta
    # power -> 1.  We pick a sensible upper bound.
    if metric_type == "binomial":
        # Effect cannot exceed the range [0, 1] for a proportion.
        upper = min(1.0 - baseline_rate, baseline_rate, 0.99)
        if upper <= 0:
            upper = 0.99
    else:
        # For continuous metrics use a heuristic upper bound.
        assert variance is not None
        upper = 10.0 * math.sqrt(variance)

    lower = 1e-12  # avoid exactly zero

    # Ensure the bracket is valid (f(lower) < 0 and f(upper) > 0).
    if _objective(upper) <= 0:
        # Sample size is too small to ever reach desired power even at the
        # largest plausible effect -- return the upper bound as a conservative
        # answer.
        return upper

    mde = optimize.brentq(_objective, lower, upper, xtol=1e-9, maxiter=500)
    return float(mde)


def compute_power(
    sample_size_per_variant: int,
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    metric_type: str = "binomial",
    variance: Optional[float] = None,
) -> float:
    """Compute the statistical power for a given design.

    Parameters
    ----------
    sample_size_per_variant : int
        Observations per arm.
    baseline_rate : float
        Control metric value.
    effect_size : float
        Absolute effect (treatment - control).  Must be positive.
    alpha : float, optional
        Significance level (two-sided).  Default 0.05.
    metric_type : str, optional
        ``"binomial"`` or ``"continuous"``.
    variance : float or None, optional
        Population variance (required for continuous metrics).

    Returns
    -------
    float
        Power in [0, 1].

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if sample_size_per_variant <= 0:
        raise ValueError(
            f"sample_size_per_variant must be positive, got {sample_size_per_variant}."
        )
    if effect_size <= 0:
        raise ValueError(f"effect_size must be positive, got {effect_size}.")
    if metric_type not in ("binomial", "continuous"):
        raise ValueError(f"metric_type must be 'binomial' or 'continuous', got '{metric_type}'.")
    if metric_type == "continuous" and variance is None:
        raise ValueError("variance is required when metric_type='continuous'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n = sample_size_per_variant
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)

    if metric_type == "binomial":
        p1 = baseline_rate
        p2 = baseline_rate + effect_size
        se = math.sqrt(p1 * (1.0 - p1) / n + p2 * (1.0 - p2) / n)
    else:
        assert variance is not None
        se = math.sqrt(2.0 * variance / n)

    if se == 0.0:
        # Degenerate case: no variability -> power is 1 if there is any
        # effect, which there is since effect_size > 0.
        return 1.0

    # Non-centrality parameter
    ncp = effect_size / se

    # Power = P(|Z| > z_alpha | ncp)
    #       = P(Z > z_alpha - ncp) + P(Z < -z_alpha - ncp)
    # The second term is negligible for moderate effects, but we include it
    # for correctness.
    power_val = (
        stats.norm.sf(z_alpha - ncp)
        + stats.norm.cdf(-z_alpha - ncp)
    )

    return float(np.clip(power_val, 0.0, 1.0))


def power_curve(
    baseline_rate: float,
    sample_size_per_variant: int,
    alpha: float = 0.05,
    metric_type: str = "binomial",
    variance: Optional[float] = None,
    n_points: int = 20,
) -> List[PowerCurvePoint]:
    """Generate a power curve over a range of effect sizes.

    Useful for plotting how power changes as the true effect varies.

    Parameters
    ----------
    baseline_rate : float
        Control metric value.
    sample_size_per_variant : int
        Observations per arm.
    alpha : float, optional
        Significance level.  Default 0.05.
    metric_type : str, optional
        ``"binomial"`` or ``"continuous"``.
    variance : float or None, optional
        Population variance (required for continuous metrics).
    n_points : int, optional
        Number of points on the curve (default 20).

    Returns
    -------
    list[PowerCurvePoint]
        Sorted by ascending ``effect_size``.

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if sample_size_per_variant <= 0:
        raise ValueError(
            f"sample_size_per_variant must be positive, got {sample_size_per_variant}."
        )
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}.")
    if metric_type not in ("binomial", "continuous"):
        raise ValueError(f"metric_type must be 'binomial' or 'continuous', got '{metric_type}'.")
    if metric_type == "continuous" and variance is None:
        raise ValueError("variance is required when metric_type='continuous'.")

    # Determine a reasonable range of effect sizes.
    if metric_type == "binomial":
        # From a tiny effect up to whichever is smaller: the room above
        # baseline or the baseline itself (so p2 stays in [0, 1]).
        max_effect = min(1.0 - baseline_rate, baseline_rate)
        if max_effect <= 0:
            max_effect = max(1.0 - baseline_rate, baseline_rate, 0.01)
    else:
        assert variance is not None
        # Heuristic: go up to ~1 standard deviation.
        max_effect = math.sqrt(variance)
        if max_effect == 0:
            max_effect = 1.0

    # Small positive lower bound to avoid zero.
    min_effect = max_effect * 0.01

    effects = np.linspace(min_effect, max_effect, n_points)

    points: List[PowerCurvePoint] = []
    for eff in effects:
        pw = compute_power(
            sample_size_per_variant=sample_size_per_variant,
            baseline_rate=baseline_rate,
            effect_size=float(eff),
            alpha=alpha,
            metric_type=metric_type,
            variance=variance,
        )
        points.append(PowerCurvePoint(effect_size=float(eff), power=pw))

    return points


def estimate_duration(
    sample_size_per_variant: int,
    daily_traffic: int,
    n_variants: int = 2,
    allocation_pct: float = 100.0,
) -> int:
    """Estimate how many days an experiment needs to run.

    Parameters
    ----------
    sample_size_per_variant : int
        Required observations per arm.
    daily_traffic : int
        Total eligible visitors per day (before allocation split).
    n_variants : int, optional
        Number of variants including control (default 2).
    allocation_pct : float, optional
        Percentage of daily traffic allocated to the experiment (0-100).
        Default 100 (all traffic participates).

    Returns
    -------
    int
        Estimated number of days (always at least 1).

    Raises
    ------
    ValueError
        If any input is non-positive or allocation is out of range.
    """
    if sample_size_per_variant <= 0:
        raise ValueError(
            f"sample_size_per_variant must be positive, got {sample_size_per_variant}."
        )
    if daily_traffic <= 0:
        raise ValueError(f"daily_traffic must be positive, got {daily_traffic}.")
    if n_variants < 2:
        raise ValueError(f"n_variants must be at least 2, got {n_variants}.")
    if not (0.0 < allocation_pct <= 100.0):
        raise ValueError(
            f"allocation_pct must be in (0, 100], got {allocation_pct}."
        )

    total_needed = sample_size_per_variant * n_variants
    effective_daily = daily_traffic * (allocation_pct / 100.0)
    days = math.ceil(total_needed / effective_daily)
    return max(days, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_common(
    alpha: float,
    power: float,
    mde: float,
    metric_type: str,
    variance: Optional[float],
) -> None:
    """Shared validation for sample-size / power functions."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if mde <= 0:
        raise ValueError(
            f"minimum_detectable_effect must be positive, got {mde}."
        )
    if metric_type not in ("binomial", "continuous"):
        raise ValueError(
            f"metric_type must be 'binomial' or 'continuous', got '{metric_type}'."
        )
    if metric_type == "continuous" and variance is None:
        raise ValueError(
            "variance is required when metric_type='continuous'."
        )
