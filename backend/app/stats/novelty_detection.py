"""
Novelty and Primacy Effect Detection.

Many experiments show strong initial effects that decay over time (novelty effect)
or weak initial effects that grow (primacy effect). Declaring a winner based on
early data without checking for these patterns is dangerous.

Approach:
    1. Compute the daily treatment effect (treatment_mean - control_mean) for each day
    2. Fit a weighted linear regression: effect ~ day, weighted by daily sample size
    3. Test whether the slope is significantly different from zero

    Negative slope = novelty effect (effect wearing off)
    Positive slope = primacy effect (effect growing)
    Near-zero slope = stable effect (safe to interpret)

References:
    - Hohnhold et al. (2015). Focusing on the Long-term: It's Good for Users and Business. KDD.
"""
import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass


@dataclass(frozen=True)
class NoveltyResult:
    """Result of novelty/primacy effect detection."""
    slope: float                    # Trend in daily effect (units per day)
    slope_se: float                 # Standard error of slope
    intercept: float                # Estimated effect at day 0
    p_value: float                  # Significance of slope
    is_significant: bool            # Whether slope significantly differs from zero
    effect_type: str                # "novelty" | "primacy" | "stable"
    r_squared: float                # How well the linear model fits
    n_days: int                     # Number of days analyzed
    daily_effects: list[float]      # Daily effect values
    daily_sample_sizes: list[int]   # Daily sample sizes


def detect_novelty_effect(
    daily_effects: list[float],
    daily_sample_sizes: list[int] | None = None,
    alpha: float = 0.05,
    min_days: int = 5,
) -> NoveltyResult:
    """
    Test whether the treatment effect is trending over time.

    Args:
        daily_effects: Daily treatment effect (treatment_mean - control_mean) per day
        daily_sample_sizes: Sample size per day (used as regression weights).
                           If None, equal weights are used.
        alpha: Significance level for the trend test
        min_days: Minimum number of days required for analysis

    Returns:
        NoveltyResult with trend analysis

    Raises:
        ValueError: If fewer than min_days of data
    """
    n_days = len(daily_effects)
    if n_days < min_days:
        raise ValueError(f"Need at least {min_days} days of data, got {n_days}")

    effects = np.array(daily_effects, dtype=float)
    days = np.arange(n_days, dtype=float)

    if daily_sample_sizes is not None:
        weights = np.array(daily_sample_sizes, dtype=float)
        if len(weights) != n_days:
            raise ValueError("daily_sample_sizes must match length of daily_effects")
        # Normalize weights
        weights = weights / weights.sum() * n_days
    else:
        weights = np.ones(n_days)

    # Weighted linear regression: effect = intercept + slope * day
    # Using weighted least squares
    W = np.diag(weights)
    X = np.column_stack([np.ones(n_days), days])

    # Beta = (X'WX)^{-1} X'Wy
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ effects

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        # Singular matrix fallback
        beta = np.array([np.mean(effects), 0.0])

    intercept = float(beta[0])
    slope = float(beta[1])

    # Compute residuals and standard error of slope
    predicted = X @ beta
    residuals = effects - predicted

    # Weighted residual sum of squares
    weighted_rss = float(residuals.T @ W @ residuals)

    # Degrees of freedom
    df = n_days - 2
    if df > 0:
        mse = weighted_rss / df
        # Variance of slope: MSE * [(X'WX)^{-1}]_{1,1}
        try:
            var_beta = mse * np.linalg.inv(XtWX)
            slope_se = float(np.sqrt(var_beta[1, 1]))
        except np.linalg.LinAlgError:
            slope_se = float('inf')
    else:
        slope_se = float('inf')

    # T-test for slope
    if slope_se > 0 and np.isfinite(slope_se):
        t_stat = slope / slope_se
        p_value = float(2 * sp_stats.t.sf(abs(t_stat), df))
    else:
        t_stat = 0.0
        p_value = 1.0

    is_significant = p_value < alpha

    # R-squared (coefficient of determination)
    ss_total = float(weights @ (effects - np.average(effects, weights=weights))**2)
    ss_residual = weighted_rss
    if ss_total > 1e-10:
        r_squared = 1 - ss_residual / ss_total
    else:
        r_squared = 0.0

    # Classify effect type
    if is_significant and slope < 0:
        effect_type = "novelty"
    elif is_significant and slope > 0:
        effect_type = "primacy"
    else:
        effect_type = "stable"

    return NoveltyResult(
        slope=slope,
        slope_se=slope_se,
        intercept=intercept,
        p_value=p_value,
        is_significant=is_significant,
        effect_type=effect_type,
        r_squared=float(max(0, r_squared)),
        n_days=n_days,
        daily_effects=list(daily_effects),
        daily_sample_sizes=list(daily_sample_sizes) if daily_sample_sizes else [1] * n_days,
    )


def compute_daily_effects(
    control_values: list[list[float]],
    treatment_values: list[list[float]],
) -> tuple[list[float], list[int]]:
    """
    Compute daily treatment effects from day-bucketed observations.

    Args:
        control_values: List of lists, where control_values[day] contains
                       all control observations for that day
        treatment_values: Same structure for treatment group

    Returns:
        Tuple of (daily_effects, daily_sample_sizes)
    """
    if len(control_values) != len(treatment_values):
        raise ValueError("Control and treatment must have same number of days")

    daily_effects = []
    daily_sizes = []

    for day_c, day_t in zip(control_values, treatment_values):
        if len(day_c) == 0 or len(day_t) == 0:
            continue
        effect = float(np.mean(day_t) - np.mean(day_c))
        size = len(day_c) + len(day_t)
        daily_effects.append(effect)
        daily_sizes.append(size)

    return daily_effects, daily_sizes
