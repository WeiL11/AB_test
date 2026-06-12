"""
Frequentist hypothesis tests for the A/B testing platform.

Provides Welch's t-test for continuous metrics, an unpooled z-test for
proportions, and a chi-squared independence test for contingency tables.

All tests are two-sided by default.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats

from .base import FrequentistResult


# ---------------------------------------------------------------------------
# Welch's t-test (continuous metrics)
# ---------------------------------------------------------------------------

def welch_t_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Two-sample Welch's t-test for the difference in means.

    This test does **not** assume equal variances between groups
    (Behrens-Fisher problem).  Degrees of freedom are estimated via the
    Welch-Satterthwaite equation.

    Parameters
    ----------
    control : np.ndarray
        1-D array of observations for the control group.
    treatment : np.ndarray
        1-D array of observations for the treatment group.
    alpha : float, optional
        Significance level (default 0.05).

    Returns
    -------
    FrequentistResult
        Detailed test result including confidence interval, effect sizes,
        and significance flag.

    Raises
    ------
    ValueError
        If either array is empty or contains fewer than 2 observations.
    """
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    # --- input validation ---------------------------------------------------
    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("control and treatment must be 1-D arrays.")
    if control.size < 2:
        raise ValueError(
            f"control must contain at least 2 observations, got {control.size}."
        )
    if treatment.size < 2:
        raise ValueError(
            f"treatment must contain at least 2 observations, got {treatment.size}."
        )

    n_c = control.size
    n_t = treatment.size
    mean_c = float(np.mean(control))
    mean_t = float(np.mean(treatment))
    var_c = float(np.var(control, ddof=1))
    var_t = float(np.var(treatment, ddof=1))

    # --- handle zero-variance edge case -------------------------------------
    # When both groups have zero variance the test is degenerate.
    if var_c == 0.0 and var_t == 0.0:
        diff = mean_t - mean_c
        is_equal = np.isclose(mean_c, mean_t)
        return FrequentistResult(
            mean_control=mean_c,
            mean_treatment=mean_t,
            absolute_effect=diff,
            relative_effect=_relative_effect(mean_c, diff),
            ci_lower=diff,
            ci_upper=diff,
            p_value=1.0 if is_equal else 0.0,
            is_significant=not is_equal,
            test_statistic=0.0 if is_equal else float("inf") * np.sign(diff),
            degrees_of_freedom=float("inf"),
            standard_error=0.0,
            sample_size_control=int(n_c),
            sample_size_treatment=int(n_t),
        )

    # --- Welch-Satterthwaite degrees of freedom -----------------------------
    se_c = var_c / n_c
    se_t = var_t / n_t
    se_diff = np.sqrt(se_c + se_t)

    numerator = (se_c + se_t) ** 2
    denominator = (se_c ** 2) / (n_c - 1) + (se_t ** 2) / (n_t - 1)
    # Guard against denominator == 0 (happens if one group has zero variance)
    dof = numerator / denominator if denominator > 0 else float("inf")

    # --- test statistic & p-value -------------------------------------------
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    # scipy returns t = (mean_control - mean_treatment) / SE -- we want
    # (treatment - control), so flip the sign.
    # Actually, scipy.stats.ttest_ind computes (a - b), where a=control, b=treatment.
    # We want treatment - control, so negate t_stat.
    t_stat = -float(t_stat)
    p_value = float(p_value)

    diff = mean_t - mean_c

    # --- confidence interval for (treatment - control) ----------------------
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, dof)
    margin = t_crit * se_diff
    ci_lower = diff - margin
    ci_upper = diff + margin

    return FrequentistResult(
        mean_control=mean_c,
        mean_treatment=mean_t,
        absolute_effect=diff,
        relative_effect=_relative_effect(mean_c, diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        is_significant=p_value < alpha,
        test_statistic=t_stat,
        degrees_of_freedom=dof,
        standard_error=se_diff,
        sample_size_control=int(n_c),
        sample_size_treatment=int(n_t),
    )


# ---------------------------------------------------------------------------
# Z-test for two proportions
# ---------------------------------------------------------------------------

def z_test_proportions(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Unpooled two-proportion z-test.

    Tests H0: p_treatment == p_control vs. H1: p_treatment != p_control.

    The standard error uses each group's own proportion (unpooled),
    which is consistent with the Wald confidence interval.

    Parameters
    ----------
    successes_control : int
        Number of successes (conversions) in control.
    n_control : int
        Total observations in control.
    successes_treatment : int
        Number of successes (conversions) in treatment.
    n_treatment : int
        Total observations in treatment.
    alpha : float, optional
        Significance level (default 0.05).

    Returns
    -------
    FrequentistResult

    Raises
    ------
    ValueError
        If sample sizes are non-positive or successes exceed sample sizes.
    """
    # --- input validation ---------------------------------------------------
    if n_control <= 0:
        raise ValueError(f"n_control must be positive, got {n_control}.")
    if n_treatment <= 0:
        raise ValueError(f"n_treatment must be positive, got {n_treatment}.")
    if not (0 <= successes_control <= n_control):
        raise ValueError(
            f"successes_control ({successes_control}) must be between 0 and "
            f"n_control ({n_control})."
        )
    if not (0 <= successes_treatment <= n_treatment):
        raise ValueError(
            f"successes_treatment ({successes_treatment}) must be between 0 "
            f"and n_treatment ({n_treatment})."
        )

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    diff = p_t - p_c

    # --- standard error (unpooled) ------------------------------------------
    se_c = p_c * (1.0 - p_c) / n_control
    se_t = p_t * (1.0 - p_t) / n_treatment
    se_diff = np.sqrt(se_c + se_t)

    # --- handle degenerate cases (all successes or all failures in both) -----
    if se_diff == 0.0:
        is_equal = np.isclose(p_c, p_t)
        return FrequentistResult(
            mean_control=p_c,
            mean_treatment=p_t,
            absolute_effect=diff,
            relative_effect=_relative_effect(p_c, diff),
            ci_lower=diff,
            ci_upper=diff,
            p_value=1.0 if is_equal else 0.0,
            is_significant=not is_equal,
            test_statistic=0.0 if is_equal else float("inf") * np.sign(diff),
            degrees_of_freedom=None,  # z-test has no finite df
            standard_error=0.0,
            sample_size_control=int(n_control),
            sample_size_treatment=int(n_treatment),
        )

    z_stat = diff / se_diff
    p_value = 2.0 * stats.norm.sf(abs(z_stat))  # two-sided

    # --- confidence interval ------------------------------------------------
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    margin = z_crit * se_diff
    ci_lower = diff - margin
    ci_upper = diff + margin

    return FrequentistResult(
        mean_control=p_c,
        mean_treatment=p_t,
        absolute_effect=diff,
        relative_effect=_relative_effect(p_c, diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(p_value),
        is_significant=float(p_value) < alpha,
        test_statistic=float(z_stat),
        degrees_of_freedom=None,
        standard_error=float(se_diff),
        sample_size_control=int(n_control),
        sample_size_treatment=int(n_treatment),
    )


# ---------------------------------------------------------------------------
# Chi-squared independence test
# ---------------------------------------------------------------------------

def chi_squared_test(
    contingency_table: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chi-squared test of independence for a contingency table.

    Thin wrapper around :func:`scipy.stats.chi2_contingency` that returns a
    plain dictionary with all relevant outputs.

    Parameters
    ----------
    contingency_table : np.ndarray
        2-D array of observed frequencies.  Each row is a group (e.g.
        control / treatment), each column is an outcome category.
        All values must be non-negative integers.
    alpha : float, optional
        Significance level (default 0.05).

    Returns
    -------
    dict
        Keys:

        - ``chi2_statistic`` (float): The chi-squared test statistic.
        - ``p_value`` (float): The p-value from the chi-squared distribution.
        - ``degrees_of_freedom`` (int): Degrees of freedom
          ((rows - 1) * (cols - 1)).
        - ``expected_frequencies`` (np.ndarray): Expected counts under H0.
        - ``is_significant`` (bool): Whether p_value < alpha.

    Raises
    ------
    ValueError
        If the table is not at least 2x2 or contains negative values.
    """
    contingency_table = np.asarray(contingency_table, dtype=np.float64)

    if contingency_table.ndim != 2:
        raise ValueError("contingency_table must be a 2-D array.")
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            "contingency_table must be at least 2x2, got shape "
            f"{contingency_table.shape}."
        )
    if np.any(contingency_table < 0):
        raise ValueError("All values in contingency_table must be non-negative.")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "expected_frequencies": expected,
        "is_significant": float(p_value) < alpha,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _relative_effect(baseline: float, absolute_diff: float) -> float | None:
    """Compute relative effect, returning None when baseline is zero."""
    if baseline == 0.0:
        return None
    return absolute_diff / abs(baseline)
