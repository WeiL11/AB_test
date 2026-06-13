"""
Always-Valid Inference via Confidence Sequences.

Unlike fixed-horizon confidence intervals, confidence sequences are valid
at ALL sample sizes simultaneously.  You can peek at results at any time
without inflating the false positive rate.

The key property: for ANY stopping time tau (even data-dependent),
``P(mu in CS_tau) >= 1 - alpha``.

The tradeoff: confidence-sequence intervals are wider than fixed-horizon
CIs, especially at small sample sizes.

This module implements the normal-mixture approach based on the mixture
sequential probability ratio test (mSPRT).

References:
- Howard et al. (2021). Time-uniform, nonparametric, nonasymptotic
  confidence sequences.
- Johari et al. (2017). Peeking at A/B Tests.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy import stats
from dataclasses import dataclass


# Minimum per-group sample size to avoid degenerate estimates.
_MIN_SAMPLE_SIZE = 2


@dataclass(frozen=True)
class ConfidenceSequenceResult:
    """Result of always-valid inference at the current sample size.

    Attributes:
        sample_size: Effective per-group sample size used (min of the two
            group sizes).
        mean_difference: Point estimate of the treatment effect
            (treatment mean - control mean).
        ci_lower: Lower bound of the confidence sequence.
        ci_upper: Upper bound of the confidence sequence.
        is_significant: Whether the confidence sequence excludes zero
            (i.e. both bounds are on the same side of zero).
        mixing_variance: The mixing-variance parameter *v* that was used.
        alpha: Significance level.
    """

    sample_size: int
    mean_difference: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    mixing_variance: float
    alpha: float


def confidence_sequence(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    v_opt: Optional[float] = None,
) -> ConfidenceSequenceResult:
    """Compute a confidence sequence for the difference in means.

    Uses the normal-mixture mSPRT approach (Johari et al. 2017).  The
    confidence-sequence half-width at effective sample size *n* is::

        half_width = sqrt(
            (2 * sigma2 * (sigma2 + n * v)) / (n**2 * v)
            * log( sqrt((sigma2 + n * v) / sigma2) / alpha )
        )

    where:

    - ``sigma2 = var(control) + var(treatment)`` is the pooled
      per-observation variance (the sum of sample variances from each
      group, NOT the variance of the mean difference).
    - ``v`` is the mixing-variance parameter that controls the shape of
      the confidence sequence (wider v makes the CS tighter at larger
      sample sizes but wider at smaller ones).
    - ``n = min(n_control, n_treatment)`` is the effective per-group
      sample size.

    If ``v_opt`` is ``None``, it defaults to ``sigma2``, which provides a
    reasonable balance for sample sizes between 100 and 100,000.

    Parameters
    ----------
    control : np.ndarray
        1-D array of control-group observations.
    treatment : np.ndarray
        1-D array of treatment-group observations.
    alpha : float, optional
        Significance level.  Default 0.05.
    v_opt : float or None, optional
        Mixing-variance parameter.  If ``None``, the sample variance of
        the mean difference is used as a sensible default.

    Returns
    -------
    ConfidenceSequenceResult

    Raises
    ------
    ValueError
        If either array has fewer than 2 observations or alpha is out of
        range.
    """
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    # --- input validation ------------------------------------------------- #
    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("control and treatment must be 1-D arrays.")
    if control.size < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"control must contain at least {_MIN_SAMPLE_SIZE} observations, "
            f"got {control.size}."
        )
    if treatment.size < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"treatment must contain at least {_MIN_SAMPLE_SIZE} observations, "
            f"got {treatment.size}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if v_opt is not None and v_opt <= 0.0:
        raise ValueError(f"v_opt must be positive, got {v_opt}.")

    n_c = control.size
    n_t = treatment.size
    n = min(n_c, n_t)

    mean_c = float(np.mean(control))
    mean_t = float(np.mean(treatment))
    mean_diff = mean_t - mean_c

    var_c = float(np.var(control, ddof=1))
    var_t = float(np.var(treatment, ddof=1))

    # Pooled per-observation variance (NOT variance of the mean).
    # The mSPRT formula expects per-observation scale; dividing by n
    # here would make the CS collapse at large sample sizes.
    sigma2 = var_c + var_t

    # --- handle degenerate zero-variance case ----------------------------- #
    if sigma2 <= 0.0:
        # Both groups have zero variance.  The CI collapses to a point.
        return ConfidenceSequenceResult(
            sample_size=n,
            mean_difference=mean_diff,
            ci_lower=mean_diff,
            ci_upper=mean_diff,
            is_significant=not np.isclose(mean_diff, 0.0),
            mixing_variance=0.0,
            alpha=alpha,
        )

    v = v_opt if v_opt is not None else sigma2

    half_width = _msprt_half_width(sigma2=sigma2, n=n, v=v, alpha=alpha)

    ci_lower = mean_diff - half_width
    ci_upper = mean_diff + half_width

    # The CS excludes zero when both bounds are strictly on the same side.
    is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)

    return ConfidenceSequenceResult(
        sample_size=n,
        mean_difference=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
        mixing_variance=v,
        alpha=alpha,
    )


def compute_confidence_sequence_over_time(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    min_samples: int = 100,
    n_points: int = 50,
) -> List[ConfidenceSequenceResult]:
    """Compute the confidence sequence at multiple sample sizes.

    Useful for plotting how the CS evolves as data accumulates.  The
    function evaluates the CS at ``n_points`` equally spaced sample sizes
    from ``min_samples`` up to the full dataset.

    Parameters
    ----------
    control : np.ndarray
        1-D array of control-group observations (full dataset).
    treatment : np.ndarray
        1-D array of treatment-group observations (full dataset).
    alpha : float, optional
        Significance level.  Default 0.05.
    min_samples : int, optional
        Smallest per-group sample size to start from.  Default 100.
    n_points : int, optional
        Number of checkpoints at which to evaluate the CS.  Default 50.

    Returns
    -------
    list[ConfidenceSequenceResult]
        One result per checkpoint, ordered by increasing sample size.

    Raises
    ------
    ValueError
        If the arrays are smaller than ``min_samples`` or inputs are
        invalid.
    """
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("control and treatment must be 1-D arrays.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}.")
    if min_samples < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"min_samples must be at least {_MIN_SAMPLE_SIZE}, "
            f"got {min_samples}."
        )

    max_n = min(control.size, treatment.size)

    if max_n < min_samples:
        raise ValueError(
            f"Both arrays must have at least min_samples={min_samples} "
            f"observations, but the smaller group has only {max_n}."
        )

    # Generate checkpoint sample sizes.
    if n_points == 1:
        checkpoints = [max_n]
    else:
        checkpoints = np.linspace(min_samples, max_n, n_points, dtype=int)
        # Ensure uniqueness and sorted order.
        checkpoints = sorted(set(int(c) for c in checkpoints))

    # Compute a single v_opt from the full dataset so the CS is consistent
    # across checkpoints (using a fixed mixing variance).
    var_c_full = float(np.var(control, ddof=1))
    var_t_full = float(np.var(treatment, ddof=1))
    sigma2_full = var_c_full + var_t_full
    v_opt = sigma2_full if sigma2_full > 0.0 else 1.0

    results: List[ConfidenceSequenceResult] = []
    for size in checkpoints:
        result = confidence_sequence(
            control=control[:size],
            treatment=treatment[:size],
            alpha=alpha,
            v_opt=v_opt,
        )
        results.append(result)

    return results


# ------------------------------------------------------------------------- #
# Internal helpers
# ------------------------------------------------------------------------- #

def _msprt_half_width(
    sigma2: float,
    n: int,
    v: float,
    alpha: float,
) -> float:
    """Compute the mSPRT confidence-sequence half-width.

    Implements the formula from Johari et al. (2017)::

        half_width = sqrt(
            (2 * sigma2 * (sigma2 + n * v)) / (n**2 * v)
            * log( sqrt((sigma2 + n * v) / sigma2) / alpha )
        )

    Parameters
    ----------
    sigma2 : float
        Pooled per-observation variance (must be positive).
    n : int
        Effective per-group sample size (must be positive).
    v : float
        Mixing-variance parameter (must be positive).
    alpha : float
        Significance level (must be in (0, 1)).

    Returns
    -------
    float
        The half-width of the confidence sequence.
    """
    if sigma2 <= 0.0 or n <= 0 or v <= 0.0:
        return float("inf")

    nv = n * v
    ratio = (sigma2 + nv) / sigma2  # always >= 1

    # The argument to log must be positive.  Since ratio >= 1 and
    # alpha < 1, sqrt(ratio) / alpha > 0.
    log_arg = math.sqrt(ratio) / alpha
    if log_arg <= 0.0:
        return float("inf")

    variance_term = (2.0 * sigma2 * (sigma2 + nv)) / (n ** 2 * v)
    log_term = math.log(log_arg)

    return math.sqrt(variance_term * log_term)
