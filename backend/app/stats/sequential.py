"""
Group Sequential Testing with Alpha-Spending Functions.

Solves the "peeking problem": without sequential testing, checking results
5 times at alpha=0.05 inflates the false positive rate to ~14.6%.

This module implements:
- O'Brien-Fleming spending: conservative early, aggressive late
- Pocock spending: constant boundaries across looks
- Lan-DeMets framework for flexible analysis timing

References:
- Lan & DeMets (1983). Discrete sequential boundaries for clinical trials.
- O'Brien & Fleming (1979). A multiple testing procedure for clinical trials.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy import stats
from dataclasses import dataclass


_VALID_SPENDING_FUNCTIONS = ("obrien_fleming", "pocock")


@dataclass(frozen=True)
class SequentialResult:
    """Result of a sequential analysis at one interim look.

    Attributes:
        analysis_number: Which look this is (1-indexed).
        total_analyses: Total planned looks (K).
        information_fraction: Fraction of max information collected
            (n_current / n_max), between 0 and 1.
        z_statistic: Current z-test statistic.
        boundary: Critical z-boundary at this look.
        p_value: Nominal (unadjusted) two-sided p-value for ``z_statistic``.
        alpha_spent: Cumulative alpha spent through this analysis.
        can_reject: Whether ``|z_statistic|`` exceeds the boundary.
        recommendation: One of ``"stop_reject"``, ``"stop_futility"``,
            or ``"continue"``.
    """

    analysis_number: int
    total_analyses: int
    information_fraction: float
    z_statistic: float
    boundary: float
    p_value: float
    alpha_spent: float
    can_reject: bool
    recommendation: str


class GroupSequentialTest:
    """Group sequential test using alpha-spending functions.

    The Lan-DeMets alpha-spending approach allows experimenters to monitor
    results at pre-planned interim analyses while controlling the overall
    Type I error rate.  At each look the method "spends" a portion of the
    total alpha budget; the boundary at that look is set so that only the
    incremental alpha is consumed.

    Usage::

        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")

        # At each interim analysis:
        result = gst.analyze(
            analysis_number=1,
            z_statistic=1.8,
            information_fraction=0.2,
        )
        if result.can_reject:
            print("Stop: significant result")

    Parameters
    ----------
    alpha : float, optional
        Overall (experiment-wide) significance level.  Default 0.05.
    n_analyses : int, optional
        Total number of planned interim looks including the final analysis.
        Default 5.
    spending : str, optional
        Alpha-spending function to use.  One of ``"obrien_fleming"`` or
        ``"pocock"``.  Default ``"obrien_fleming"``.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1), n_analyses < 1, or spending is unknown.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_analyses: int = 5,
        spending: str = "obrien_fleming",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_analyses < 1:
            raise ValueError(
                f"n_analyses must be at least 1, got {n_analyses}."
            )
        if spending not in _VALID_SPENDING_FUNCTIONS:
            raise ValueError(
                f"spending must be one of {_VALID_SPENDING_FUNCTIONS}, "
                f"got '{spending}'."
            )

        self.alpha = alpha
        self.n_analyses = n_analyses
        self.spending = spending

    # --------------------------------------------------------------------- #
    # Alpha-spending functions
    # --------------------------------------------------------------------- #

    def _alpha_spending(self, t: float) -> float:
        """Compute cumulative alpha spent at information fraction *t*.

        O'Brien-Fleming spending::

            alpha_spent(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))

        This spends almost no alpha early (e.g. boundaries near z = 4.5
        at 20%) and relaxes to approximately the unadjusted boundary
        (z ~ 2.0) at the final look.

        Pocock spending::

            alpha_spent(t) = alpha * ln(1 + (e - 1) * t)

        This spreads alpha more uniformly, yielding nearly constant
        boundaries across looks (z ~ 2.4--2.5).

        Parameters
        ----------
        t : float
            Information fraction, clamped to [0, 1].

        Returns
        -------
        float
            Cumulative alpha spent at information fraction *t*.
        """
        t = float(np.clip(t, 0.0, 1.0))

        if t <= 0.0:
            return 0.0

        if self.spending == "obrien_fleming":
            z_crit = stats.norm.ppf(1.0 - self.alpha / 2.0)
            spent = 2.0 * (1.0 - stats.norm.cdf(z_crit / math.sqrt(t)))
            # Clamp to [0, alpha] for numerical safety.
            return float(min(spent, self.alpha))

        # Pocock
        spent = self.alpha * math.log(1.0 + (math.e - 1.0) * t)
        return float(min(spent, self.alpha))

    # --------------------------------------------------------------------- #
    # Boundary computation
    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_boundary(alpha_increment: float) -> float:
        """Find the z-boundary that spends exactly *alpha_increment*.

        The boundary satisfies ``P(|Z| > boundary) = alpha_increment``,
        i.e. ``boundary = z_{1 - alpha_increment / 2}``.

        Parameters
        ----------
        alpha_increment : float
            The alpha to spend at this single look.

        Returns
        -------
        float
            The critical z-value (always positive).
        """
        if alpha_increment <= 0.0:
            return float("inf")
        if alpha_increment >= 1.0:
            return 0.0

        return float(stats.norm.ppf(1.0 - alpha_increment / 2.0))

    # --------------------------------------------------------------------- #
    # Main analysis method
    # --------------------------------------------------------------------- #

    def analyze(
        self,
        analysis_number: int,
        z_statistic: float,
        information_fraction: Optional[float] = None,
    ) -> SequentialResult:
        """Perform one interim analysis.

        Computes the incremental alpha to spend at this look, derives the
        corresponding z-boundary, and compares ``|z_statistic|`` against it.

        Parameters
        ----------
        analysis_number : int
            Which look this is (1-indexed, must be between 1 and
            ``n_analyses``).
        z_statistic : float
            The z-statistic computed from the current accumulated data.
        information_fraction : float or None, optional
            Fraction of the maximum planned sample size that has been
            collected.  If ``None``, defaults to
            ``analysis_number / n_analyses`` (equally spaced looks).

        Returns
        -------
        SequentialResult

        Raises
        ------
        ValueError
            If ``analysis_number`` is out of range or
            ``information_fraction`` is not in (0, 1].
        """
        if not (1 <= analysis_number <= self.n_analyses):
            raise ValueError(
                f"analysis_number must be between 1 and {self.n_analyses}, "
                f"got {analysis_number}."
            )

        if information_fraction is None:
            information_fraction = analysis_number / self.n_analyses

        if not (0.0 < information_fraction <= 1.0):
            raise ValueError(
                f"information_fraction must be in (0, 1], "
                f"got {information_fraction}."
            )

        # Cumulative alpha spent through the current and previous looks.
        alpha_current = self._alpha_spending(information_fraction)

        if analysis_number == 1:
            alpha_previous = 0.0
        else:
            t_prev = (analysis_number - 1) / self.n_analyses
            # If the caller supplies a custom information_fraction, compute
            # the previous fraction proportionally.
            if information_fraction != analysis_number / self.n_analyses:
                t_prev = information_fraction * (analysis_number - 1) / analysis_number
            alpha_previous = self._alpha_spending(t_prev)

        alpha_increment = max(alpha_current - alpha_previous, 0.0)

        boundary = self._compute_boundary(alpha_increment)

        # Nominal two-sided p-value (unadjusted).
        p_value = float(2.0 * stats.norm.sf(abs(z_statistic)))

        can_reject = abs(z_statistic) > boundary

        # Decision logic.
        if can_reject:
            recommendation = "stop_reject"
        elif analysis_number == self.n_analyses:
            # Final analysis reached without rejection.
            recommendation = "stop_futility"
        else:
            recommendation = "continue"

        return SequentialResult(
            analysis_number=analysis_number,
            total_analyses=self.n_analyses,
            information_fraction=information_fraction,
            z_statistic=z_statistic,
            boundary=boundary,
            p_value=p_value,
            alpha_spent=alpha_current,
            can_reject=can_reject,
            recommendation=recommendation,
        )

    # --------------------------------------------------------------------- #
    # Pre-compute all boundaries
    # --------------------------------------------------------------------- #

    def compute_all_boundaries(self) -> List[tuple[float, float]]:
        """Pre-compute boundaries for all planned analyses.

        Useful for plotting the boundary corridor before data collection
        begins.  Assumes equally spaced information fractions.

        Returns
        -------
        list[tuple[float, float]]
            A list of ``(information_fraction, boundary)`` pairs, one per
            planned analysis.
        """
        boundaries: List[tuple[float, float]] = []
        alpha_prev = 0.0

        for k in range(1, self.n_analyses + 1):
            t_k = k / self.n_analyses
            alpha_k = self._alpha_spending(t_k)
            alpha_increment = max(alpha_k - alpha_prev, 0.0)
            boundary = self._compute_boundary(alpha_increment)
            boundaries.append((t_k, boundary))
            alpha_prev = alpha_k

        return boundaries


# ------------------------------------------------------------------------- #
# Convenience function
# ------------------------------------------------------------------------- #

def sequential_z_test(
    control: np.ndarray,
    treatment: np.ndarray,
    analysis_number: int,
    n_analyses: int,
    max_sample_size: Optional[int] = None,
    alpha: float = 0.05,
    spending: str = "obrien_fleming",
) -> SequentialResult:
    """Compute a z-statistic from raw data and run a sequential test.

    For continuous data the two-sample z-statistic is::

        z = (mean_treatment - mean_control) / SE

    where ``SE = sqrt(var_control / n_control + var_treatment / n_treatment)``.

    Parameters
    ----------
    control : np.ndarray
        1-D array of control-group observations.
    treatment : np.ndarray
        1-D array of treatment-group observations.
    analysis_number : int
        Which interim look this is (1-indexed).
    n_analyses : int
        Total number of planned looks.
    max_sample_size : int or None, optional
        Maximum planned per-group sample size (used to compute the
        information fraction).  If ``None``, defaults to the current
        per-group sample size at the final analysis (i.e. equally spaced
        looks are assumed).
    alpha : float, optional
        Overall significance level.  Default 0.05.
    spending : str, optional
        Alpha-spending function.  Default ``"obrien_fleming"``.

    Returns
    -------
    SequentialResult

    Raises
    ------
    ValueError
        If either array has fewer than 2 observations.
    """
    control = np.asarray(control, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)

    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("control and treatment must be 1-D arrays.")
    if control.size < 2:
        raise ValueError(
            f"control must contain at least 2 observations, got {control.size}."
        )
    if treatment.size < 2:
        raise ValueError(
            f"treatment must contain at least 2 observations, "
            f"got {treatment.size}."
        )

    n_c = control.size
    n_t = treatment.size
    mean_c = float(np.mean(control))
    mean_t = float(np.mean(treatment))
    var_c = float(np.var(control, ddof=1))
    var_t = float(np.var(treatment, ddof=1))

    se = math.sqrt(var_c / n_c + var_t / n_t)
    if se == 0.0:
        # Degenerate: both groups have zero variance.
        z_stat = 0.0 if np.isclose(mean_c, mean_t) else (
            float("inf") * np.sign(mean_t - mean_c)
        )
    else:
        z_stat = (mean_t - mean_c) / se

    # Information fraction.
    if max_sample_size is not None:
        n_current = min(n_c, n_t)
        information_fraction = float(np.clip(n_current / max_sample_size, 0.0, 1.0))
    else:
        information_fraction = analysis_number / n_analyses

    gst = GroupSequentialTest(
        alpha=alpha,
        n_analyses=n_analyses,
        spending=spending,
    )

    return gst.analyze(
        analysis_number=analysis_number,
        z_statistic=z_stat,
        information_fraction=information_fraction,
    )
