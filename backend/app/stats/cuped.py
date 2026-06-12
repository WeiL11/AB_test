"""
CUPED: Controlled-experiment Using Pre-Experiment Data.

Reduces metric variance by 30-50% using pre-experiment user behavior
as a covariate, allowing experiments to reach significance with fewer users.

Mathematical foundation:
    Y_adjusted = Y - theta * (X - E[X])

    where:
        Y = post-experiment metric (e.g., revenue during experiment)
        X = pre-experiment covariate (e.g., revenue 2 weeks before)
        theta = Cov(Y, X) / Var(X)  (optimal regression coefficient)

    Var(Y_adjusted) = Var(Y) * (1 - rho^2)
    where rho = Cor(Y, X)

    If rho = 0.7, variance reduction = 1 - 0.49 = 51%

The key insight: if we know that user_123 spent $50/week before the experiment,
and they spend $52 during the experiment, the $2 *difference* is more informative
than the raw $52 — because we've removed the between-user noise.

CUPED is unbiased: E[Y_adjusted] = E[Y] since E[X - E[X]] = 0.
The adjustment only reduces variance, never introduces bias.

References:
- Deng et al. (2013). Improving the Sensitivity of Online Controlled Experiments
  by Utilizing Pre-Experiment Data. WSDM.
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class CUPEDResult:
    """Result of CUPED variance reduction."""
    # Adjusted means
    adjusted_mean_control: float
    adjusted_mean_treatment: float
    adjusted_effect: float                # adjusted_treatment - adjusted_control

    # Original (unadjusted) means for comparison
    original_mean_control: float
    original_mean_treatment: float
    original_effect: float

    # Variance information
    original_variance_control: float
    original_variance_treatment: float
    adjusted_variance_control: float
    adjusted_variance_treatment: float
    variance_reduction_pct: float         # How much CUPED helped (e.g., 0.45 = 45%)

    # CUPED parameters
    theta: float                          # Regression coefficient Cov(Y,X)/Var(X)
    correlation: float                    # Pearson correlation between Y and X

    # Sample sizes
    n_control: int
    n_treatment: int


def cuped_adjust(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
) -> CUPEDResult:
    """
    Apply CUPED variance reduction.

    Args:
        y_control: Post-experiment metric values for control group
        y_treatment: Post-experiment metric values for treatment group
        x_control: Pre-experiment covariate values for control group
        x_treatment: Pre-experiment covariate values for treatment group

    Returns:
        CUPEDResult with adjusted means and variance reduction info

    Steps:
        1. Pool all (Y, X) pairs to estimate theta = Cov(Y, X) / Var(X)
        2. Compute x_mean = mean of all X values (pooled)
        3. Y_adj_i = Y_i - theta * (X_i - x_mean) for each observation
        4. Compute adjusted means and variances per group
    """
    # Validate inputs
    if len(y_control) != len(x_control):
        raise ValueError("y_control and x_control must have same length")
    if len(y_treatment) != len(x_treatment):
        raise ValueError("y_treatment and x_treatment must have same length")
    if len(y_control) < 2 or len(y_treatment) < 2:
        raise ValueError("Need at least 2 observations per group")

    # Pool all data to estimate theta
    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])

    # Compute theta = Cov(Y, X) / Var(X)
    cov_yx = np.cov(y_all, x_all, ddof=1)[0, 1]
    var_x = np.var(x_all, ddof=1)

    if var_x < 1e-10:
        # No variance in covariate, CUPED can't help
        theta = 0.0
    else:
        theta = cov_yx / var_x

    # Compute correlation
    std_y = np.std(y_all, ddof=1)
    std_x = np.std(x_all, ddof=1)
    if std_y < 1e-10 or std_x < 1e-10:
        correlation = 0.0
    else:
        correlation = cov_yx / (std_y * std_x)

    # Pooled mean of X
    x_mean = np.mean(x_all)

    # Apply CUPED adjustment
    y_adj_control = y_control - theta * (x_control - x_mean)
    y_adj_treatment = y_treatment - theta * (x_treatment - x_mean)

    # Compute stats
    orig_mean_c = float(np.mean(y_control))
    orig_mean_t = float(np.mean(y_treatment))
    adj_mean_c = float(np.mean(y_adj_control))
    adj_mean_t = float(np.mean(y_adj_treatment))

    orig_var_c = float(np.var(y_control, ddof=1))
    orig_var_t = float(np.var(y_treatment, ddof=1))
    adj_var_c = float(np.var(y_adj_control, ddof=1))
    adj_var_t = float(np.var(y_adj_treatment, ddof=1))

    # Variance reduction percentage (average across groups)
    orig_var_avg = (orig_var_c + orig_var_t) / 2
    adj_var_avg = (adj_var_c + adj_var_t) / 2
    if orig_var_avg > 1e-10:
        var_reduction = 1 - (adj_var_avg / orig_var_avg)
    else:
        var_reduction = 0.0

    return CUPEDResult(
        adjusted_mean_control=adj_mean_c,
        adjusted_mean_treatment=adj_mean_t,
        adjusted_effect=adj_mean_t - adj_mean_c,
        original_mean_control=orig_mean_c,
        original_mean_treatment=orig_mean_t,
        original_effect=orig_mean_t - orig_mean_c,
        original_variance_control=orig_var_c,
        original_variance_treatment=orig_var_t,
        adjusted_variance_control=adj_var_c,
        adjusted_variance_treatment=adj_var_t,
        variance_reduction_pct=float(max(0, var_reduction)),
        theta=float(theta),
        correlation=float(correlation),
        n_control=len(y_control),
        n_treatment=len(y_treatment),
    )
