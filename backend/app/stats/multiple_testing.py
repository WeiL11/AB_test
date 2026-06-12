"""
Multiple Testing Correction Methods.

When testing many metrics simultaneously, the probability of at least one
false positive grows rapidly. With 20 metrics at alpha=0.05:
P(at least one false positive) = 1 - (1-0.05)^20 = 64.2%

Three correction methods, from most to least conservative:

1. Bonferroni: alpha_adj = alpha / m
   - Controls FWER (family-wise error rate)
   - Very conservative, especially with many tests

2. Holm-Bonferroni (step-down):
   - Also controls FWER
   - Uniformly more powerful than Bonferroni (strictly better)
   - Sort p-values, apply decreasing alpha thresholds

3. Benjamini-Hochberg (FDR):
   - Controls FDR (false discovery rate), not FWER
   - Much more powerful (less conservative) than Bonferroni/Holm
   - Best choice when testing many metrics (10+)

References:
- Bonferroni (1936). Teoria statistica delle classi.
- Holm (1979). A simple sequentially rejective multiple test procedure.
- Benjamini & Hochberg (1995). Controlling the false discovery rate.
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectedResult:
    """Result for a single hypothesis after correction."""
    original_p_value: float
    adjusted_p_value: float
    is_significant: bool
    method: str
    metric_name: str | None = None    # Optional label for the test


def bonferroni(
    p_values: list[float],
    alpha: float = 0.05,
    metric_names: list[str] | None = None,
) -> list[CorrectedResult]:
    """
    Bonferroni correction: multiply each p-value by m (number of tests).

    adjusted_p = min(p * m, 1.0)
    Reject if adjusted_p < alpha.

    Most conservative. Controls FWER (probability of ANY false positive).
    """
    m = len(p_values)
    if m == 0:
        return []

    results = []
    for i, p in enumerate(p_values):
        adj_p = min(p * m, 1.0)
        name = metric_names[i] if metric_names else None
        results.append(CorrectedResult(
            original_p_value=p,
            adjusted_p_value=adj_p,
            is_significant=adj_p < alpha,
            method="bonferroni",
            metric_name=name,
        ))
    return results


def holm(
    p_values: list[float],
    alpha: float = 0.05,
    metric_names: list[str] | None = None,
) -> list[CorrectedResult]:
    """
    Holm-Bonferroni step-down procedure.

    Algorithm:
    1. Sort p-values: p(1) <= p(2) <= ... <= p(m)
    2. For k = 1, 2, ..., m:
       - adjusted_p(k) = max(adjusted_p(k-1), p(k) * (m - k + 1))
    3. Reject if adjusted_p < alpha

    Uniformly more powerful than Bonferroni (always rejects at least as many).
    Still controls FWER.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]

    # Compute adjusted p-values (step-down)
    adjusted = np.zeros(m)
    for k in range(m):
        adjusted[k] = sorted_p[k] * (m - k)

    # Enforce monotonicity (each adjusted p >= previous)
    for k in range(1, m):
        adjusted[k] = max(adjusted[k], adjusted[k-1])

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result_adjusted = np.zeros(m)
    for k in range(m):
        result_adjusted[indices[k]] = adjusted[k]

    results = []
    for i in range(m):
        name = metric_names[i] if metric_names else None
        results.append(CorrectedResult(
            original_p_value=p_values[i],
            adjusted_p_value=float(result_adjusted[i]),
            is_significant=result_adjusted[i] < alpha,
            method="holm",
            metric_name=name,
        ))
    return results


def benjamini_hochberg(
    p_values: list[float],
    alpha: float = 0.05,
    metric_names: list[str] | None = None,
) -> list[CorrectedResult]:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate (FDR).

    Algorithm:
    1. Sort p-values: p(1) <= p(2) <= ... <= p(m)
    2. For k = m, m-1, ..., 1:
       - adjusted_p(k) = min(adjusted_p(k+1), p(k) * m / k)
    3. Reject if adjusted_p < alpha

    Controls FDR: expected proportion of false positives among rejections.
    Much less conservative than Bonferroni/Holm for many tests.
    """
    m = len(p_values)
    if m == 0:
        return []

    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]

    # Compute adjusted p-values (step-up, from the bottom)
    adjusted = np.zeros(m)
    adjusted[m-1] = sorted_p[m-1]  # last one is unchanged (or * m/m = same)

    for k in range(m-1, -1, -1):
        adjusted[k] = sorted_p[k] * m / (k + 1)

    # Enforce monotonicity (step-up: each adjusted p <= next)
    for k in range(m-2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k+1])

    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result_adjusted = np.zeros(m)
    for k in range(m):
        result_adjusted[indices[k]] = adjusted[k]

    results = []
    for i in range(m):
        name = metric_names[i] if metric_names else None
        results.append(CorrectedResult(
            original_p_value=p_values[i],
            adjusted_p_value=float(result_adjusted[i]),
            is_significant=result_adjusted[i] < alpha,
            method="benjamini_hochberg",
            metric_name=name,
        ))
    return results


def apply_correction(
    p_values: list[float],
    method: str = "holm",
    alpha: float = 0.05,
    metric_names: list[str] | None = None,
) -> list[CorrectedResult]:
    """
    Convenience function to apply any correction method.

    Args:
        p_values: List of raw p-values from individual tests
        method: "bonferroni" | "holm" | "benjamini_hochberg" | "fdr_bh"
        alpha: Significance level (default 0.05)
        metric_names: Optional labels for each test
    """
    if method == "bonferroni":
        return bonferroni(p_values, alpha, metric_names)
    elif method == "holm":
        return holm(p_values, alpha, metric_names)
    elif method in ("benjamini_hochberg", "fdr_bh", "bh"):
        return benjamini_hochberg(p_values, alpha, metric_names)
    else:
        raise ValueError(f"Unknown correction method: {method}. Use 'bonferroni', 'holm', or 'benjamini_hochberg'.")
