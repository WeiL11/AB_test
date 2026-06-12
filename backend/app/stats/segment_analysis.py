"""
Segment Analysis: Heterogeneous Treatment Effect Detection.

Tests whether the treatment effect differs across user segments
(e.g., country, platform, user cohort). This is critical because:

1. An overall positive effect can hide a negative effect in a key segment
2. An overall null result can mask a strong positive effect in a subgroup
3. Segment-level effects inform targeting and rollout decisions

Approach:
    1. Run the A/B test within each segment separately
    2. Test for interaction: does the treatment effect differ significantly
       across segments? (heterogeneity test)
    3. Apply multiple testing correction across segments

References:
    - Athey & Imbens (2016). Recursive partitioning for heterogeneous causal effects.
"""
import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass
from app.stats.frequentist import welch_t_test, z_test_proportions, FrequentistResult
from app.stats.multiple_testing import apply_correction, CorrectedResult


@dataclass(frozen=True)
class SegmentResult:
    """Result for a single segment."""
    segment_name: str
    segment_value: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    absolute_effect: float
    relative_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    corrected_p_value: float | None = None
    is_significant_corrected: bool | None = None


@dataclass(frozen=True)
class SegmentAnalysisResult:
    """Full segment analysis result."""
    segment_name: str                    # Which segment dimension (e.g., "country")
    segment_results: list[SegmentResult] # Per-segment results
    interaction_p_value: float           # P-value for heterogeneity test
    interaction_significant: bool        # Is the treatment effect significantly different across segments?
    correction_method: str               # Multiple testing correction used
    overall_effect: float                # Overall treatment effect (for reference)
    n_segments: int


def analyze_segments(
    control_values: dict[str, np.ndarray],
    treatment_values: dict[str, np.ndarray],
    segment_name: str = "segment",
    metric_type: str = "continuous",
    alpha: float = 0.05,
    correction_method: str = "holm",
) -> SegmentAnalysisResult:
    """
    Analyze treatment effect across segments.

    Args:
        control_values: Dict mapping segment_value -> array of observations
                       e.g., {"US": array([...]), "UK": array([...]), ...}
        treatment_values: Same structure for treatment group
        segment_name: Name of the segment dimension (e.g., "country")
        metric_type: "continuous" or "binomial"
        alpha: Significance level
        correction_method: Multiple testing correction method

    Returns:
        SegmentAnalysisResult with per-segment effects and interaction test
    """
    segments = sorted(set(control_values.keys()) & set(treatment_values.keys()))

    if len(segments) < 2:
        raise ValueError("Need at least 2 segments for segment analysis")

    # Run test within each segment
    segment_results_raw = []
    p_values = []
    effects = []

    all_control = []
    all_treatment = []

    for seg in segments:
        c = control_values[seg]
        t = treatment_values[seg]

        if len(c) < 2 or len(t) < 2:
            continue

        all_control.extend(c)
        all_treatment.extend(t)

        if metric_type == "binomial":
            result = z_test_proportions(
                int(np.sum(c)), len(c),
                int(np.sum(t)), len(t),
                alpha=alpha,
            )
        else:
            result = welch_t_test(c, t, alpha=alpha)

        segment_results_raw.append((seg, result))
        p_values.append(result.p_value)
        effects.append(result.absolute_effect)

    if len(segment_results_raw) < 2:
        raise ValueError("Need at least 2 segments with sufficient data")

    # Apply multiple testing correction
    corrected = apply_correction(p_values, method=correction_method, alpha=alpha)

    # Build segment results
    segment_results = []
    for i, (seg, result) in enumerate(segment_results_raw):
        segment_results.append(SegmentResult(
            segment_name=segment_name,
            segment_value=seg,
            n_control=result.sample_size_control,
            n_treatment=result.sample_size_treatment,
            mean_control=result.mean_control,
            mean_treatment=result.mean_treatment,
            absolute_effect=result.absolute_effect,
            relative_effect=result.relative_effect,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            p_value=result.p_value,
            is_significant=result.is_significant,
            corrected_p_value=corrected[i].adjusted_p_value,
            is_significant_corrected=corrected[i].is_significant,
        ))

    # Interaction test: is the treatment effect different across segments?
    # Use a test of heterogeneity (Cochran's Q-like approach)
    interaction_p = _test_heterogeneity(segment_results_raw)

    # Overall effect
    all_c = np.array(all_control)
    all_t = np.array(all_treatment)
    overall_effect = float(np.mean(all_t) - np.mean(all_c))

    return SegmentAnalysisResult(
        segment_name=segment_name,
        segment_results=segment_results,
        interaction_p_value=interaction_p,
        interaction_significant=interaction_p < alpha,
        correction_method=correction_method,
        overall_effect=overall_effect,
        n_segments=len(segment_results),
    )


def _test_heterogeneity(
    segment_results: list[tuple[str, FrequentistResult]],
) -> float:
    """
    Test whether treatment effects are homogeneous across segments.

    Uses Cochran's Q test:
    Q = sum(w_i * (effect_i - effect_weighted)^2)

    where w_i = 1/var_i (inverse variance weight)
    Q ~ chi2(k-1) under H0 of homogeneity

    Returns p-value for the heterogeneity test.
    """
    effects = []
    weights = []

    for _, result in segment_results:
        effect = result.absolute_effect
        se = result.standard_error
        if se > 1e-10:
            w = 1.0 / (se ** 2)
        else:
            w = 1e10  # Very large weight for near-zero SE
        effects.append(effect)
        weights.append(w)

    effects = np.array(effects)
    weights = np.array(weights)

    # Weighted mean effect
    weighted_mean = np.sum(weights * effects) / np.sum(weights)

    # Cochran's Q statistic
    Q = float(np.sum(weights * (effects - weighted_mean) ** 2))

    # Degrees of freedom
    df = len(effects) - 1

    # P-value from chi-squared distribution
    p_value = float(sp_stats.chi2.sf(Q, df))

    return p_value
