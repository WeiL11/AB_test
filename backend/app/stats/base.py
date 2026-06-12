"""
Base classes and result dataclasses for the A/B testing statistical engine.

This module defines the core data structures used throughout the stats package
to represent test results, power analysis outputs, and power curve data points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FrequentistResult:
    """Result of a frequentist hypothesis test (t-test or z-test).

    All effect sizes are expressed as (treatment - control), so a positive
    ``absolute_effect`` means the treatment performed better than control.

    Attributes:
        mean_control: Observed mean (or proportion) for the control group.
        mean_treatment: Observed mean (or proportion) for the treatment group.
        absolute_effect: Difference in means (treatment - control).
        relative_effect: Relative lift, computed as
            (treatment - control) / |control|.  ``None`` when the control
            mean is zero (division undefined).
        ci_lower: Lower bound of the confidence interval for the absolute
            difference.
        ci_upper: Upper bound of the confidence interval for the absolute
            difference.
        p_value: Two-sided p-value of the test.
        is_significant: Whether ``p_value`` is less than the chosen alpha.
        test_statistic: The test statistic (t or z value).
        degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite for
            t-tests; ``None`` for z-tests).
        standard_error: Standard error of the difference in means.
        sample_size_control: Number of observations in the control group.
        sample_size_treatment: Number of observations in the treatment group.
    """

    mean_control: float
    mean_treatment: float
    absolute_effect: float
    relative_effect: Optional[float]
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    test_statistic: float
    degrees_of_freedom: Optional[float]
    standard_error: float
    sample_size_control: int
    sample_size_treatment: int


@dataclass(frozen=True)
class PowerResult:
    """Result of a statistical power / sample-size calculation.

    Attributes:
        required_sample_size_per_variant: Observations needed in *each* arm
            (control and treatment) to achieve the desired power.
        total_sample_size: ``required_sample_size_per_variant * n_variants``
            (defaults to 2 variants).
        power: Statistical power (1 - beta), i.e. the probability of
            detecting the effect if it truly exists.
        alpha: Significance level used for the calculation.
        minimum_detectable_effect: The smallest absolute effect size the
            experiment is powered to detect.
        baseline_rate: The baseline metric value (e.g. control conversion
            rate or control mean) used in the calculation.
        estimated_days: Optional estimate of how many days the experiment
            needs to run, based on daily traffic.
    """

    required_sample_size_per_variant: int
    total_sample_size: int
    power: float
    alpha: float
    minimum_detectable_effect: float
    baseline_rate: float
    estimated_days: Optional[int] = field(default=None)


@dataclass(frozen=True)
class PowerCurvePoint:
    """A single point on a power curve.

    Attributes:
        effect_size: The absolute effect size (difference in means or
            proportions).
        power: The statistical power at this effect size.
    """

    effect_size: float
    power: float
