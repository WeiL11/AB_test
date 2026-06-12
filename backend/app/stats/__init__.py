"""
Statistical engine for the A/B testing platform.

Re-exports all public symbols so callers can do::

    from app.stats import welch_t_test, FrequentistResult, required_sample_size
"""

from .base import FrequentistResult, PowerCurvePoint, PowerResult
from .confidence_sequence import (
    ConfidenceSequenceResult,
    compute_confidence_sequence_over_time,
    confidence_sequence,
)
from .frequentist import chi_squared_test, welch_t_test, z_test_proportions
from .power_analysis import (
    compute_power,
    estimate_duration,
    minimum_detectable_effect_func,
    power_curve,
    required_sample_size,
)
from .sequential import GroupSequentialTest, SequentialResult, sequential_z_test
from .bayesian import BayesianResult, beta_binomial_test, normal_test
from .bandit import (
    BanditResult,
    ThompsonSampling,
    UCB1,
    EpsilonGreedy,
    run_bandit_simulation,
    compute_cumulative_regret,
)
from .cuped import CUPEDResult, cuped_adjust
from .multiple_testing import (
    CorrectedResult,
    bonferroni,
    holm,
    benjamini_hochberg,
    apply_correction,
)
from .novelty_detection import NoveltyResult, detect_novelty_effect, compute_daily_effects
from .segment_analysis import SegmentResult, SegmentAnalysisResult, analyze_segments

__all__ = [
    # --- data classes -------------------------------------------------------
    "FrequentistResult",
    "PowerResult",
    "PowerCurvePoint",
    "SequentialResult",
    "ConfidenceSequenceResult",
    "BayesianResult",
    "BanditResult",
    "CUPEDResult",
    "CorrectedResult",
    "NoveltyResult",
    "SegmentResult",
    "SegmentAnalysisResult",
    # --- frequentist tests --------------------------------------------------
    "welch_t_test",
    "z_test_proportions",
    "chi_squared_test",
    # --- power analysis -----------------------------------------------------
    "required_sample_size",
    "minimum_detectable_effect_func",
    "compute_power",
    "power_curve",
    "estimate_duration",
    # --- sequential testing -------------------------------------------------
    "GroupSequentialTest",
    "sequential_z_test",
    # --- confidence sequences -----------------------------------------------
    "confidence_sequence",
    "compute_confidence_sequence_over_time",
    # --- bayesian -----------------------------------------------------------
    "beta_binomial_test",
    "normal_test",
    # --- bandits ------------------------------------------------------------
    "ThompsonSampling",
    "UCB1",
    "EpsilonGreedy",
    "run_bandit_simulation",
    "compute_cumulative_regret",
    # --- CUPED --------------------------------------------------------------
    "cuped_adjust",
    # --- multiple testing ---------------------------------------------------
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "apply_correction",
    # --- novelty detection --------------------------------------------------
    "detect_novelty_effect",
    "compute_daily_effects",
    # --- segment analysis ---------------------------------------------------
    "analyze_segments",
]
