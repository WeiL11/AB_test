"""
Comprehensive tests for the Novelty and Primacy Effect Detection module.

Tests cover:
- Detection of decaying effects (novelty)
- Detection of growing effects (primacy)
- Detection of stable effects
- Minimum day requirements
- Weighted regression with sample sizes
- compute_daily_effects helper function
- Input validation and edge cases
"""
import numpy as np
import pytest

from app.stats.novelty_detection import (
    detect_novelty_effect,
    compute_daily_effects,
    NoveltyResult,
)


class TestDetectNoveltyEffect:
    """Tests for the main novelty/primacy detection function."""

    def test_decaying_effect_detected_as_novelty(self):
        """A clearly decaying effect should be classified as 'novelty'."""
        np.random.seed(42)
        # Effect starts at 10, decays by ~0.6 per day over 14 days
        daily_effects = [10 - 0.6 * i + np.random.normal(0, 0.3) for i in range(14)]

        result = detect_novelty_effect(daily_effects)

        assert result.effect_type == "novelty"
        assert result.slope < 0
        assert result.is_significant

    def test_growing_effect_detected_as_primacy(self):
        """A growing effect should be classified as 'primacy'."""
        np.random.seed(42)
        daily_effects = [2 + 0.5 * i + np.random.normal(0, 0.3) for i in range(14)]

        result = detect_novelty_effect(daily_effects)

        assert result.effect_type == "primacy"
        assert result.slope > 0
        assert result.is_significant

    def test_stable_effect_detected_as_stable(self):
        """A constant effect with small noise should be classified as 'stable'."""
        np.random.seed(42)
        daily_effects = [5.0 + np.random.normal(0, 0.5) for _ in range(14)]

        result = detect_novelty_effect(daily_effects)

        assert result.effect_type == "stable"
        assert not result.is_significant

    def test_strongly_stable_effect(self):
        """A nearly constant effect with tiny noise should be stable."""
        np.random.seed(42)
        # Use very small noise so slope is not significant
        daily_effects = [5.0 + np.random.normal(0, 0.01) for _ in range(10)]

        result = detect_novelty_effect(daily_effects, min_days=5)

        assert result.effect_type == "stable"
        assert abs(result.slope) < 0.01
        assert not result.is_significant

    def test_n_days_returned_correctly(self):
        """n_days should match the number of daily effects provided."""
        daily_effects = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        result = detect_novelty_effect(daily_effects, min_days=5)
        assert result.n_days == 7

    def test_daily_effects_stored_in_result(self):
        """The daily_effects list should be stored in the result."""
        daily_effects = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_novelty_effect(daily_effects, min_days=5)
        assert result.daily_effects == daily_effects


class TestDetectNoveltyMinDays:
    """Tests for the min_days parameter."""

    def test_too_few_days_raises(self):
        """Fewer than min_days should raise ValueError."""
        with pytest.raises(ValueError, match="Need at least 5 days"):
            detect_novelty_effect([1.0, 2.0, 3.0], min_days=5)

    def test_custom_min_days(self):
        """Custom min_days should be respected."""
        result = detect_novelty_effect([1.0, 2.0, 3.0], min_days=3)
        assert result.n_days == 3

    def test_exact_min_days(self):
        """Exactly min_days should not raise."""
        result = detect_novelty_effect([1.0, 2.0, 3.0, 4.0, 5.0], min_days=5)
        assert result.n_days == 5


class TestDetectNoveltyWeighting:
    """Tests for weighted regression with sample sizes."""

    def test_weighted_regression_accepts_sample_sizes(self):
        """Sample sizes should be accepted and used as weights."""
        np.random.seed(42)
        effects = [5.0 + np.random.normal(0, 0.5) for _ in range(10)]
        sizes = [1000] * 10

        result = detect_novelty_effect(effects, daily_sample_sizes=sizes)

        assert result.n_days == 10
        assert result.daily_sample_sizes == sizes

    def test_mismatched_sample_sizes_raises(self):
        """daily_sample_sizes must match length of daily_effects."""
        with pytest.raises(ValueError, match="daily_sample_sizes must match"):
            detect_novelty_effect(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                daily_sample_sizes=[100, 200, 300],
                min_days=5,
            )

    def test_default_sample_sizes_are_ones(self):
        """When daily_sample_sizes is None, default weights should be [1]*n_days."""
        effects = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_novelty_effect(effects, min_days=5)
        assert result.daily_sample_sizes == [1] * 5

    def test_unequal_weights_influence_fit(self):
        """Larger weights on later days should pull the fit toward later data points."""
        np.random.seed(42)
        # Effect decays but last few days get much higher weight
        effects = [10.0, 8.0, 6.0, 4.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        # Heavily weight the stable later days
        sizes_uniform = [100] * 10
        sizes_weighted = [10, 10, 10, 10, 10, 10000, 10000, 10000, 10000, 10000]

        result_uniform = detect_novelty_effect(effects, daily_sample_sizes=sizes_uniform, min_days=5)
        result_weighted = detect_novelty_effect(effects, daily_sample_sizes=sizes_weighted, min_days=5)

        # With heavy weighting on stable later days, slope magnitude should be smaller
        assert abs(result_weighted.slope) < abs(result_uniform.slope)


class TestDetectNoveltyResultFields:
    """Tests for NoveltyResult field values and consistency."""

    def test_result_is_novelty_result(self):
        """detect_novelty_effect should return a NoveltyResult instance."""
        result = detect_novelty_effect([1.0, 2.0, 3.0, 4.0, 5.0], min_days=5)
        assert isinstance(result, NoveltyResult)

    def test_r_squared_non_negative(self):
        """R-squared should be clamped to >= 0."""
        np.random.seed(42)
        effects = [5.0 + np.random.normal(0, 0.5) for _ in range(10)]
        result = detect_novelty_effect(effects, min_days=5)
        assert result.r_squared >= 0.0

    def test_r_squared_high_for_strong_trend(self):
        """R-squared should be high when data follows a clear linear trend."""
        # Nearly perfect linear trend
        effects = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_novelty_effect(effects, min_days=5)
        assert result.r_squared > 0.9

    def test_p_value_in_valid_range(self):
        """p_value should be between 0 and 1."""
        np.random.seed(42)
        effects = [5.0 + np.random.normal(0, 0.5) for _ in range(10)]
        result = detect_novelty_effect(effects, min_days=5)
        assert 0.0 <= result.p_value <= 1.0

    def test_slope_se_positive(self):
        """Standard error of slope should be positive."""
        np.random.seed(42)
        effects = [5.0 + np.random.normal(0, 1.0) for _ in range(10)]
        result = detect_novelty_effect(effects, min_days=5)
        assert result.slope_se > 0

    def test_frozen_dataclass(self):
        """NoveltyResult should be immutable."""
        result = detect_novelty_effect([1.0, 2.0, 3.0, 4.0, 5.0], min_days=5)
        with pytest.raises(AttributeError):
            result.slope = 999.0

    def test_effect_type_values(self):
        """effect_type should be one of 'novelty', 'primacy', or 'stable'."""
        np.random.seed(42)
        effects = [5.0 + np.random.normal(0, 0.5) for _ in range(10)]
        result = detect_novelty_effect(effects, min_days=5)
        assert result.effect_type in ("novelty", "primacy", "stable")

    def test_custom_alpha(self):
        """A very strict alpha should make it harder to detect effects."""
        np.random.seed(42)
        # Moderate trend with some noise
        daily_effects = [5 - 0.2 * i + np.random.normal(0, 0.5) for i in range(10)]

        result_lenient = detect_novelty_effect(daily_effects, alpha=0.10, min_days=5)
        result_strict = detect_novelty_effect(daily_effects, alpha=0.001, min_days=5)

        # Strict alpha should be at least as conservative
        if result_strict.is_significant:
            assert result_lenient.is_significant


class TestComputeDailyEffects:
    """Tests for the compute_daily_effects helper function."""

    def test_basic_computation(self):
        """Daily effects should be treatment_mean - control_mean per day."""
        control = [[1, 2, 3], [4, 5, 6]]  # 2 days
        treatment = [[4, 5, 6], [7, 8, 9]]
        effects, sizes = compute_daily_effects(control, treatment)

        assert len(effects) == 2
        assert effects[0] == pytest.approx(3.0)  # mean([4,5,6]) - mean([1,2,3])
        assert effects[1] == pytest.approx(3.0)  # mean([7,8,9]) - mean([4,5,6])

    def test_daily_sample_sizes(self):
        """Daily sample sizes should be sum of control + treatment per day."""
        control = [[1, 2, 3], [4, 5]]  # day 0: 3 obs, day 1: 2 obs
        treatment = [[4, 5, 6, 7], [8, 9, 10]]  # day 0: 4 obs, day 1: 3 obs
        effects, sizes = compute_daily_effects(control, treatment)

        assert sizes[0] == 7  # 3 + 4
        assert sizes[1] == 5  # 2 + 3

    def test_skips_empty_days(self):
        """Days where either control or treatment is empty should be skipped."""
        control = [[1, 2], [], [3, 4]]
        treatment = [[3, 4], [], [5, 6]]
        effects, sizes = compute_daily_effects(control, treatment)

        assert len(effects) == 2  # Skipped the empty day
        assert len(sizes) == 2

    def test_skips_partially_empty_days(self):
        """Days where only one group is empty should also be skipped."""
        control = [[1, 2], [3, 4], []]
        treatment = [[3, 4], [], [5, 6]]
        effects, sizes = compute_daily_effects(control, treatment)

        assert len(effects) == 1  # Only day 0 has data in both groups

    def test_mismatched_days_raises(self):
        """Control and treatment must have the same number of days."""
        with pytest.raises(ValueError, match="same number of days"):
            compute_daily_effects([[1, 2]], [[3, 4], [5, 6]])

    def test_all_empty_days(self):
        """When all days are empty, return empty lists."""
        control = [[], [], []]
        treatment = [[], [], []]
        effects, sizes = compute_daily_effects(control, treatment)

        assert effects == []
        assert sizes == []

    def test_single_day(self):
        """Should work with a single day of data."""
        control = [[10, 20, 30]]
        treatment = [[15, 25, 35]]
        effects, sizes = compute_daily_effects(control, treatment)

        assert len(effects) == 1
        assert effects[0] == pytest.approx(5.0)  # mean([15,25,35]) - mean([10,20,30])
        assert sizes[0] == 6

    def test_unequal_daily_sizes(self):
        """Days can have different numbers of observations."""
        control = [[1], [1, 2, 3, 4, 5]]
        treatment = [[10, 20], [10]]
        effects, sizes = compute_daily_effects(control, treatment)

        assert len(effects) == 2
        assert effects[0] == pytest.approx(14.0)  # mean([10,20]) - mean([1])
        assert effects[1] == pytest.approx(7.0)   # mean([10]) - mean([1,2,3,4,5])

    def test_returns_tuple_of_lists(self):
        """Return type should be a tuple of two lists."""
        control = [[1, 2], [3, 4]]
        treatment = [[5, 6], [7, 8]]
        result = compute_daily_effects(control, treatment)

        assert isinstance(result, tuple)
        assert len(result) == 2
        effects, sizes = result
        assert isinstance(effects, list)
        assert isinstance(sizes, list)
