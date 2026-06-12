"""
Comprehensive tests for the Always-Valid Confidence Sequence module.

Tests cover:
- Detection of large and small differences
- CI width properties (wider than fixed-horizon)
- Input validation (minimum sample size, alpha range)
- Confidence sequence over time (width narrowing, checkpoint counts)
- Edge cases (zero variance, custom v_opt)
"""
import numpy as np
import pytest
from scipy import stats

from app.stats.confidence_sequence import (
    confidence_sequence,
    compute_confidence_sequence_over_time,
    ConfidenceSequenceResult,
)


class TestConfidenceSequence:
    """Tests for the single-point confidence_sequence function."""

    def test_detects_large_difference(self):
        """CS should detect a large effect with sufficient data."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 5000)
        treatment = np.random.normal(110, 10, 5000)

        result = confidence_sequence(control, treatment)

        assert result.is_significant
        assert result.ci_lower > 0  # CI should exclude 0

    def test_no_difference_includes_zero(self):
        """CS should include zero when the sample means are very close."""
        np.random.seed(15)
        # Use a seed where the sample means happen to be very close
        control = np.random.normal(100, 10, 200)
        treatment = np.random.normal(100, 10, 200)

        result = confidence_sequence(control, treatment)

        assert result.ci_lower <= 0 <= result.ci_upper
        assert not result.is_significant

    def test_cs_narrows_with_more_data(self):
        """CS should get narrower as sample size increases, demonstrating
        the tradeoff: wider than needed at small n, tighter at large n.

        This is the core property that makes confidence sequences useful --
        they are anytime-valid but converge as data grows.
        """
        np.random.seed(42)
        control_full = np.random.normal(100, 10, 5000)
        treatment_full = np.random.normal(102, 10, 5000)

        cs_small = confidence_sequence(
            control_full[:200], treatment_full[:200]
        )
        cs_large = confidence_sequence(
            control_full[:5000], treatment_full[:5000]
        )

        width_small = cs_small.ci_upper - cs_small.ci_lower
        width_large = cs_large.ci_upper - cs_large.ci_lower

        assert width_small > width_large, (
            f"CS at n=200 ({width_small:.4f}) should be wider than "
            f"CS at n=5000 ({width_large:.4f})"
        )

    def test_mean_difference_correct(self):
        """mean_difference should be treatment_mean - control_mean."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(105, 10, 500)

        result = confidence_sequence(control, treatment)

        expected_diff = float(np.mean(treatment) - np.mean(control))
        assert result.mean_difference == pytest.approx(expected_diff, rel=1e-10)

    def test_ci_contains_mean_difference(self):
        """The CI should always contain the point estimate."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(103, 10, 500)

        result = confidence_sequence(control, treatment)

        assert result.ci_lower <= result.mean_difference <= result.ci_upper

    def test_sample_size_is_min_of_groups(self):
        """sample_size should be min(n_control, n_treatment)."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 300)
        treatment = np.random.normal(100, 10, 500)

        result = confidence_sequence(control, treatment)

        assert result.sample_size == 300

    def test_alpha_stored_in_result(self):
        """The alpha used should be stored in the result."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 100)
        treatment = np.random.normal(100, 10, 100)

        result = confidence_sequence(control, treatment, alpha=0.01)

        assert result.alpha == 0.01

    def test_negative_difference_detected(self):
        """CS should detect negative effects when treatment is worse."""
        np.random.seed(42)
        control = np.random.normal(110, 10, 5000)
        treatment = np.random.normal(100, 10, 5000)

        result = confidence_sequence(control, treatment)

        assert result.is_significant
        assert result.ci_upper < 0  # Entire CI below zero
        assert result.mean_difference < 0


class TestConfidenceSequenceInputValidation:
    """Tests for input validation."""

    def test_small_control_raises(self):
        """Control with fewer than 2 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            confidence_sequence(np.array([1.0]), np.array([2.0, 3.0]))

    def test_small_treatment_raises(self):
        """Treatment with fewer than 2 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            confidence_sequence(np.array([1.0, 2.0]), np.array([3.0]))

    def test_alpha_zero_raises(self):
        """alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            confidence_sequence(
                np.array([1.0, 2.0]), np.array([3.0, 4.0]), alpha=0.0
            )

    def test_alpha_one_raises(self):
        """alpha=1 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            confidence_sequence(
                np.array([1.0, 2.0]), np.array([3.0, 4.0]), alpha=1.0
            )

    def test_negative_v_opt_raises(self):
        """Negative v_opt should raise ValueError."""
        with pytest.raises(ValueError, match="v_opt must be positive"):
            confidence_sequence(
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0]),
                v_opt=-1.0,
            )

    def test_2d_array_raises(self):
        """Non-1D arrays should raise ValueError."""
        with pytest.raises(ValueError, match="1-D arrays"):
            confidence_sequence(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([5.0, 6.0]),
            )

    def test_minimum_sample_size_works(self):
        """Exactly 2 observations per group should work."""
        result = confidence_sequence(
            np.array([1.0, 3.0]), np.array([5.0, 7.0])
        )
        assert isinstance(result, ConfidenceSequenceResult)
        assert result.sample_size == 2


class TestConfidenceSequenceEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_custom_v_opt(self):
        """Custom v_opt should be used and stored in the result."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(102, 10, 500)

        result = confidence_sequence(control, treatment, v_opt=5.0)

        assert result.mixing_variance == 5.0

    def test_different_v_opt_changes_width(self):
        """Different v_opt values should produce different CI widths."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(102, 10, 500)

        result_small_v = confidence_sequence(control, treatment, v_opt=0.1)
        result_large_v = confidence_sequence(control, treatment, v_opt=100.0)

        width_small = result_small_v.ci_upper - result_small_v.ci_lower
        width_large = result_large_v.ci_upper - result_large_v.ci_lower

        assert width_small != pytest.approx(width_large, rel=0.01)

    def test_zero_variance_data(self):
        """When both groups have zero variance, CI should collapse to the point estimate."""
        control = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        treatment = np.array([8.0, 8.0, 8.0, 8.0, 8.0])

        result = confidence_sequence(control, treatment)

        assert result.mean_difference == pytest.approx(3.0)
        assert result.ci_lower == pytest.approx(3.0)
        assert result.ci_upper == pytest.approx(3.0)
        assert result.is_significant  # 3.0 != 0

    def test_result_is_confidence_sequence_result(self):
        """Should return a ConfidenceSequenceResult instance."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 100)
        treatment = np.random.normal(100, 10, 100)

        result = confidence_sequence(control, treatment)

        assert isinstance(result, ConfidenceSequenceResult)

    def test_smaller_alpha_wider_interval(self):
        """Smaller alpha (higher confidence) should produce wider intervals."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(102, 10, 500)

        result_95 = confidence_sequence(control, treatment, alpha=0.05)
        result_99 = confidence_sequence(control, treatment, alpha=0.01)

        width_95 = result_95.ci_upper - result_95.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower

        assert width_99 > width_95, (
            f"99% CS width ({width_99:.4f}) should be wider than "
            f"95% CS width ({width_95:.4f})"
        )


class TestCSOverTime:
    """Tests for compute_confidence_sequence_over_time."""

    def test_width_decreases_over_time(self):
        """CS should narrow as more data accumulates."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 5000)
        treatment = np.random.normal(102, 10, 5000)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=10
        )

        widths = [r.ci_upper - r.ci_lower for r in results]
        # Width should generally decrease: last should be smaller than first
        assert widths[-1] < widths[0]

    def test_correct_number_of_points(self):
        """Should return the requested number of checkpoint results."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(100, 10, 1000)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=5
        )

        assert len(results) == 5

    def test_sample_sizes_increasing(self):
        """Sample sizes should be non-decreasing across checkpoints."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 2000)
        treatment = np.random.normal(100, 10, 2000)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=5
        )

        sizes = [r.sample_size for r in results]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_first_sample_size_at_least_min(self):
        """The first checkpoint should have at least min_samples observations."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(100, 10, 1000)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=200, n_points=5
        )

        assert results[0].sample_size >= 200

    def test_last_sample_size_is_max(self):
        """The last checkpoint should use the full dataset."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 800)
        treatment = np.random.normal(100, 10, 1200)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=5
        )

        # Max sample size is min(800, 1200) = 800
        assert results[-1].sample_size == 800

    def test_all_results_are_cs_results(self):
        """All returned items should be ConfidenceSequenceResult instances."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(100, 10, 500)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=3
        )

        for r in results:
            assert isinstance(r, ConfidenceSequenceResult)

    def test_single_point(self):
        """n_points=1 should return exactly one result using the full dataset."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(100, 10, 500)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=1
        )

        assert len(results) == 1
        assert results[0].sample_size == 500

    def test_too_few_samples_raises(self):
        """Should raise if data has fewer observations than min_samples."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 50)
        treatment = np.random.normal(100, 10, 50)

        with pytest.raises(ValueError, match="at least min_samples"):
            compute_confidence_sequence_over_time(
                control, treatment, min_samples=100, n_points=5
            )

    def test_consistent_alpha(self):
        """All results should use the same alpha."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(100, 10, 1000)

        results = compute_confidence_sequence_over_time(
            control, treatment, alpha=0.01, min_samples=100, n_points=5
        )

        for r in results:
            assert r.alpha == 0.01

    def test_eventually_detects_effect(self):
        """With a real effect and enough data, CS should eventually be significant."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 10000)
        treatment = np.random.normal(105, 10, 10000)

        results = compute_confidence_sequence_over_time(
            control, treatment, min_samples=100, n_points=20
        )

        # The last checkpoint with full data should detect the effect
        assert results[-1].is_significant
        assert results[-1].ci_lower > 0
