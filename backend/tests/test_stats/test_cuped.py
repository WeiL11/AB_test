"""
Comprehensive tests for the CUPED (Controlled-experiment Using Pre-Experiment Data) module.

Tests cover:
- Variance reduction with correlated covariates
- Treatment effect preservation (unbiasedness)
- Behavior with uncorrelated covariates
- High-correlation scenarios
- Sample size tracking
- Input validation (mismatched lengths, too few observations)
- Edge cases (zero-variance covariate, identical data)
"""
import numpy as np
import pytest

from app.stats.cuped import cuped_adjust, CUPEDResult


class TestCUPEDVarianceReduction:
    """Tests verifying that CUPED reduces variance when covariates are correlated."""

    def test_reduces_variance_with_correlated_covariate(self):
        """CUPED should reduce variance when covariate is correlated with outcome."""
        np.random.seed(42)
        n = 5000

        # Create correlated pre/post data for separate control/treatment groups
        pre_c = np.random.normal(50, 10, n)
        pre_t = np.random.normal(50, 10, n)
        noise_c = np.random.normal(0, 5, n)
        noise_t = np.random.normal(0, 5, n)
        y_c = pre_c + noise_c
        y_t = pre_t + noise_t + 2.0  # 2.0 treatment effect

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        assert result.variance_reduction_pct > 0.3, (
            f"Expected >30% variance reduction, got {result.variance_reduction_pct:.1%}"
        )
        assert result.adjusted_variance_control < result.original_variance_control
        assert result.adjusted_variance_treatment < result.original_variance_treatment

    def test_high_correlation_high_reduction(self):
        """With highly correlated covariate, variance reduction should be substantial."""
        np.random.seed(42)
        n = 5000
        x_c = np.random.normal(100, 10, n)
        x_t = np.random.normal(100, 10, n)
        # y = 0.9 * x + noise => correlation ~ 0.9
        y_c = 0.9 * x_c + np.random.normal(0, 5, n)
        y_t = 0.9 * x_t + np.random.normal(2, 5, n)

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.variance_reduction_pct > 0.5, (
            f"Expected >50% variance reduction with high correlation, "
            f"got {result.variance_reduction_pct:.1%}"
        )
        assert result.correlation > 0.7

    def test_no_correlation_no_reduction(self):
        """With uncorrelated covariate, variance reduction should be near zero."""
        np.random.seed(42)
        n = 5000
        y_c = np.random.normal(50, 10, n)
        y_t = np.random.normal(52, 10, n)
        x_c = np.random.normal(0, 1, n)  # Completely uncorrelated covariate
        x_t = np.random.normal(0, 1, n)

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.variance_reduction_pct < 0.05, (
            f"Expected <5% reduction with uncorrelated covariate, "
            f"got {result.variance_reduction_pct:.1%}"
        )
        assert abs(result.correlation) < 0.05


class TestCUPEDTreatmentEffect:
    """Tests verifying CUPED preserves the true treatment effect (unbiasedness)."""

    def test_preserves_treatment_effect(self):
        """CUPED adjustment should not bias the estimated treatment effect."""
        np.random.seed(42)
        n = 10000
        true_effect = 3.0
        pre_c = np.random.normal(100, 20, n)
        pre_t = np.random.normal(100, 20, n)
        y_c = pre_c + np.random.normal(0, 10, n)
        y_t = pre_t + np.random.normal(true_effect, 10, n)

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        # Adjusted effect should be close to the true effect
        assert result.adjusted_effect == pytest.approx(true_effect, abs=0.5)
        # Original effect should also be close
        assert result.original_effect == pytest.approx(true_effect, abs=0.5)

    def test_zero_treatment_effect(self):
        """When there is no treatment effect, CUPED should report near-zero effect."""
        np.random.seed(42)
        n = 10000
        pre_c = np.random.normal(100, 20, n)
        pre_t = np.random.normal(100, 20, n)
        y_c = pre_c + np.random.normal(0, 10, n)
        y_t = pre_t + np.random.normal(0, 10, n)

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        assert result.adjusted_effect == pytest.approx(0.0, abs=0.5)
        assert result.original_effect == pytest.approx(0.0, abs=0.5)

    def test_effect_direction_preserved(self):
        """Negative treatment effects should remain negative after adjustment."""
        np.random.seed(42)
        n = 5000
        true_effect = -5.0
        pre_c = np.random.normal(50, 10, n)
        pre_t = np.random.normal(50, 10, n)
        y_c = pre_c + np.random.normal(0, 5, n)
        y_t = pre_t + np.random.normal(true_effect, 5, n)

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        assert result.adjusted_effect < 0
        assert result.original_effect < 0


class TestCUPEDSampleSizes:
    """Tests for sample size tracking."""

    def test_sample_sizes_correct(self):
        """n_control and n_treatment should match input array lengths."""
        y_c = np.array([1.0, 2.0, 3.0])
        y_t = np.array([4.0, 5.0, 6.0, 7.0])
        x_c = np.array([1.0, 1.5, 2.0])
        x_t = np.array([3.0, 4.0, 5.0, 6.0])

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.n_control == 3
        assert result.n_treatment == 4

    def test_unequal_group_sizes(self):
        """CUPED should handle groups of different sizes."""
        np.random.seed(42)
        n_c, n_t = 1000, 3000
        pre_c = np.random.normal(50, 10, n_c)
        pre_t = np.random.normal(50, 10, n_t)
        y_c = pre_c + np.random.normal(0, 5, n_c)
        y_t = pre_t + np.random.normal(1, 5, n_t)

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        assert result.n_control == n_c
        assert result.n_treatment == n_t
        assert isinstance(result.adjusted_effect, float)


class TestCUPEDInputValidation:
    """Tests for input validation and error handling."""

    def test_mismatched_control_lengths_raise(self):
        """y_control and x_control must have the same length."""
        with pytest.raises(ValueError, match="y_control and x_control must have same length"):
            cuped_adjust(
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0]),
                np.array([1.0, 2.0]),  # length 2, but y_control has length 3
                np.array([3.0, 4.0]),
            )

    def test_mismatched_treatment_lengths_raise(self):
        """y_treatment and x_treatment must have the same length."""
        with pytest.raises(ValueError, match="y_treatment and x_treatment must have same length"):
            cuped_adjust(
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([3.0, 4.0]),  # length 2, but y_treatment has length 3
            )

    def test_too_few_observations_raise(self):
        """Need at least 2 observations per group."""
        with pytest.raises(ValueError, match="Need at least 2 observations"):
            cuped_adjust(
                np.array([1.0]),
                np.array([2.0, 3.0]),
                np.array([1.0]),
                np.array([2.0, 3.0]),
            )

    def test_too_few_treatment_observations_raise(self):
        """Need at least 2 observations in treatment group too."""
        with pytest.raises(ValueError, match="Need at least 2 observations"):
            cuped_adjust(
                np.array([1.0, 2.0]),
                np.array([3.0]),
                np.array([1.0, 2.0]),
                np.array([3.0]),
            )


class TestCUPEDResultFields:
    """Tests verifying the structure and consistency of CUPEDResult."""

    def test_result_is_cuped_result(self):
        """cuped_adjust should return a CUPEDResult instance."""
        y_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_t = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        x_c = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        x_t = np.array([2.5, 3.5, 4.5, 5.5, 6.5])

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert isinstance(result, CUPEDResult)

    def test_adjusted_effect_is_difference_of_adjusted_means(self):
        """adjusted_effect should equal adjusted_mean_treatment - adjusted_mean_control."""
        np.random.seed(42)
        n = 500
        pre_c = np.random.normal(50, 10, n)
        pre_t = np.random.normal(50, 10, n)
        y_c = pre_c + np.random.normal(0, 5, n)
        y_t = pre_t + np.random.normal(2, 5, n)

        result = cuped_adjust(y_c, y_t, pre_c, pre_t)

        expected_effect = result.adjusted_mean_treatment - result.adjusted_mean_control
        assert result.adjusted_effect == pytest.approx(expected_effect, rel=1e-10)

    def test_original_effect_is_difference_of_original_means(self):
        """original_effect should equal original_mean_treatment - original_mean_control."""
        np.random.seed(42)
        n = 500
        y_c = np.random.normal(50, 10, n)
        y_t = np.random.normal(52, 10, n)
        x_c = np.random.normal(50, 10, n)
        x_t = np.random.normal(50, 10, n)

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        expected_effect = result.original_mean_treatment - result.original_mean_control
        assert result.original_effect == pytest.approx(expected_effect, rel=1e-10)

    def test_variance_reduction_pct_non_negative(self):
        """variance_reduction_pct should be clamped to >= 0."""
        np.random.seed(42)
        n = 5000
        y_c = np.random.normal(50, 10, n)
        y_t = np.random.normal(52, 10, n)
        x_c = np.random.normal(0, 1, n)
        x_t = np.random.normal(0, 1, n)

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.variance_reduction_pct >= 0.0

    def test_theta_sign_matches_correlation(self):
        """Theta and correlation should have the same sign."""
        np.random.seed(42)
        n = 5000
        x_c = np.random.normal(100, 10, n)
        x_t = np.random.normal(100, 10, n)
        y_c = 0.8 * x_c + np.random.normal(0, 5, n)
        y_t = 0.8 * x_t + np.random.normal(2, 5, n)

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        # Both should be positive since y positively correlates with x
        assert result.theta > 0
        assert result.correlation > 0


class TestCUPEDEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_variance_covariate(self):
        """When covariate has zero variance, theta should be 0 and no adjustment occurs."""
        y_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_t = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        x_c = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # constant
        x_t = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.theta == 0.0
        assert result.correlation == 0.0
        # Adjusted means should equal original means
        assert result.adjusted_mean_control == pytest.approx(
            result.original_mean_control, abs=1e-10
        )
        assert result.adjusted_mean_treatment == pytest.approx(
            result.original_mean_treatment, abs=1e-10
        )

    def test_minimum_sample_size(self):
        """Should work with exactly 2 observations per group."""
        y_c = np.array([1.0, 3.0])
        y_t = np.array([4.0, 6.0])
        x_c = np.array([0.5, 2.5])
        x_t = np.array([3.5, 5.5])

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        assert result.n_control == 2
        assert result.n_treatment == 2
        assert isinstance(result.adjusted_effect, float)

    def test_result_is_frozen_dataclass(self):
        """CUPEDResult should be immutable (frozen dataclass)."""
        y_c = np.array([1.0, 2.0, 3.0])
        y_t = np.array([4.0, 5.0, 6.0])
        x_c = np.array([1.0, 1.5, 2.0])
        x_t = np.array([3.0, 4.0, 5.0])

        result = cuped_adjust(y_c, y_t, x_c, x_t)

        with pytest.raises(AttributeError):
            result.adjusted_effect = 999.0
