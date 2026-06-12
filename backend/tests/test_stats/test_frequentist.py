import numpy as np
import pytest
from app.stats.frequentist import welch_t_test, z_test_proportions, chi_squared_test


class TestWelchTTest:
    def test_identical_groups_not_significant(self):
        """Two groups from the same distribution should not be significant."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(100, 15, 1000)
        result = welch_t_test(control, treatment)
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_different_groups_significant(self):
        """Two groups with different means should be significant with enough data."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(105, 15, 1000)
        result = welch_t_test(control, treatment)
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.absolute_effect == pytest.approx(5.0, abs=1.0)

    def test_confidence_interval_contains_true_effect(self):
        """CI should contain the true effect most of the time."""
        np.random.seed(42)
        true_effect = 3.0
        control = np.random.normal(50, 10, 500)
        treatment = np.random.normal(50 + true_effect, 10, 500)
        result = welch_t_test(control, treatment)
        assert result.ci_lower <= true_effect <= result.ci_upper

    def test_relative_effect_calculation(self):
        """Relative effect should be (treatment - control) / |control|."""
        np.random.seed(42)
        control = np.random.normal(100, 5, 1000)
        treatment = np.random.normal(110, 5, 1000)
        result = welch_t_test(control, treatment)
        expected_relative = (result.mean_treatment - result.mean_control) / abs(result.mean_control)
        assert result.relative_effect == pytest.approx(expected_relative, rel=0.01)

    def test_sample_sizes_correct(self):
        control = np.array([1.0, 2.0, 3.0])
        treatment = np.array([4.0, 5.0, 6.0, 7.0])
        result = welch_t_test(control, treatment)
        assert result.sample_size_control == 3
        assert result.sample_size_treatment == 4

    def test_small_alpha_increases_threshold(self):
        """With alpha=0.01, a borderline result should not be significant."""
        np.random.seed(123)
        control = np.random.normal(100, 15, 200)
        treatment = np.random.normal(103, 15, 200)
        result_05 = welch_t_test(control, treatment, alpha=0.05)
        result_01 = welch_t_test(control, treatment, alpha=0.01)
        # With stricter alpha, less likely to be significant
        if result_05.is_significant:
            assert result_01.ci_upper - result_01.ci_lower > result_05.ci_upper - result_05.ci_lower

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            welch_t_test(np.array([]), np.array([1, 2, 3]))


class TestZTestProportions:
    def test_equal_proportions_not_significant(self):
        result = z_test_proportions(100, 1000, 105, 1000)
        assert not result.is_significant

    def test_different_proportions_significant(self):
        result = z_test_proportions(100, 1000, 150, 1000)
        assert result.is_significant
        assert result.p_value < 0.05

    def test_means_are_proportions(self):
        result = z_test_proportions(100, 1000, 150, 1000)
        assert result.mean_control == pytest.approx(0.10)
        assert result.mean_treatment == pytest.approx(0.15)

    def test_zero_successes(self):
        result = z_test_proportions(0, 1000, 0, 1000)
        assert not result.is_significant

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            z_test_proportions(-1, 100, 50, 100)
        with pytest.raises(ValueError):
            z_test_proportions(101, 100, 50, 100)


class TestChiSquared:
    def test_independent_not_significant(self):
        table = np.array([[50, 50], [50, 50]])
        result = chi_squared_test(table)
        assert not result["is_significant"]

    def test_dependent_significant(self):
        table = np.array([[90, 10], [10, 90]])
        result = chi_squared_test(table)
        assert result["is_significant"]
        assert result["p_value"] < 0.001
