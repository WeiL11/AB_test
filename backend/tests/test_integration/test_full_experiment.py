"""
Integration test: Full experiment lifecycle without database.

Simulates a complete A/B test:
- Experiment: Blue vs Green checkout button
- Primary metric: conversion rate (binomial, 10% vs 12.5%)
- Secondary metric: revenue per user (continuous)
- 20,000 users per variant
- Runs all 9 statistical methods and verifies consistency
"""
import numpy as np
import pytest


class TestFullExperimentLifecycle:
    """Simulate a complete A/B test and verify all methods agree."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Generate realistic experiment data."""
        np.random.seed(42)
        self.n_per_group = 20000

        # Conversion rate: control=10%, treatment=12.5% (true positive)
        self.control_conversions = np.random.binomial(1, 0.10, self.n_per_group).astype(float)
        self.treatment_conversions = np.random.binomial(1, 0.125, self.n_per_group).astype(float)

        # Revenue: correlated with conversion, with pre-experiment covariate
        self.pre_control = np.random.normal(50, 15, self.n_per_group)
        self.pre_treatment = np.random.normal(50, 15, self.n_per_group)

        noise_c = np.random.normal(0, 8, self.n_per_group)
        noise_t = np.random.normal(0, 8, self.n_per_group)
        self.revenue_control = self.pre_control + noise_c
        self.revenue_treatment = self.pre_treatment + noise_t + 2.0  # $2 lift

    def test_step1_power_analysis(self):
        """Before running: verify we have enough power to detect the expected effect."""
        from app.stats.power_analysis import required_sample_size, compute_power

        # For 10% baseline, 2.5pp MDE
        result = required_sample_size(0.10, 0.025)
        assert result.required_sample_size_per_variant < self.n_per_group

        # With our actual sample size, power should be very high
        power = compute_power(self.n_per_group, 0.10, 0.025)
        assert power > 0.95

    def test_step2_frequentist_detects_conversion_lift(self):
        """Frequentist z-test should detect the conversion rate difference."""
        from app.stats.frequentist import z_test_proportions

        result = z_test_proportions(
            int(self.control_conversions.sum()), self.n_per_group,
            int(self.treatment_conversions.sum()), self.n_per_group,
        )
        assert result.is_significant
        assert result.absolute_effect > 0
        assert result.p_value < 0.01
        # Effect should be near 2.5pp
        assert 0.01 < result.absolute_effect < 0.05

    def test_step3_frequentist_detects_revenue_lift(self):
        """Welch's t-test should detect the revenue difference."""
        from app.stats.frequentist import welch_t_test

        result = welch_t_test(self.revenue_control, self.revenue_treatment)
        assert result.is_significant
        assert result.absolute_effect > 0
        # Effect should be near $2
        assert 1.0 < result.absolute_effect < 3.0

    def test_step4_bayesian_agrees_with_frequentist(self):
        """Bayesian analysis should also show treatment is better."""
        from app.stats.bayesian import beta_binomial_test, normal_test

        # Conversion
        conv_result = beta_binomial_test(
            int(self.control_conversions.sum()), self.n_per_group,
            int(self.treatment_conversions.sum()), self.n_per_group,
        )
        assert conv_result.prob_treatment_better > 0.99
        assert conv_result.recommendation == "ship"

        # Revenue
        rev_result = normal_test(self.revenue_control, self.revenue_treatment)
        assert rev_result.prob_treatment_better > 0.95

    def test_step5_sequential_allows_stopping(self):
        """Sequential test should allow stopping at full data."""
        from app.stats.sequential import sequential_z_test

        result = sequential_z_test(
            self.control_conversions, self.treatment_conversions,
            analysis_number=5, n_analyses=5,
        )
        assert result.can_reject
        assert result.recommendation == "stop_reject"

    def test_step6_confidence_sequence_significant(self):
        """Confidence sequence should exclude zero at full data."""
        from app.stats.confidence_sequence import confidence_sequence

        result = confidence_sequence(self.revenue_control, self.revenue_treatment)
        assert result.is_significant
        assert result.ci_lower > 0

    def test_step7_cuped_reduces_variance(self):
        """CUPED should reduce variance using pre-experiment revenue."""
        from app.stats.cuped import cuped_adjust

        result = cuped_adjust(
            self.revenue_control, self.revenue_treatment,
            self.pre_control, self.pre_treatment,
        )
        assert result.variance_reduction_pct > 0.2  # At least 20% reduction
        # Adjusted effect should still be positive and close to original
        assert result.adjusted_effect > 0
        assert abs(result.adjusted_effect - result.original_effect) < 1.0

    def test_step8_multiple_testing_correction(self):
        """Multiple testing should still find the real effect significant."""
        from app.stats.frequentist import z_test_proportions, welch_t_test
        from app.stats.multiple_testing import apply_correction

        # Get p-values from both metrics
        conv = z_test_proportions(
            int(self.control_conversions.sum()), self.n_per_group,
            int(self.treatment_conversions.sum()), self.n_per_group,
        )
        rev = welch_t_test(self.revenue_control, self.revenue_treatment)

        p_values = [conv.p_value, rev.p_value]

        # Even after correction, both should be significant
        corrected = apply_correction(p_values, method="holm")
        assert all(r.is_significant for r in corrected)

    def test_step9_novelty_detection_stable(self):
        """With constant true effect, no novelty should be detected."""
        from app.stats.novelty_detection import detect_novelty_effect

        np.random.seed(42)
        # Simulate stable daily effects (true effect = 0.025, with noise)
        daily_effects = [0.025 + np.random.normal(0, 0.005) for _ in range(14)]
        result = detect_novelty_effect(daily_effects)
        assert result.effect_type == "stable"

    def test_step10_bandit_finds_better_arm(self):
        """Thompson Sampling should identify the better arm."""
        from app.stats.bandit import run_bandit_simulation

        result = run_bandit_simulation([0.10, 0.125], algorithm="thompson", n_rounds=10000)
        assert result.best_arm_index == 1
        assert result.recommended_allocation[1] > result.recommended_allocation[0]

    def test_step11_segment_analysis(self):
        """Segment analysis should work across user segments."""
        from app.stats.segment_analysis import analyze_segments

        np.random.seed(42)
        # Split users into segments
        n = 5000
        control_segments = {
            "US": np.random.binomial(1, 0.10, n).astype(float),
            "UK": np.random.binomial(1, 0.10, n).astype(float),
            "DE": np.random.binomial(1, 0.10, n).astype(float),
        }
        treatment_segments = {
            "US": np.random.binomial(1, 0.125, n).astype(float),
            "UK": np.random.binomial(1, 0.13, n).astype(float),
            "DE": np.random.binomial(1, 0.12, n).astype(float),
        }

        result = analyze_segments(
            control_segments, treatment_segments,
            segment_name="country", metric_type="binomial",
        )
        assert result.n_segments == 3
        assert len(result.segment_results) == 3

    def test_all_methods_agree_on_direction(self):
        """All methods should agree that treatment is better than control."""
        from app.stats.frequentist import z_test_proportions
        from app.stats.bayesian import beta_binomial_test
        from app.stats.sequential import sequential_z_test

        freq = z_test_proportions(
            int(self.control_conversions.sum()), self.n_per_group,
            int(self.treatment_conversions.sum()), self.n_per_group,
        )
        bayes = beta_binomial_test(
            int(self.control_conversions.sum()), self.n_per_group,
            int(self.treatment_conversions.sum()), self.n_per_group,
        )
        seq = sequential_z_test(
            self.control_conversions, self.treatment_conversions,
            analysis_number=5, n_analyses=5,
        )

        # All agree treatment is better
        assert freq.absolute_effect > 0
        assert bayes.prob_treatment_better > 0.95
        assert seq.can_reject and seq.z_statistic > 0
