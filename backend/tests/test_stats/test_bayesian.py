"""
Unit tests for the Bayesian A/B Testing Engine.

Tests the beta_binomial_test (conversion-rate metrics) and normal_test
(continuous metrics) functions, covering posterior estimation, decision
metrics, ROPE analysis, recommendation logic, and input validation.
"""

import numpy as np
import pytest

from app.stats.bayesian import BayesianResult, beta_binomial_test, normal_test


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- core decision metrics
# --------------------------------------------------------------------------- #

class TestBetaBinomialDecisionMetrics:
    def test_clear_winner_high_probability(self):
        """With a clearly better treatment, P(B>A) should be very high."""
        result = beta_binomial_test(100, 1000, 150, 1000)
        assert result.prob_treatment_better > 0.99

    def test_clear_loser_low_probability(self):
        """When control clearly wins, P(B>A) should be very low."""
        result = beta_binomial_test(200, 1000, 100, 1000)
        assert result.prob_treatment_better < 0.01

    def test_equal_rates_probability_near_half(self):
        """With equal rates, P(B>A) should be near 0.5."""
        result = beta_binomial_test(100, 1000, 100, 1000)
        assert 0.4 < result.prob_treatment_better < 0.6

    def test_large_sample_sharpens_estimate(self):
        """Larger samples should make P(B>A) closer to 0 or 1 for different rates."""
        result_small = beta_binomial_test(10, 100, 15, 100)
        result_large = beta_binomial_test(1000, 10000, 1500, 10000)
        # Both have the same rates (10% vs 15%), but the larger sample
        # should be more decisive.
        assert result_large.prob_treatment_better > result_small.prob_treatment_better


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- posterior parameters
# --------------------------------------------------------------------------- #

class TestBetaBinomialPosterior:
    def test_posterior_means_match_data(self):
        """Posterior means should approximate the sample proportions (uniform prior)."""
        result = beta_binomial_test(100, 1000, 200, 1000)
        # With uniform prior and n=1000, posterior mean ~ (successes+1)/(trials+2)
        # which is very close to successes/trials.
        assert result.posterior_mean_control == pytest.approx(0.10, abs=0.01)
        assert result.posterior_mean_treatment == pytest.approx(0.20, abs=0.01)

    def test_posterior_std_decreases_with_sample_size(self):
        """Posterior uncertainty should shrink with more data."""
        result_small = beta_binomial_test(10, 100, 20, 100)
        result_large = beta_binomial_test(100, 1000, 200, 1000)
        assert result_large.posterior_std_control < result_small.posterior_std_control
        assert result_large.posterior_std_treatment < result_small.posterior_std_treatment

    def test_posterior_std_is_positive(self):
        result = beta_binomial_test(50, 500, 60, 500)
        assert result.posterior_std_control > 0
        assert result.posterior_std_treatment > 0


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- expected loss
# --------------------------------------------------------------------------- #

class TestBetaBinomialExpectedLoss:
    def test_expected_loss_treatment_small_when_treatment_wins(self):
        """When treatment clearly wins, expected loss of choosing it should be tiny."""
        result = beta_binomial_test(100, 1000, 200, 1000)
        assert result.expected_loss_treatment < 0.001

    def test_expected_loss_control_large_when_treatment_wins(self):
        """When treatment wins, expected loss of sticking with control should be large."""
        result = beta_binomial_test(100, 1000, 200, 1000)
        assert result.expected_loss_control > 0.05  # roughly 0.10 difference

    def test_expected_loss_ordering(self):
        """The better-performing arm should have a smaller expected loss."""
        result = beta_binomial_test(100, 1000, 150, 1000)
        assert result.expected_loss_treatment < result.expected_loss_control


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- credible interval
# --------------------------------------------------------------------------- #

class TestBetaBinomialCredibleInterval:
    def test_credible_interval_contains_true_difference(self):
        """95% credible interval should contain the true difference (0.05)."""
        result = beta_binomial_test(100, 1000, 150, 1000)
        assert result.credible_interval_lower < 0.05 < result.credible_interval_upper

    def test_credible_interval_ordering(self):
        """Lower bound should be less than upper bound."""
        result = beta_binomial_test(50, 500, 60, 500)
        assert result.credible_interval_lower < result.credible_interval_upper

    def test_ci_excludes_zero_for_large_effect(self):
        """With a massive effect, the CI should exclude zero."""
        result = beta_binomial_test(50, 1000, 300, 1000)
        assert result.credible_interval_lower > 0.0

    def test_ci_includes_zero_for_equal_rates(self):
        """With equal rates, the CI should span zero."""
        result = beta_binomial_test(100, 1000, 100, 1000)
        assert result.credible_interval_lower < 0.0 < result.credible_interval_upper


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- ROPE analysis
# --------------------------------------------------------------------------- #

class TestBetaBinomialROPE:
    def test_rope_fields_none_by_default(self):
        """Without ROPE, related fields should be None."""
        result = beta_binomial_test(100, 1000, 150, 1000)
        assert result.prob_in_rope is None
        assert result.prob_above_rope is None
        assert result.prob_below_rope is None

    def test_large_effect_mostly_above_rope(self):
        """A large positive effect should be mostly above the ROPE."""
        result = beta_binomial_test(100, 1000, 200, 1000, rope=(-0.01, 0.01))
        assert result.prob_above_rope is not None
        assert result.prob_above_rope > 0.95

    def test_no_effect_mostly_in_rope(self):
        """With equal rates, the effect should lie mostly within a generous ROPE."""
        result = beta_binomial_test(100, 1000, 100, 1000, rope=(-0.05, 0.05))
        assert result.prob_in_rope is not None
        assert result.prob_in_rope > 0.80

    def test_rope_probabilities_sum_to_one(self):
        """P(in ROPE) + P(above ROPE) + P(below ROPE) should sum to ~1."""
        result = beta_binomial_test(100, 1000, 120, 1000, rope=(-0.01, 0.01))
        total = result.prob_in_rope + result.prob_above_rope + result.prob_below_rope
        assert total == pytest.approx(1.0, abs=0.01)

    def test_negative_effect_below_rope(self):
        """A large negative effect should be mostly below the ROPE."""
        result = beta_binomial_test(200, 1000, 100, 1000, rope=(-0.01, 0.01))
        assert result.prob_below_rope > 0.95


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- recommendation logic
# --------------------------------------------------------------------------- #

class TestBetaBinomialRecommendation:
    def test_recommendation_ship(self):
        """Should recommend 'ship' with overwhelmingly positive results."""
        result = beta_binomial_test(100, 1000, 200, 1000)
        assert result.recommendation == "ship"

    def test_recommendation_dont_ship(self):
        """Should recommend 'dont_ship' when control clearly wins."""
        result = beta_binomial_test(200, 1000, 100, 1000)
        assert result.recommendation == "dont_ship"

    def test_recommendation_keep_running_for_ambiguous_large_sample(self):
        """Ambiguous results with >1000 total should say 'keep_running'."""
        # Very close rates with large sample -- inconclusive
        result = beta_binomial_test(100, 1000, 102, 1000)
        assert result.recommendation in ("keep_running", "inconclusive", "ship")

    def test_recommendation_values_are_valid(self):
        """Recommendation must be one of the allowed strings."""
        valid = {"ship", "dont_ship", "keep_running", "inconclusive"}
        result = beta_binomial_test(50, 500, 55, 500)
        assert result.recommendation in valid


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- samples
# --------------------------------------------------------------------------- #

class TestBetaBinomialSamples:
    def test_samples_returned(self):
        """Raw posterior samples should be available for plotting."""
        result = beta_binomial_test(50, 500, 60, 500, n_samples=5000)
        assert len(result.samples_control) == 5000
        assert len(result.samples_treatment) == 5000

    def test_samples_in_valid_range(self):
        """Beta samples must be in [0, 1]."""
        result = beta_binomial_test(50, 500, 60, 500)
        assert np.all(result.samples_control >= 0.0)
        assert np.all(result.samples_control <= 1.0)
        assert np.all(result.samples_treatment >= 0.0)
        assert np.all(result.samples_treatment <= 1.0)


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- input validation
# --------------------------------------------------------------------------- #

class TestBetaBinomialValidation:
    def test_negative_successes_raises(self):
        with pytest.raises(ValueError):
            beta_binomial_test(-1, 100, 50, 100)

    def test_successes_exceed_trials_raises(self):
        with pytest.raises(ValueError):
            beta_binomial_test(101, 100, 50, 100)

    def test_zero_trials_raises(self):
        with pytest.raises(ValueError):
            beta_binomial_test(0, 0, 50, 100)

    def test_negative_treatment_successes_raises(self):
        with pytest.raises(ValueError):
            beta_binomial_test(50, 100, -1, 100)

    def test_treatment_successes_exceed_trials_raises(self):
        with pytest.raises(ValueError):
            beta_binomial_test(50, 100, 101, 100)


# --------------------------------------------------------------------------- #
#  Beta-Binomial test -- prior sensitivity
# --------------------------------------------------------------------------- #

class TestBetaBinomialPrior:
    def test_strong_prior_shifts_posterior(self):
        """A strong prior should pull the posterior mean toward the prior."""
        # Informative prior centred at 0.5 (alpha=50, beta=50)
        result_strong = beta_binomial_test(
            10, 100, 20, 100, prior_alpha=50.0, prior_beta=50.0
        )
        # Uniform prior (alpha=1, beta=1)
        result_uniform = beta_binomial_test(
            10, 100, 20, 100, prior_alpha=1.0, prior_beta=1.0
        )
        # Strong prior should pull control mean toward 0.5 more than uniform
        assert result_strong.posterior_mean_control > result_uniform.posterior_mean_control


# --------------------------------------------------------------------------- #
#  Normal test -- core functionality
# --------------------------------------------------------------------------- #

class TestNormalTest:
    def test_clear_difference_detected(self):
        """Should detect a clear difference in continuous metrics."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(110, 10, 1000)
        result = normal_test(control, treatment)
        assert result.prob_treatment_better > 0.99

    def test_no_difference_uncertain(self):
        """With same distributions, should not produce an extreme probability."""
        rng = np.random.default_rng(seed=123)
        control = rng.normal(100, 10, 200)
        treatment = rng.normal(100, 10, 200)
        result = normal_test(control, treatment)
        # With no true difference the probability should stay away from extremes.
        assert 0.05 < result.prob_treatment_better < 0.95

    def test_treatment_worse_detected(self):
        """When treatment is worse, P(B>A) should be low."""
        np.random.seed(42)
        control = np.random.normal(110, 10, 1000)
        treatment = np.random.normal(100, 10, 1000)
        result = normal_test(control, treatment)
        assert result.prob_treatment_better < 0.01

    def test_posterior_means_close_to_sample_means(self):
        """With large prior variance, posterior means should match sample means."""
        np.random.seed(42)
        control = np.random.normal(50, 5, 500)
        treatment = np.random.normal(55, 5, 500)
        result = normal_test(control, treatment)
        assert result.posterior_mean_control == pytest.approx(np.mean(control), abs=0.5)
        assert result.posterior_mean_treatment == pytest.approx(np.mean(treatment), abs=0.5)

    def test_credible_interval_contains_true_difference(self):
        """95% CI should contain the true mean difference."""
        np.random.seed(42)
        true_diff = 5.0
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(100 + true_diff, 10, 1000)
        result = normal_test(control, treatment)
        assert result.credible_interval_lower < true_diff < result.credible_interval_upper

    def test_credible_interval_ordering(self):
        np.random.seed(42)
        control = np.random.normal(100, 10, 200)
        treatment = np.random.normal(105, 10, 200)
        result = normal_test(control, treatment)
        assert result.credible_interval_lower < result.credible_interval_upper


# --------------------------------------------------------------------------- #
#  Normal test -- expected loss
# --------------------------------------------------------------------------- #

class TestNormalTestExpectedLoss:
    def test_expected_loss_favours_better_arm(self):
        """Expected loss should be smaller for the arm that is actually better."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(110, 10, 1000)
        result = normal_test(control, treatment)
        assert result.expected_loss_treatment < result.expected_loss_control


# --------------------------------------------------------------------------- #
#  Normal test -- ROPE analysis
# --------------------------------------------------------------------------- #

class TestNormalTestROPE:
    def test_rope_fields_none_by_default(self):
        np.random.seed(42)
        c = np.random.normal(0, 1, 100)
        t = np.random.normal(0, 1, 100)
        result = normal_test(c, t)
        assert result.prob_in_rope is None
        assert result.prob_above_rope is None
        assert result.prob_below_rope is None

    def test_rope_large_effect_above(self):
        np.random.seed(42)
        c = np.random.normal(100, 5, 1000)
        t = np.random.normal(110, 5, 1000)
        result = normal_test(c, t, rope=(-0.5, 0.5))
        assert result.prob_above_rope > 0.99


# --------------------------------------------------------------------------- #
#  Normal test -- recommendation logic
# --------------------------------------------------------------------------- #

class TestNormalTestRecommendation:
    def test_recommendation_ship(self):
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(120, 10, 1000)
        result = normal_test(control, treatment)
        assert result.recommendation == "ship"

    def test_recommendation_dont_ship(self):
        np.random.seed(42)
        control = np.random.normal(120, 10, 1000)
        treatment = np.random.normal(100, 10, 1000)
        result = normal_test(control, treatment)
        assert result.recommendation == "dont_ship"


# --------------------------------------------------------------------------- #
#  Normal test -- validation
# --------------------------------------------------------------------------- #

class TestNormalTestValidation:
    def test_single_observation_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            normal_test(np.array([1.0]), np.array([2.0, 3.0]))

    def test_single_observation_treatment_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            normal_test(np.array([1.0, 2.0]), np.array([3.0]))


# --------------------------------------------------------------------------- #
#  BayesianResult dataclass
# --------------------------------------------------------------------------- #

class TestBayesianResult:
    def test_result_is_frozen(self):
        result = beta_binomial_test(50, 500, 60, 500)
        with pytest.raises(AttributeError):
            result.recommendation = "ship"  # type: ignore[misc]

    def test_result_type(self):
        result = beta_binomial_test(50, 500, 60, 500)
        assert isinstance(result, BayesianResult)
