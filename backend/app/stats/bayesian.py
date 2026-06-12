"""
Bayesian A/B Testing Engine.

Instead of p-values, directly answers:
- "What is the probability that B is better than A?" -> P(B > A)
- "If I ship B and I'm wrong, how much do I lose?" -> Expected Loss
- "Is the difference practically meaningful?" -> ROPE analysis

Implements:
- Beta-Binomial model for conversion rate metrics
- Normal-Normal model for continuous metrics
- Monte Carlo estimation of P(B>A) and expected loss
- Region of Practical Equivalence (ROPE) analysis

References:
- Kruschke (2013). Bayesian estimation supersedes the t-test.
"""
import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BayesianResult:
    """Result of Bayesian A/B test."""
    # Posterior parameters
    posterior_mean_control: float
    posterior_mean_treatment: float
    posterior_std_control: float
    posterior_std_treatment: float

    # Decision metrics
    prob_treatment_better: float      # P(B > A) via Monte Carlo
    expected_loss_treatment: float    # E[max(0, A - B)] -- risk of choosing B
    expected_loss_control: float      # E[max(0, B - A)] -- risk of choosing A

    # Credible interval for the difference (treatment - control)
    credible_interval_lower: float    # 2.5th percentile (95% HDI)
    credible_interval_upper: float    # 97.5th percentile

    # ROPE analysis (if ROPE specified)
    prob_in_rope: float | None = None           # P(effect in ROPE)
    prob_above_rope: float | None = None        # P(effect > ROPE upper)
    prob_below_rope: float | None = None        # P(effect < ROPE lower)

    # Raw samples for plotting
    samples_control: np.ndarray = field(default_factory=lambda: np.array([]))
    samples_treatment: np.ndarray = field(default_factory=lambda: np.array([]))

    # Recommendation
    recommendation: str = "inconclusive"  # "ship" | "dont_ship" | "keep_running" | "inconclusive"


def beta_binomial_test(
    successes_control: int,
    trials_control: int,
    successes_treatment: int,
    trials_treatment: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    rope: tuple[float, float] | None = None,
    decision_threshold: float = 0.95,
    loss_threshold: float = 0.001,
) -> BayesianResult:
    """
    Bayesian A/B test for conversion rates using Beta-Binomial model.

    Model:
        Prior: theta ~ Beta(prior_alpha, prior_beta)
        Likelihood: X ~ Binomial(n, theta)
        Posterior: theta | data ~ Beta(prior_alpha + successes, prior_beta + failures)

    Decision rules:
        Ship if: P(B > A) > decision_threshold AND expected_loss < loss_threshold
        Don't ship if: P(A > B) > decision_threshold
        Otherwise: keep running

    Args:
        successes_control: Number of conversions in control
        trials_control: Total users in control
        successes_treatment: Number of conversions in treatment
        trials_treatment: Total users in treatment
        prior_alpha: Beta prior alpha (default 1 = uniform)
        prior_beta: Beta prior beta (default 1 = uniform)
        n_samples: Number of Monte Carlo samples for estimation
        rope: Region of Practical Equivalence as (lower, upper).
              e.g., (-0.001, 0.001) means effects < 0.1% are practically zero.
        decision_threshold: P(B>A) threshold for shipping (default 0.95)
        loss_threshold: Maximum acceptable expected loss (default 0.001 = 0.1%)
    """
    # Validate inputs
    if successes_control < 0 or trials_control <= 0:
        raise ValueError("Invalid control data")
    if successes_treatment < 0 or trials_treatment <= 0:
        raise ValueError("Invalid treatment data")
    if successes_control > trials_control or successes_treatment > trials_treatment:
        raise ValueError("Successes cannot exceed trials")

    # Compute posterior parameters
    alpha_c = prior_alpha + successes_control
    beta_c = prior_beta + (trials_control - successes_control)
    alpha_t = prior_alpha + successes_treatment
    beta_t = prior_beta + (trials_treatment - successes_treatment)

    # Sample from posteriors
    rng = np.random.default_rng(seed=42)
    samples_c = rng.beta(alpha_c, beta_c, size=n_samples)
    samples_t = rng.beta(alpha_t, beta_t, size=n_samples)

    # Compute decision metrics
    diff = samples_t - samples_c
    prob_better = np.mean(diff > 0)
    expected_loss_t = np.mean(np.maximum(0, samples_c - samples_t))  # loss if we choose treatment
    expected_loss_c = np.mean(np.maximum(0, samples_t - samples_c))  # loss if we choose control

    # Credible interval for the difference
    ci_lower = float(np.percentile(diff, 2.5))
    ci_upper = float(np.percentile(diff, 97.5))

    # ROPE analysis
    prob_in_rope = None
    prob_above = None
    prob_below = None
    if rope is not None:
        prob_in_rope = float(np.mean((diff >= rope[0]) & (diff <= rope[1])))
        prob_above = float(np.mean(diff > rope[1]))
        prob_below = float(np.mean(diff < rope[0]))

    # Recommendation
    if prob_better > decision_threshold and expected_loss_t < loss_threshold:
        recommendation = "ship"
    elif (1 - prob_better) > decision_threshold:
        recommendation = "dont_ship"
    elif trials_control + trials_treatment > 1000:
        recommendation = "keep_running"
    else:
        recommendation = "inconclusive"

    # Posterior stats
    posterior_mean_c = alpha_c / (alpha_c + beta_c)
    posterior_mean_t = alpha_t / (alpha_t + beta_t)
    posterior_std_c = float(np.sqrt(alpha_c * beta_c / ((alpha_c + beta_c)**2 * (alpha_c + beta_c + 1))))
    posterior_std_t = float(np.sqrt(alpha_t * beta_t / ((alpha_t + beta_t)**2 * (alpha_t + beta_t + 1))))

    return BayesianResult(
        posterior_mean_control=posterior_mean_c,
        posterior_mean_treatment=posterior_mean_t,
        posterior_std_control=posterior_std_c,
        posterior_std_treatment=posterior_std_t,
        prob_treatment_better=float(prob_better),
        expected_loss_treatment=float(expected_loss_t),
        expected_loss_control=float(expected_loss_c),
        credible_interval_lower=ci_lower,
        credible_interval_upper=ci_upper,
        prob_in_rope=prob_in_rope,
        prob_above_rope=prob_above,
        prob_below_rope=prob_below,
        samples_control=samples_c,
        samples_treatment=samples_t,
        recommendation=recommendation,
    )


def normal_test(
    control: np.ndarray,
    treatment: np.ndarray,
    prior_mean: float = 0.0,
    prior_variance: float = 1e6,
    n_samples: int = 100_000,
    rope: tuple[float, float] | None = None,
    decision_threshold: float = 0.95,
    loss_threshold: float = 0.01,
) -> BayesianResult:
    """
    Bayesian A/B test for continuous metrics using Normal model.

    Uses Normal-Normal conjugate update with known variance approximation:
    - Estimate variance from data (pooled)
    - Use Normal prior on the mean with large variance (vague prior)
    - Posterior mean is approximately the sample mean (with large prior variance)

    For practical purposes with reasonable sample sizes, the posterior
    is well approximated by Normal(sample_mean, sample_variance/n).
    """
    # Validate
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Need at least 2 observations per group")

    # Compute sample statistics
    mean_c, var_c, n_c = float(np.mean(control)), float(np.var(control, ddof=1)), len(control)
    mean_t, var_t, n_t = float(np.mean(treatment)), float(np.var(treatment, ddof=1)), len(treatment)

    # Posterior (approximately Normal with large prior variance)
    post_var_c = var_c / n_c
    post_var_t = var_t / n_t

    # Sample from posteriors
    rng = np.random.default_rng(seed=42)
    samples_c = rng.normal(mean_c, np.sqrt(post_var_c), size=n_samples)
    samples_t = rng.normal(mean_t, np.sqrt(post_var_t), size=n_samples)

    # Decision metrics
    diff = samples_t - samples_c
    prob_better = float(np.mean(diff > 0))
    expected_loss_t = float(np.mean(np.maximum(0, samples_c - samples_t)))
    expected_loss_c = float(np.mean(np.maximum(0, samples_t - samples_c)))

    ci_lower = float(np.percentile(diff, 2.5))
    ci_upper = float(np.percentile(diff, 97.5))

    # ROPE
    prob_in_rope = prob_above = prob_below = None
    if rope:
        prob_in_rope = float(np.mean((diff >= rope[0]) & (diff <= rope[1])))
        prob_above = float(np.mean(diff > rope[1]))
        prob_below = float(np.mean(diff < rope[0]))

    # Recommendation
    if prob_better > decision_threshold and expected_loss_t < loss_threshold:
        recommendation = "ship"
    elif (1 - prob_better) > decision_threshold:
        recommendation = "dont_ship"
    else:
        recommendation = "keep_running"

    return BayesianResult(
        posterior_mean_control=mean_c,
        posterior_mean_treatment=mean_t,
        posterior_std_control=float(np.sqrt(post_var_c)),
        posterior_std_treatment=float(np.sqrt(post_var_t)),
        prob_treatment_better=prob_better,
        expected_loss_treatment=expected_loss_t,
        expected_loss_control=expected_loss_c,
        credible_interval_lower=ci_lower,
        credible_interval_upper=ci_upper,
        prob_in_rope=prob_in_rope,
        prob_above_rope=prob_above,
        prob_below_rope=prob_below,
        samples_control=samples_c,
        samples_treatment=samples_t,
        recommendation=recommendation,
    )
