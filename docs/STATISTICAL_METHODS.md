# Statistical Methods Reference

## Experimentor A/B Testing Platform

> Mathematical foundations for every statistical method implemented in the platform.
> This document serves as both a design reference for contributors and a demonstration
> of the rigorous statistical thinking behind each feature.

---

## Table of Contents

1. [Welch's T-Test (Frequentist)](#1-welchs-t-test-frequentist)
2. [Z-Test for Proportions](#2-z-test-for-proportions)
3. [Group Sequential Testing (O'Brien-Fleming)](#3-group-sequential-testing-obrien-fleming)
4. [Confidence Sequences (Always-Valid Inference)](#4-confidence-sequences-always-valid-inference)
5. [Bayesian A/B Testing](#5-bayesian-ab-testing)
6. [Multi-Armed Bandits](#6-multi-armed-bandits)
7. [CUPED (Variance Reduction)](#7-cuped-variance-reduction)
8. [Multiple Testing Correction](#8-multiple-testing-correction)
9. [Power Analysis & Sample Size](#9-power-analysis--sample-size)
10. [Novelty & Primacy Detection](#10-novelty--primacy-detection)
11. [Segment Analysis](#11-segment-analysis)
12. [References](#references)

---

## 1. Welch's T-Test (Frequentist)

### Problem

Compare the means of a continuous metric (e.g., revenue per user, session duration) between
a treatment group and a control group when the two groups may have **unequal variances** and
**unequal sample sizes**.

### Why Welch's and Not Student's T-Test

Student's t-test assumes equal variances across groups (`sigma_t^2 = sigma_c^2`). In A/B
testing this assumption almost never holds: the treatment itself is designed to shift the
distribution. Welch's t-test relaxes this assumption at virtually no cost in power when
variances happen to be equal, and avoids inflated Type I error when they are not.

### Mathematical Formulation

**Test statistic:**

```
t = (X_bar_t - X_bar_c) / SE
```

where the standard error is:

```
SE = sqrt(s_t^2 / n_t + s_c^2 / n_c)
```

- `X_bar_t`, `X_bar_c` -- sample means of treatment and control
- `s_t^2`, `s_c^2` -- sample variances (using Bessel's correction, i.e., dividing by `n - 1`)
- `n_t`, `n_c` -- sample sizes

**Welch-Satterthwaite degrees of freedom:**

```
df = (s_t^2/n_t + s_c^2/n_c)^2
     / ( (s_t^2/n_t)^2/(n_t - 1) + (s_c^2/n_c)^2/(n_c - 1) )
```

This approximation accounts for unequal variances. When `s_t^2 = s_c^2` and `n_t = n_c`,
it reduces to `df = n_t + n_c - 2` (Student's case).

**p-value (two-sided):**

```
p = 2 * P(T > |t|)    where T ~ t(df)
```

**Confidence interval for the difference of means:**

```
(X_bar_t - X_bar_c) +/- t_{alpha/2, df} * SE
```

### Key Assumptions

1. Observations within each group are independent.
2. Each group is drawn from a population with a finite mean and variance.
3. The sampling distribution of the mean is approximately normal (justified by CLT for
   `n >= 30`, or exact for normally distributed data).

### Implementation Notes

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
# equal_var=False triggers Welch's variant

# Manual CI construction:
se = np.sqrt(np.var(treatment, ddof=1)/len(treatment)
           + np.var(control, ddof=1)/len(control))
df = welch_satterthwaite_df(treatment, control)  # custom helper
t_crit = stats.t.ppf(1 - alpha/2, df)
ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
```

### References

- Welch, B. L. (1947). "The generalization of Student's problem when several different
  population variances are involved." *Biometrika*, 34(1-2), 28-35.
- Satterthwaite, F. E. (1946). "An approximate distribution of estimates of variance
  components." *Biometrics Bulletin*, 2(6), 110-114.

---

## 2. Z-Test for Proportions

### Problem

Compare conversion rates (or any binary outcome) between treatment and control. For example:
"Did the new checkout flow increase the purchase rate from 3.2% to 3.5%?"

### Mathematical Formulation

Let `p_t = s_t / n_t` and `p_c = s_c / n_c` be the observed proportions (successes / total).

**Test statistic (unpooled):**

```
z = (p_t - p_c) / SE
```

**Unpooled standard error:**

```
SE = sqrt( p_c*(1 - p_c)/n_c + p_t*(1 - p_t)/n_t )
```

**p-value (two-sided):**

```
p = 2 * (1 - Phi(|z|))
```

where `Phi` is the standard normal CDF.

**Confidence interval:**

```
(p_t - p_c) +/- z_{alpha/2} * SE
```

### Why Unpooled Standard Error

The pooled variant (`p_pool = (s_t + s_c) / (n_t + n_c)`) assumes the null hypothesis
`H0: p_t = p_c` is true when computing the SE. This is standard for hypothesis testing
but creates an inconsistency: the confidence interval (which should be valid under the
alternative) uses a variance estimate derived under the null.

The **unpooled** SE:
- Gives a more conservative test (slightly wider CI).
- Is consistent: the same SE formula drives both the test and the CI.
- Does not assume `H0` is true, making it appropriate for confidence intervals.
- Matches the approach used by major industry platforms (Statsig, Optimizely).

### Key Assumptions

1. Observations are independent Bernoulli trials.
2. `n * p >= 5` and `n * (1-p) >= 5` in both groups (normal approximation validity).
3. Stable conversion rate over the experiment window (no time-varying effects).

### Implementation Notes

```python
from scipy import stats

se = np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t)
z_stat = (p_t - p_c) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

z_crit = stats.norm.ppf(1 - alpha/2)
ci = (p_t - p_c - z_crit * se, p_t - p_c + z_crit * se)
```

### References

- Agresti, A., & Caffo, B. (2000). "Simple and effective confidence intervals for
  proportions and differences of proportions result from adding two successes and two
  failures." *The American Statistician*, 54(4), 280-288.

---

## 3. Group Sequential Testing (O'Brien-Fleming)

### Problem

In practice, experimenters peek at results before the planned sample size is reached. Naive
repeated testing inflates the false positive rate catastrophically.

**The peeking problem:**

| Planned peeks (at equal intervals) | Nominal alpha | Actual FPR |
|-------------------------------------|---------------|------------|
| 1 (no peeking)                      | 0.05          | 0.050      |
| 2                                   | 0.05          | 0.083      |
| 5                                   | 0.05          | 0.146      |
| 10                                  | 0.05          | 0.193      |
| 20                                  | 0.05          | 0.250      |

With 5 peeks, the true FPR is approximately **14.6%**, nearly triple the intended 5%.
Group sequential designs solve this by "spending" alpha across planned interim analyses.

### Mathematical Formulation

#### Lan-DeMets Alpha Spending Framework

Rather than fixing the number and timing of interim looks in advance, the Lan-DeMets
framework defines an **alpha spending function** `alpha*(t)` where `t in [0, 1]` is the
**information fraction** (proportion of total planned observations collected).

The function must satisfy:
- `alpha*(0) = 0`
- `alpha*(1) = alpha`
- `alpha*(t)` is non-decreasing

At each interim look `k` with information fraction `t_k`, the **incremental alpha spent**
is `alpha*(t_k) - alpha*(t_{k-1})`, and the critical boundary `z_k` is computed such that:

```
P(reject at look k | not rejected at looks 1..k-1, H0 true) = alpha*(t_k) - alpha*(t_{k-1})
```

#### O'Brien-Fleming (OBF) Spending Function

```
alpha*_OBF(t) = 2 - 2 * Phi( z_{alpha/2} / sqrt(t) )
```

where `Phi` is the standard normal CDF and `z_{alpha/2} = Phi^{-1}(1 - alpha/2)`.

Properties:
- Very conservative early: barely spends any alpha in initial looks.
- At `t = 0.5` (halfway), the critical z-value is approximately `2.80` (vs. `1.96` fixed).
- At `t = 1.0` (final look), the critical z-value is approximately `2.02`, very close to the
  fixed-sample value of `1.96`.
- Minimal loss of power compared to the fixed-sample design.

#### Pocock Spending Function

```
alpha*_Pocock(t) = alpha * ln(1 + (e - 1) * t)
```

Properties:
- Spends alpha more uniformly across looks.
- Uses the same critical value at every interim analysis.
- Final critical z-value is higher than OBF (more power lost at the end).
- Better when early stopping for efficacy is a priority.

#### Boundary Computation

At each analysis `k` (of `K` planned), the critical z-boundary `z_k` is the solution to:

```
P(|Z_k| >= z_k, |Z_1| < z_1, ..., |Z_{k-1}| < z_{k-1} | H0) = alpha*(t_k) - alpha*(t_{k-1})
```

This requires numerical integration over the joint distribution of the sequential test
statistics. In practice we use recursive numerical integration over the canonical joint
distribution of Brownian motion increments.

### Decision Rules

At each interim analysis `k`:
- **Stop for efficacy** if `|Z_k| >= z_k` (reject `H0`).
- **Continue** if `|Z_k| < z_k` and `k < K`.
- **Final decision** at `k = K`: reject if `|Z_K| >= z_K`.

Optionally, futility boundaries can be added (stop early if the effect is clearly too small
to reach significance by the final analysis).

### Key Assumptions

1. The number of planned interim analyses `K` is specified in advance.
2. Information fractions `t_1, ..., t_K` can be approximate (Lan-DeMets is flexible).
3. Test statistics at each look are computed from cumulative data.
4. The underlying test statistic satisfies the independent increments property
   (holds for z-tests and approximately for t-tests with large samples).

### Implementation Notes

```python
from scipy import stats
import numpy as np

def obf_spending(t, alpha=0.05):
    """O'Brien-Fleming alpha spending function."""
    z_alpha2 = stats.norm.ppf(1 - alpha/2)
    return 2 * (1 - stats.norm.cdf(z_alpha2 / np.sqrt(t)))

def pocock_spending(t, alpha=0.05):
    """Pocock alpha spending function."""
    return alpha * np.log(1 + (np.e - 1) * t)

def compute_boundary(info_frac, spending_func, alpha=0.05):
    """
    Compute z-boundaries for each look.
    Uses numerical root-finding on the multivariate normal joint CDF
    (via scipy.stats.multivariate_normal or recursive integration).
    """
    # Incremental alpha at each look
    K = len(info_frac)
    alpha_spent = [spending_func(info_frac[0], alpha)]
    for k in range(1, K):
        alpha_spent.append(
            spending_func(info_frac[k], alpha) - spending_func(info_frac[k-1], alpha)
        )
    # Root-find for boundaries (simplified; full version uses recursive integration)
    boundaries = []
    for k in range(K):
        z_k = stats.norm.ppf(1 - alpha_spent[k] / 2)  # approximate
        boundaries.append(z_k)
    return boundaries
```

### References

- O'Brien, P. C., & Fleming, T. R. (1979). "A multiple testing procedure for clinical
  trials." *Biometrics*, 35(3), 549-556.
- Lan, K. K. G., & DeMets, D. L. (1983). "Discrete sequential boundaries for clinical
  trials." *Biometrika*, 70(3), 659-663.
- Pocock, S. J. (1977). "Group sequential methods in the design and analysis of clinical
  trials." *Biometrika*, 64(2), 191-199.

---

## 4. Confidence Sequences (Always-Valid Inference)

### Problem

Group sequential testing requires pre-specifying the number of looks. In a modern
experimentation platform, dashboards update continuously and stakeholders check results at
unpredictable times. We need inference that is **valid at every stopping time**, not just
at pre-planned analysis points.

A **confidence sequence** `(CS_t)_{t=1}^{infty}` is a sequence of confidence sets such
that for a given confidence level `1 - alpha`:

```
P( theta in CS_t for all t >= 1 ) >= 1 - alpha
```

This is much stronger than the fixed-sample guarantee `P(theta in CI_n) >= 1 - alpha` for
a single pre-specified `n`.

### Mathematical Formulation

#### Mixture Sequential Probability Ratio Test (mSPRT)

We implement the mSPRT approach. For a stream of i.i.d. observations with known variance
`sigma^2`, the confidence sequence for the mean difference `delta` at sample size `n` is:

```
delta_hat_n +/- halfwidth_n
```

where the **half-width** is:

```
halfwidth_n = sqrt( (2 * sigma^2 * (sigma^2 + n * v)) / (n^2 * v)
                    * log( sqrt((sigma^2 + n * v) / sigma^2) / alpha ) )
```

- `delta_hat_n` -- observed mean difference at sample size `n`
- `sigma^2` -- pooled variance (estimated or known)
- `v` -- mixing variance parameter (tunes how the confidence sequence shrinks over time)
- `alpha` -- significance level

#### Choosing the Mixing Parameter `v`

The mixing parameter `v` determines the "shape" of the confidence sequence:
- Larger `v`: the CS is tighter at large sample sizes but wider at small sizes.
- Smaller `v`: the CS is tighter early on but converges more slowly.

A practical rule of thumb: set `v` such that the expected sample size under the alternative
minimizes the half-width. Typically:

```
v = sigma^2 * (z_{alpha/2} + z_{beta})^2 / n_planned
```

where `n_planned` is the planned sample size from a fixed-sample power calculation.

#### Key Property: Uniform Validity

Unlike a fixed-sample CI where `P(theta in CI_n) >= 1 - alpha` holds only at the single
time `n`, a confidence sequence satisfies:

```
P( theta in CS_n for ALL n = 1, 2, 3, ... ) >= 1 - alpha
```

This means:
- You can check results at any time without alpha inflation.
- You can stop the experiment whenever the CS excludes zero (or any practical threshold).
- No need to pre-specify looks or use alpha-spending functions.

#### Connection to Sequential Testing

Rejecting `H0: delta = 0` when `0 not in CS_n` is equivalent to an mSPRT with Type I error
at most `alpha`, uniformly over all stopping times. The confidence sequence inverts a family
of sequential likelihood ratio tests mixed over the alternative parameter space.

### Key Assumptions

1. Observations are independent (or satisfy a martingale condition).
2. The variance `sigma^2` is known or reliably estimated from a pre-experiment period.
3. The mixing distribution is specified before data collection begins.

### Implementation Notes

```python
import numpy as np

def confidence_sequence_halfwidth(n, sigma2, v, alpha):
    """
    Compute the half-width of the mSPRT confidence sequence at sample size n.

    Parameters:
        n      : current sample size (per group)
        sigma2 : pooled variance estimate
        v      : mixing variance parameter
        alpha  : significance level

    Returns:
        halfwidth : float
    """
    ratio = (sigma2 + n * v) / sigma2
    log_term = np.log(np.sqrt(ratio) / alpha)
    halfwidth = np.sqrt(
        (2 * sigma2 * (sigma2 + n * v)) / (n**2 * v) * log_term
    )
    return halfwidth

def always_valid_test(delta_hat, n, sigma2, v, alpha):
    """Returns (reject_null, ci_lower, ci_upper)."""
    hw = confidence_sequence_halfwidth(n, sigma2, v, alpha)
    ci_lower = delta_hat - hw
    ci_upper = delta_hat + hw
    reject = (ci_lower > 0) or (ci_upper < 0)
    return reject, ci_lower, ci_upper
```

### References

- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). "Time-uniform,
  nonparametric, nonasymptotic confidence sequences." *The Annals of Statistics*,
  49(2), 1055-1080.
- Johari, R., Pekelis, L., & Walsh, D. J. (2017). "Always valid inference: Continuous
  monitoring of A/B tests." *Operations Research*, 70(3), 1806-1821.
- Robbins, H. (1970). "Statistical methods related to the law of the iterated logarithm."
  *The Annals of Mathematical Statistics*, 41(5), 1397-1409.

---

## 5. Bayesian A/B Testing

### Problem

Frequentist methods answer: "What is the probability of seeing data this extreme if
there were no effect?" Bayesian methods answer a more natural question: "Given the data
we observed, what is the probability that the treatment is better than control?"

### Mathematical Formulation

#### 5a. Beta-Binomial Model (Binary Metrics)

**Prior:**

```
theta ~ Beta(alpha_0, beta_0)
```

Common choices:
- `Beta(1, 1)` -- uniform (non-informative)
- `Beta(0.5, 0.5)` -- Jeffreys prior (non-informative, invariant under reparameterization)
- `Beta(alpha_0, beta_0)` -- informative, encoding prior beliefs about the conversion rate

**Likelihood:**

```
s | theta ~ Binomial(n, theta)
```

where `s` = number of successes, `n` = number of trials.

**Posterior (conjugate update):**

```
theta | data ~ Beta(alpha_0 + s, beta_0 + n - s)
```

Denote the posterior for control as `Beta(a_c, b_c)` and treatment as `Beta(a_t, b_t)`.

#### Probability of Treatment Being Better

```
P(theta_t > theta_c | data)
```

This is computed via **Monte Carlo sampling**:

```
1. Draw N samples: theta_t^(i) ~ Beta(a_t, b_t), theta_c^(i) ~ Beta(a_c, b_c)
2. P(B > A) = (1/N) * sum_{i=1}^{N} I(theta_t^(i) > theta_c^(i))
```

For the Beta distribution specifically, an exact closed-form exists but the Monte Carlo
approach generalizes to all distributions and is computationally cheap (`N = 100,000`
samples take milliseconds).

#### Expected Loss (Risk)

Rather than just asking "is treatment better?", we want to know "how much do we lose by
choosing the wrong variant?" The expected loss of choosing treatment is:

```
E[loss | choose treatment] = E[max(0, theta_c - theta_t) | data]
```

Estimated via Monte Carlo:

```
expected_loss = (1/N) * sum_{i=1}^{N} max(0, theta_c^(i) - theta_t^(i))
```

**Decision rule:** Choose treatment if `E[loss | choose treatment] < epsilon` where
`epsilon` is a practical significance threshold (e.g., 0.1% conversion rate).

#### ROPE (Region of Practical Equivalence)

Define a region around zero where differences are practically meaningless:

```
ROPE = [-delta, +delta]
```

For example, `delta = 0.001` (0.1 percentage points).

Compute the posterior probability that the true difference falls inside the ROPE:

```
P(|theta_t - theta_c| < delta | data)
```

Decision rules:
- If `P(diff in ROPE) > 95%`: declare practical equivalence (no meaningful difference).
- If `P(diff > delta) > 95%`: declare treatment practically superior.
- Otherwise: inconclusive, collect more data.

#### 5b. Normal-Normal Model (Continuous Metrics)

For continuous metrics (revenue, session duration), we use a Normal likelihood with a
Normal prior on the mean.

**Prior:**

```
mu ~ Normal(mu_0, sigma_0^2)
```

**Likelihood (known variance sigma^2):**

```
X_bar | mu ~ Normal(mu, sigma^2 / n)
```

**Posterior:**

```
mu | data ~ Normal(mu_n, sigma_n^2)
```

where:

```
sigma_n^2 = 1 / (1/sigma_0^2 + n/sigma^2)
mu_n = sigma_n^2 * (mu_0/sigma_0^2 + n*X_bar/sigma^2)
```

With a vague prior (`sigma_0 -> infinity`), the posterior converges to:

```
mu | data ~ Normal(X_bar, sigma^2 / n)
```

which reproduces the frequentist result.

### Key Assumptions

1. **Beta-Binomial:** Each observation is an independent Bernoulli trial.
2. **Normal-Normal:** Observations are independent with known (or well-estimated) variance.
3. The prior is specified before observing the data.
4. Bayesian methods do not control frequentist error rates (Type I / Type II) by default,
   though calibrated priors can provide approximate frequentist guarantees.

### Implementation Notes

```python
from scipy import stats
import numpy as np

# Beta-Binomial posterior
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures
posterior = stats.beta(alpha_post, beta_post)

# P(treatment > control) via Monte Carlo
N = 100_000
samples_t = stats.beta.rvs(a_t, b_t, size=N)
samples_c = stats.beta.rvs(a_c, b_c, size=N)
prob_t_wins = np.mean(samples_t > samples_c)

# Expected loss
expected_loss_t = np.mean(np.maximum(0, samples_c - samples_t))
expected_loss_c = np.mean(np.maximum(0, samples_t - samples_c))

# ROPE
diff_samples = samples_t - samples_c
prob_in_rope = np.mean(np.abs(diff_samples) < rope_threshold)
```

### References

- Berry, D. A. (2006). "Bayesian clinical trials." *Nature Reviews Drug Discovery*,
  5(1), 27-36.
- Deng, A., Lu, J., & Chen, S. (2016). "Continuous monitoring of A/B tests without pain:
  Optional stopping in Bayesian testing." *IEEE DSAA*, 243-252.

---

## 6. Multi-Armed Bandits

### Problem

Traditional A/B testing allocates traffic equally between variants for the entire experiment
duration. This means that even when one variant is clearly worse, it continues receiving 50%
of traffic, resulting in **opportunity cost**. Multi-armed bandits adaptively shift traffic
toward better-performing variants during the experiment, reducing cumulative regret.

### Mathematical Formulation

#### 6a. Thompson Sampling

Thompson Sampling is a Bayesian approach that balances exploration and exploitation by
sampling from the posterior distribution of each arm's reward.

**Algorithm:**

```
At each time step t:
    For each arm i = 1, ..., K:
        Sample theta_i ~ Beta(alpha_i, beta_i)     # posterior for arm i
    Select arm a_t = argmax_i theta_i               # play the arm with highest sample
    Observe reward r_t
    Update posterior:
        If r_t = 1: alpha_{a_t} += 1
        If r_t = 0: beta_{a_t} += 1
```

**Properties:**
- Achieves asymptotically optimal regret: `O(K * log(T))` where `K` is the number of arms.
- Naturally adapts exploration: uncertain arms get explored more.
- Equivalent to "probability matching": each arm is pulled with probability equal to
  its posterior probability of being the best arm.

#### 6b. Upper Confidence Bound (UCB1)

UCB1 is a frequentist approach that selects the arm with the highest upper confidence bound.

**Algorithm:**

```
At each time step t:
    Select arm a_t = argmax_i [ X_bar_i + c * sqrt(ln(t) / n_i) ]
```

where:
- `X_bar_i` -- empirical mean reward of arm `i`
- `n_i` -- number of times arm `i` has been pulled
- `c` -- exploration parameter (typically `c = sqrt(2)` for theoretical guarantees)
- `t` -- total number of rounds so far

**Regret bound:**

```
E[R_T] <= 8 * sum_{i: mu_i < mu*} (ln(T) / Delta_i) + (1 + pi^2/3) * sum_{i} Delta_i
```

where `Delta_i = mu* - mu_i` is the gap between the best arm and arm `i`.

#### 6c. Epsilon-Greedy

The simplest bandit strategy: exploit the current best arm most of the time, explore
uniformly at random with small probability.

**Algorithm:**

```
At each time step t:
    With probability (1 - epsilon):
        Select arm a_t = argmax_i X_bar_i           # exploit
    With probability epsilon:
        Select arm a_t uniformly at random from {1, ..., K}  # explore
```

**Decaying epsilon** variant: `epsilon_t = min(1, c * K / (d^2 * t))` where `d` is a
lower bound on the minimum gap and `c > 0` is a tuning constant.

#### Cumulative Regret

The key metric for evaluating bandit algorithms:

```
R_T = sum_{t=1}^{T} (mu* - mu_{a_t})
```

where `mu*` is the mean reward of the best arm and `mu_{a_t}` is the mean reward of the
arm selected at time `t`.

**Expected regret comparison** (for two arms, gap `Delta`, `T` rounds):

| Algorithm        | Expected Regret          |
|------------------|--------------------------|
| Equal allocation | `Delta * T / 2`          |
| Epsilon-greedy   | `O(epsilon * T)`         |
| UCB1             | `O(ln(T) / Delta)`       |
| Thompson         | `O(ln(T) / Delta)`       |

#### Explore-Exploit Tradeoff vs. Statistical Validity

Bandits optimize for **cumulative reward during the experiment** but complicate
**post-experiment inference**. Because the assignment probabilities change over time, naive
estimators of the treatment effect are biased. For valid inference after a bandit experiment,
use inverse-propensity weighting (IPW):

```
delta_hat_IPW = (1/T) * sum_{t=1}^{T} [ r_t * I(a_t = treatment) / pi_t
                                        - r_t * I(a_t = control) / (1 - pi_t) ]
```

where `pi_t` is the probability of assigning to treatment at time `t`.

### Key Assumptions

1. Rewards are i.i.d. given the arm (stationarity).
2. Arms are independent.
3. For Thompson Sampling: the posterior model is correctly specified.
4. For UCB1: rewards are bounded in `[0, 1]` (or sub-Gaussian).

### Implementation Notes

```python
import numpy as np
from scipy import stats

class ThompsonSampling:
    def __init__(self, n_arms, alpha_prior=1, beta_prior=1):
        self.alphas = np.full(n_arms, alpha_prior, dtype=float)
        self.betas = np.full(n_arms, beta_prior, dtype=float)

    def select_arm(self):
        samples = stats.beta.rvs(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward

class UCB1:
    def __init__(self, n_arms, c=np.sqrt(2)):
        self.means = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.c = c
        self.t = 0

    def select_arm(self):
        self.t += 1
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        ucb = self.means + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] = self.means[arm] * (n - 1) / n + reward / n
```

### References

- Thompson, W. R. (1933). "On the likelihood that one unknown probability exceeds
  another in view of the evidence of two samples." *Biometrika*, 25(3/4), 285-294.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time analysis of the
  multiarmed bandit problem." *Machine Learning*, 47(2), 235-256.
- Chapelle, O., & Li, L. (2011). "An empirical evaluation of Thompson sampling."
  *Advances in Neural Information Processing Systems*, 24.

---

## 7. CUPED (Variance Reduction)

### Problem

A/B tests require large sample sizes because metrics have high variance. If we can reduce
the variance of our estimator, we can detect smaller effects with fewer samples (or detect
the same effects sooner). **CUPED** (Controlled-experiment Using Pre-Experiment Data) uses
pre-experiment covariates to reduce variance, analogous to control variates in Monte Carlo
simulation.

### Mathematical Formulation

Let:
- `Y` -- the metric of interest during the experiment (e.g., revenue per user this week)
- `X` -- a pre-experiment covariate (e.g., revenue per user last week)
- `E[X]` -- the known (or estimated) expectation of the covariate

**CUPED-adjusted metric:**

```
Y_adj = Y - theta * (X - E[X])
```

where:

```
theta = Cov(Y, X) / Var(X)
```

This is the coefficient from an OLS regression of `Y` on `X`.

**Key insight:** Since randomization ensures `E[X]` is the same across treatment and control,
the adjustment does not introduce bias:

```
E[Y_adj | treatment] - E[Y_adj | control]
    = E[Y | treatment] - E[Y | control]
    - theta * (E[X | treatment] - E[X | control])
                         ^^^^^^^^^^^^^^^^^^^^^^^^^
                         = 0 by randomization
```

**Variance reduction:**

```
Var(Y_adj) = Var(Y - theta * X)
           = Var(Y) - 2*theta*Cov(Y,X) + theta^2*Var(X)
           = Var(Y) - Cov(Y,X)^2 / Var(X)
           = Var(Y) * (1 - rho^2)
```

where `rho = Corr(Y, X)` is the Pearson correlation between the metric and the covariate.

**Variance reduction factor:**

```
VRF = 1 - rho^2
```

| Correlation (rho) | Variance Reduction | Equivalent Sample Size Multiplier |
|--------------------|--------------------|-----------------------------------|
| 0.0                | 0%                 | 1.00x                             |
| 0.3                | 9%                 | 1.10x                             |
| 0.5                | 25%                | 1.33x                             |
| 0.7                | 51%                | 2.04x                             |
| 0.8                | 64%                | 2.78x                             |
| 0.9                | 81%                | 5.26x                             |

A covariate with `rho = 0.7` effectively **doubles** the sample size for free.

### Choice of Covariate

The best covariate is the same metric measured in a pre-experiment period. For example:
- Metric: "revenue in experiment week" -> Covariate: "revenue in pre-experiment week"
- Metric: "page views during experiment" -> Covariate: "page views before experiment"

The covariate must be measured **before** the experiment starts (otherwise it may be
affected by the treatment, violating the unbiasedness guarantee).

### Extension: Multiple Covariates

With a vector of covariates `X = (X_1, ..., X_p)`:

```
Y_adj = Y - theta^T * (X - E[X])
```

where `theta = Var(X)^{-1} * Cov(X, Y)` (multivariate OLS coefficient). The variance
reduction is `1 - R^2` where `R^2` is the coefficient of determination from regressing
`Y` on `X`.

### Key Assumptions

1. The covariate `X` is measured **before** the experiment (pre-treatment).
2. Assignment to treatment/control is independent of `X` (randomization).
3. `Cov(Y, X)` and `Var(X)` are estimated from the combined (pooled) data.
4. `theta` can be computed from historical data or the current experiment data
   (both are valid; using current data gives a slightly biased variance estimate
   but is simpler).

### Implementation Notes

```python
import numpy as np

def cuped_adjust(y, x, x_mean=None):
    """
    Apply CUPED variance reduction.

    Parameters:
        y      : array, metric values during the experiment
        x      : array, pre-experiment covariate values
        x_mean : float, population mean of x (if known; otherwise estimated)

    Returns:
        y_adj  : array, CUPED-adjusted metric values
    """
    if x_mean is None:
        x_mean = np.mean(x)

    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x_mean)
    return y_adj

# Then run Welch's t-test on y_adj_treatment vs y_adj_control
```

### References

- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). "Improving the sensitivity of
  online controlled experiments by utilizing pre-experiment data." *Proceedings of the
  6th ACM International Conference on Web Search and Data Mining (WSDM)*, 123-132.
- Xie, H., & Aurisset, J. (2016). "Improving the sensitivity of online controlled
  experiments: Case studies at Netflix." *Proceedings of the 22nd ACM SIGKDD*, 645-654.

---

## 8. Multiple Testing Correction

### Problem

When an experiment tracks multiple metrics (e.g., conversion rate, revenue, engagement,
retention), each tested at `alpha = 0.05`, the probability of at least one false positive
grows rapidly:

```
P(at least one FP) = 1 - (1 - alpha)^m
```

| Metrics (m) | P(at least one FP) |
|-------------|---------------------|
| 1           | 0.050               |
| 5           | 0.226               |
| 10          | 0.401               |
| 20          | 0.642               |
| 50          | 0.923               |

### Two Philosophies

**FWER (Family-Wise Error Rate):** Control the probability of making **any** false
positive across all tests. Conservative. Appropriate when any single false positive has
serious consequences (e.g., regulatory submissions, guardrail metrics).

**FDR (False Discovery Rate):** Control the **expected proportion** of false positives
among all rejected hypotheses. Less conservative. Appropriate when testing many metrics
and accepting some false positives is tolerable (e.g., exploratory metric analysis).

### Mathematical Formulation

#### 8a. Bonferroni Correction (FWER)

The simplest FWER control: reject hypothesis `i` if `p_i <= alpha / m`.

Equivalently, compute adjusted p-values:

```
p_adj_i = min(p_i * m, 1)
```

**Properties:**
- Guarantees `FWER <= alpha` regardless of dependence structure.
- Very conservative when `m` is large (low power).
- Valid for any dependence structure between tests.

#### 8b. Holm's Step-Down Procedure (FWER)

A uniformly more powerful alternative to Bonferroni.

**Algorithm:**

```
1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
2. For k = 1, 2, ..., m:
       If p_(k) > alpha / (m - k + 1):
           Accept H_(k), H_(k+1), ..., H_(m)
           Break
       Else:
           Reject H_(k)
```

Equivalently, adjusted p-values:

```
p_adj_(k) = min( max_{j <= k} { p_(j) * (m - j + 1) }, 1 )
```

**Properties:**
- Uniformly more powerful than Bonferroni (rejects everything Bonferroni rejects, plus more).
- Still controls FWER at level `alpha`.
- Valid for any dependence structure.

#### 8c. Benjamini-Hochberg (BH) Procedure (FDR)

Controls the false discovery rate at level `q` (typically `q = 0.05`).

**Algorithm (step-up):**

```
1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
2. Find the largest k such that p_(k) <= k * q / m
3. Reject H_(1), H_(2), ..., H_(k)
```

Equivalently, adjusted p-values:

```
p_adj_(k) = min( min_{j >= k} { p_(j) * m / j }, 1 )
```

(computed from the largest index downward, taking running minimums).

**Properties:**
- Controls `FDR <= q` when tests are independent or positively correlated (PRDS condition).
- Much more powerful than FWER methods when `m` is large.
- The adjusted p-value can be interpreted as: "If I reject all hypotheses with
  adjusted `p <= q`, at most a fraction `q` of my rejections are expected to be false."

### When to Use Which

| Scenario                                   | Method               |
|--------------------------------------------|----------------------|
| 2-3 primary metrics, high stakes           | Holm (FWER)          |
| Guardrail metrics (must not regress)       | Bonferroni (FWER)    |
| 10+ exploratory metrics                    | Benjamini-Hochberg   |
| Single primary metric                      | No correction needed |
| Primary + secondary metric hierarchy       | Hierarchical testing |

### Key Assumptions

1. **Bonferroni / Holm:** No assumptions on dependence; always valid.
2. **BH:** Requires independence or positive regression dependence (PRDS). This holds for
   most A/B testing scenarios where metrics are positively correlated.
3. All methods assume each individual test has correct (unconditional) Type I error control.

### Implementation Notes

```python
from scipy import stats
import numpy as np

def bonferroni(p_values, alpha=0.05):
    """Bonferroni correction."""
    m = len(p_values)
    adjusted = np.minimum(np.array(p_values) * m, 1.0)
    rejected = adjusted <= alpha
    return adjusted, rejected

def holm(p_values, alpha=0.05):
    """Holm's step-down procedure."""
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    adjusted = np.zeros(m)
    for k in range(m):
        adjusted[k] = sorted_p[k] * (m - k)

    # Enforce monotonicity (step-down: take running max)
    for k in range(1, m):
        adjusted[k] = max(adjusted[k], adjusted[k-1])
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(m)
    result[order] = adjusted
    return result, result <= alpha

def benjamini_hochberg(p_values, q=0.05):
    """Benjamini-Hochberg FDR control."""
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    adjusted = np.zeros(m)
    for k in range(m):
        adjusted[k] = sorted_p[k] * m / (k + 1)

    # Enforce monotonicity (step-up: take running min from the end)
    for k in range(m - 2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k+1])
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(m)
    result[order] = adjusted
    return result, result <= q

# Or use statsmodels:
# from statsmodels.stats.multitest import multipletests
# rejected, adjusted, _, _ = multipletests(p_values, method='holm')
```

### References

- Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo delle
  probabilita." *Pubblicazioni del R Istituto Superiore di Scienze Economiche e
  Commerciali di Firenze*, 8, 3-62.
- Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
  *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: A
  practical and powerful approach to multiple testing." *Journal of the Royal Statistical
  Society: Series B*, 57(1), 289-300.

---

## 9. Power Analysis & Sample Size

### Problem

Before launching an experiment, determine how many observations are needed to detect a
given **minimum detectable effect (MDE)** with a specified **power** (typically 80%) at a
given **significance level** (typically 5%).

### Mathematical Formulation

#### 9a. Binary Metrics (Proportions)

Given:
- `p_c` -- baseline conversion rate (control)
- `p_t = p_c + delta` -- expected treatment conversion rate
- `alpha` -- significance level (two-sided)
- `beta` -- Type II error rate (power = `1 - beta`)

**Sample size per group:**

```
n = (z_{alpha/2} + z_{beta})^2 * (p_c*(1 - p_c) + p_t*(1 - p_t)) / delta^2
```

where `z_{alpha/2} = Phi^{-1}(1 - alpha/2)` and `z_{beta} = Phi^{-1}(1 - beta)`.

For `alpha = 0.05` and `power = 0.80`: `z_{0.025} = 1.96`, `z_{0.20} = 0.842`.

**Example:** Detect a lift from 5.0% to 5.5% (absolute `delta = 0.005`):

```
n = (1.96 + 0.842)^2 * (0.05*0.95 + 0.055*0.945) / 0.005^2
  = 7.85 * 0.09948 / 0.000025
  = 31,236 per group
```

#### 9b. Continuous Metrics

Given:
- `sigma^2` -- pooled variance of the metric
- `delta = mu_t - mu_c` -- minimum detectable difference in means

**Sample size per group:**

```
n = (z_{alpha/2} + z_{beta})^2 * 2 * sigma^2 / delta^2
```

The factor `2` accounts for comparing two groups with equal allocation.

#### 9c. Minimum Detectable Effect (MDE)

Given a fixed sample size `n`, the MDE is the smallest effect that can be detected:

```
delta = (z_{alpha/2} + z_{beta}) * sqrt(2 * sigma^2 / n)           # continuous
delta = (z_{alpha/2} + z_{beta}) * sqrt((p*(1-p) + p*(1-p)) / n)   # binary (approx)
```

#### 9d. Power as a Function of Effect Size

Given `n` and `delta`, the power is:

```
power = Phi( |delta| / SE - z_{alpha/2} )
```

where `SE = sqrt(2 * sigma^2 / n)` for continuous metrics.

#### 9e. Inverse Power (MDE via Root-Finding)

When the relationship between `n` and `delta` is not analytically invertible (e.g., with
CUPED adjustments or unequal allocation), we solve numerically:

```
Find delta such that: power(n, delta, alpha) = target_power
```

using `scipy.optimize.brentq` (bisection/Brent's method) on the function
`f(delta) = power(n, delta, alpha) - target_power`.

### Adjustments

| Factor                  | Effect on sample size              |
|-------------------------|------------------------------------|
| One-sided test          | Replace `z_{alpha/2}` with `z_alpha` (smaller n) |
| Unequal allocation k:1  | Multiply by `(1 + 1/k) / 2`       |
| CUPED with correlation rho | Multiply by `(1 - rho^2)`       |
| Multiple comparisons     | Replace `alpha` with `alpha/m` (Bonferroni) |

### Key Assumptions

1. The test statistic is approximately normally distributed (large sample).
2. The variance estimate is accurate (use historical data).
3. The effect size assumption is realistic.
4. No peeking (for fixed-sample power; adjust for sequential designs).

### Implementation Notes

```python
from scipy import stats, optimize
import numpy as np

def sample_size_proportion(p_c, p_t, alpha=0.05, power=0.80):
    """Sample size per group for a two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    delta = abs(p_t - p_c)
    numerator = (z_alpha + z_beta)**2 * (p_c*(1-p_c) + p_t*(1-p_t))
    n = numerator / delta**2
    return int(np.ceil(n))

def sample_size_continuous(sigma, delta, alpha=0.05, power=0.80):
    """Sample size per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n = (z_alpha + z_beta)**2 * 2 * sigma**2 / delta**2
    return int(np.ceil(n))

def compute_mde(n, sigma, alpha=0.05, power=0.80):
    """Minimum detectable effect for given sample size (continuous)."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    mde = (z_alpha + z_beta) * np.sqrt(2 * sigma**2 / n)
    return mde

def compute_mde_proportion(n, p_c, alpha=0.05, power=0.80):
    """MDE via root-finding for proportions."""
    def power_at_delta(delta):
        p_t = p_c + delta
        se = np.sqrt(p_c*(1-p_c)/n + p_t*(1-p_t)/n)
        z_crit = stats.norm.ppf(1 - alpha/2)
        noncentrality = delta / se
        return stats.norm.cdf(noncentrality - z_crit) - power
    mde = optimize.brentq(power_at_delta, 1e-8, 0.5)
    return mde
```

### References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
  Lawrence Erlbaum Associates.
- Lehr, R. (1992). "Sixteen S-squared over D-squared: A relation for crude sample size
  estimates." *Statistics in Medicine*, 11(8), 1099-1102.

---

## 10. Novelty & Primacy Detection

### Problem

Observed treatment effects may be unstable over time:
- **Novelty effect:** Users react positively to any change simply because it is new.
  The effect decays as the novelty wears off, leading to an overestimate of the long-term
  impact.
- **Primacy effect:** Users initially resist change (e.g., a new UI layout), but gradually
  adapt and even prefer it. The effect grows over time, leading to an underestimate if the
  experiment is stopped too early.

Detecting these temporal patterns is critical for making valid long-term decisions.

### Mathematical Formulation

#### Weighted Linear Regression

Compute the daily (or weekly) treatment effect `delta_d` for each day `d = 1, ..., D`
of the experiment, then fit a weighted linear regression:

```
delta_d = beta_0 + beta_1 * d + epsilon_d
```

where `epsilon_d ~ N(0, sigma_d^2)` and the weights are `w_d = 1 / sigma_d^2` (inverse
variance weights, where `sigma_d^2` is the squared standard error of the daily effect
estimate).

**Interpretation of `beta_1` (slope):**

| Slope       | Interpretation  | Action                              |
|-------------|-----------------|-------------------------------------|
| `beta_1 < 0` (significant) | Novelty effect  | Extend experiment or use later-period estimate |
| `beta_1 > 0` (significant) | Primacy effect  | Extend experiment or use later-period estimate |
| `beta_1 ~ 0` (not significant) | Stable effect   | Effect estimate is reliable          |

**Test for slope significance:**

```
t = beta_1_hat / SE(beta_1_hat)
```

where `SE(beta_1_hat)` is the standard error of the slope from the weighted least squares
fit. Compare against `t(D - 2)` distribution.

#### Practical Considerations

- **Minimum experiment duration:** Run for at least 2 weeks to detect novelty/primacy.
  Weekly cycles can confound daily trend analysis.
- **Day-of-week effects:** Include day-of-week indicators in the regression or analyze
  week-over-week effects.
- **New vs. returning users:** Analyze separately. New users cannot exhibit primacy effects
  (they have no prior experience). Returning users are most susceptible to both effects.

### Key Assumptions

1. Daily treatment effects are approximately normally distributed.
2. The time trend is approximately linear (can be extended to include quadratic terms).
3. Daily effect estimates are independent (holds approximately with non-overlapping
   cohorts or large sample sizes).

### Implementation Notes

```python
import numpy as np
from scipy import stats

def detect_novelty_primacy(daily_effects, daily_se):
    """
    Test for novelty/primacy effects via weighted linear regression.

    Parameters:
        daily_effects : array of daily treatment effect estimates
        daily_se      : array of standard errors for each daily estimate

    Returns:
        slope         : estimated trend (negative = novelty, positive = primacy)
        p_value       : significance of the slope
        interpretation: string description
    """
    D = len(daily_effects)
    days = np.arange(1, D + 1)
    weights = 1.0 / (daily_se ** 2)

    # Weighted least squares: delta_d = beta_0 + beta_1 * d
    W = np.diag(weights)
    X = np.column_stack([np.ones(D), days])
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ daily_effects)

    # Residuals and standard error of slope
    residuals = daily_effects - X @ beta
    sigma2 = np.sum(weights * residuals**2) / (D - 2)
    cov_beta = sigma2 * np.linalg.inv(X.T @ W @ X)
    se_slope = np.sqrt(cov_beta[1, 1])

    t_stat = beta[1] / se_slope
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=D - 2))

    if p_value < 0.05 and beta[1] < 0:
        interpretation = "Novelty effect detected (declining treatment effect)"
    elif p_value < 0.05 and beta[1] > 0:
        interpretation = "Primacy effect detected (increasing treatment effect)"
    else:
        interpretation = "No significant temporal trend (stable effect)"

    return beta[1], p_value, interpretation
```

### References

- Hohnhold, H., O'Brien, D., & Tang, D. (2015). "Focusing on the long-term: It's good
  for users and business." *Proceedings of the 21st ACM SIGKDD*, 1849-1858.
- Dmitriev, P., Gupta, S., Kim, D. W., & Vaz, G. (2017). "A dirty dozen: Twelve common
  metric interpretation pitfalls in online controlled experiments." *Proceedings of the
  23rd ACM SIGKDD*, 1427-1436.

---

## 11. Segment Analysis

### Problem

An overall treatment effect may mask heterogeneous effects across user segments. For example,
a feature change might help mobile users but hurt desktop users. Segment analysis tests for
such **heterogeneous treatment effects (HTE)** while controlling for the increased false
positive risk from multiple comparisons.

### Mathematical Formulation

#### Per-Segment Testing

For each segment `s in {1, ..., S}`, compute the treatment effect `delta_s` and p-value
`p_s` using the appropriate test (z-test for proportions, Welch's t-test for continuous
metrics).

Apply **multiple testing correction** (Section 8) across the `S` segment-level tests to
control for multiplicity.

#### Cochran's Q Test for Heterogeneity

Before examining individual segments, test whether there is **any** heterogeneity in
treatment effects across segments.

**Test statistic:**

```
Q = sum_{s=1}^{S} w_s * (delta_s - delta_overall)^2
```

where:
- `delta_s` -- treatment effect in segment `s`
- `delta_overall = sum(w_s * delta_s) / sum(w_s)` -- weighted average effect
- `w_s = 1 / SE_s^2` -- inverse-variance weight for segment `s`

Under `H0` (homogeneous effects), `Q ~ chi^2(S - 1)`.

**p-value:**

```
p = P(chi^2(S-1) > Q)
```

**Interpretation:**
- If `p < 0.05`: there is significant evidence of heterogeneity. Examine per-segment results.
- If `p >= 0.05`: no strong evidence of heterogeneity. Per-segment differences may be noise.

#### I-Squared Statistic

A descriptive measure of heterogeneity:

```
I^2 = max(0, (Q - (S - 1)) / Q) * 100%
```

| I-squared | Interpretation        |
|-----------|-----------------------|
| 0-25%     | Low heterogeneity     |
| 25-50%    | Moderate heterogeneity|
| 50-75%    | Substantial heterogeneity |
| 75-100%   | Considerable heterogeneity |

#### Workflow

```
1. Run Cochran's Q test for heterogeneity
2. If Q is significant (p < 0.05):
   a. Run per-segment tests
   b. Apply Benjamini-Hochberg FDR correction across segments
   c. Report segments with adjusted p < 0.05
3. If Q is not significant:
   a. Report overall effect; per-segment differences are likely noise
   b. Optionally show per-segment results with a caveat
```

### Key Assumptions

1. Users are independently assigned to treatment/control within each segment.
2. Segments are pre-defined (not data-dredged after seeing results).
3. Each segment has sufficient sample size for the per-segment test to be valid.
4. For Cochran's Q: the effect size estimator is approximately normal in each segment.

### Implementation Notes

```python
from scipy import stats
import numpy as np

def cochrans_q_test(effects, standard_errors):
    """
    Cochran's Q test for heterogeneity of treatment effects across segments.

    Parameters:
        effects          : array of per-segment treatment effect estimates
        standard_errors  : array of per-segment standard errors

    Returns:
        Q        : test statistic
        p_value  : p-value from chi-squared distribution
        I_squared: I^2 heterogeneity measure (percentage)
    """
    weights = 1.0 / (standard_errors ** 2)
    weighted_mean = np.sum(weights * effects) / np.sum(weights)
    Q = np.sum(weights * (effects - weighted_mean) ** 2)

    S = len(effects)
    df = S - 1
    p_value = 1 - stats.chi2.cdf(Q, df)

    I_squared = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    return Q, p_value, I_squared

def segment_analysis(segment_effects, segment_ses, segment_names, fdr_q=0.05):
    """
    Full segment analysis with heterogeneity test and FDR correction.
    """
    # Step 1: Heterogeneity test
    Q, q_pvalue, I2 = cochrans_q_test(segment_effects, segment_ses)

    # Step 2: Per-segment p-values
    z_stats = segment_effects / segment_ses
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    # Step 3: BH correction
    from statsmodels.stats.multitest import multipletests
    rejected, adjusted_p, _, _ = multipletests(p_values, alpha=fdr_q, method='fdr_bh')

    return {
        'heterogeneity_Q': Q,
        'heterogeneity_p': q_pvalue,
        'I_squared': I2,
        'segment_results': [
            {
                'segment': segment_names[i],
                'effect': segment_effects[i],
                'se': segment_ses[i],
                'p_raw': p_values[i],
                'p_adjusted': adjusted_p[i],
                'significant': rejected[i]
            }
            for i in range(len(segment_names))
        ]
    }
```

### References

- Cochran, W. G. (1954). "The combination of estimates from different experiments."
  *Biometrics*, 10(1), 101-129.
- Higgins, J. P. T., & Thompson, S. G. (2002). "Quantifying heterogeneity in a
  meta-analysis." *Statistics in Medicine*, 21(11), 1539-1558.

---

## References

### Foundational Texts

1. **Welch's T-Test:** Welch, B. L. (1947). "The generalization of Student's problem when
   several different population variances are involved." *Biometrika*, 34(1-2), 28-35.

2. **Group Sequential Methods:** Lan, K. K. G., & DeMets, D. L. (1983). "Discrete
   sequential boundaries for clinical trials." *Biometrika*, 70(3), 659-663.

3. **O'Brien-Fleming Boundaries:** O'Brien, P. C., & Fleming, T. R. (1979). "A multiple
   testing procedure for clinical trials." *Biometrics*, 35(3), 549-556.

4. **Confidence Sequences:** Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J.
   (2021). "Time-uniform, nonparametric, nonasymptotic confidence sequences." *The Annals
   of Statistics*, 49(2), 1055-1080.

5. **Always-Valid Inference:** Johari, R., Pekelis, L., & Walsh, D. J. (2017). "Always
   valid inference: Continuous monitoring of A/B tests." *Operations Research*, 70(3),
   1806-1821.

6. **Thompson Sampling:** Thompson, W. R. (1933). "On the likelihood that one unknown
   probability exceeds another in view of the evidence of two samples." *Biometrika*,
   25(3/4), 285-294.

7. **UCB1:** Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time analysis
   of the multiarmed bandit problem." *Machine Learning*, 47(2), 235-256.

8. **CUPED:** Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). "Improving the
   sensitivity of online controlled experiments by utilizing pre-experiment data."
   *Proceedings of WSDM*, 123-132.

9. **BH Procedure:** Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
   discovery rate." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

10. **Holm's Method:** Holm, S. (1979). "A simple sequentially rejective multiple test
    procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.

11. **Power Analysis:** Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
    Sciences* (2nd ed.). Lawrence Erlbaum Associates.

12. **Cochran's Q:** Cochran, W. G. (1954). "The combination of estimates from different
    experiments." *Biometrics*, 10(1), 101-129.

### Industry References

13. Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments:
    A Practical Guide to A/B Testing*. Cambridge University Press.

14. Dmitriev, P., Gupta, S., Kim, D. W., & Vaz, G. (2017). "A dirty dozen: Twelve
    common metric interpretation pitfalls in online controlled experiments." *Proceedings
    of KDD*, 1427-1436.

15. Deng, A., Lu, J., & Chen, S. (2016). "Continuous monitoring of A/B tests without
    pain: Optional stopping in Bayesian testing." *IEEE DSAA*, 243-252.

16. Chapelle, O., & Li, L. (2011). "An empirical evaluation of Thompson sampling."
    *NeurIPS*, 24.

17. Xie, H., & Aurisset, J. (2016). "Improving the sensitivity of online controlled
    experiments: Case studies at Netflix." *Proceedings of KDD*, 645-654.

18. Hohnhold, H., O'Brien, D., & Tang, D. (2015). "Focusing on the long-term: It's
    good for users and business." *Proceedings of KDD*, 1849-1858.

---

*This document is part of the Experimentor A/B Testing Platform.
For implementation details, see the source code in `/backend/app/services/statistics/`.*
