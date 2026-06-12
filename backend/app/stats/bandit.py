"""
Multi-Armed Bandit Algorithms for Adaptive Traffic Allocation.

Instead of fixed 50/50 splits, bandits dynamically shift traffic
toward better-performing variants, reducing opportunity cost.

Implements:
- Thompson Sampling: Sample from posterior, pick highest. Optimal exploration-exploitation.
- UCB1: Pick arm with highest upper confidence bound. Deterministic.
- Epsilon-Greedy: Exploit best arm (1-eps) of the time, explore uniformly otherwise.

Also tracks cumulative regret to evaluate bandit performance.

References:
- Thompson (1933). On the likelihood that one unknown probability exceeds another.
- Auer et al. (2002). Finite-time analysis of the multiarmed bandit problem.
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class BanditResult:
    """Result of bandit allocation computation."""
    arm_names: list[str]
    recommended_allocation: list[float]   # Traffic % per arm (sums to 100)
    estimated_means: list[float]          # Current mean reward per arm
    total_pulls: list[int]                # Number of observations per arm
    best_arm_index: int                   # Index of the currently best arm
    cumulative_regret: float | None = None


class ThompsonSampling:
    """
    Thompson Sampling for Bernoulli (conversion) bandits.

    Each arm has a Beta posterior. At each round:
    1. Sample theta_i ~ Beta(alpha_i, beta_i) for each arm
    2. Pick the arm with highest sampled theta

    For allocation: simulate many rounds, return fraction of wins per arm.
    """

    def get_allocation(
        self,
        successes: list[int],
        failures: list[int],
        n_simulations: int = 10_000,
    ) -> list[float]:
        """
        Compute recommended traffic allocation via Thompson Sampling.

        For each simulation round, sample theta_i ~ Beta(successes_i + 1, failures_i + 1)
        for every arm, and record which arm had the highest sampled theta.
        The allocation for each arm is the fraction of simulations it won, scaled to 100.

        Args:
            successes: List of success counts per arm
            failures: List of failure counts per arm
            n_simulations: Number of Monte Carlo simulations

        Returns:
            List of allocation percentages (sum to 100)
        """
        k = len(successes)
        if k != len(failures):
            raise ValueError("successes and failures must have the same length")
        if k == 0:
            raise ValueError("Need at least one arm")

        rng = np.random.default_rng(seed=42)

        # Posterior parameters: Beta(successes_i + 1, failures_i + 1)
        alphas = np.array([s + 1 for s in successes], dtype=float)
        betas = np.array([f + 1 for f in failures], dtype=float)

        # Draw samples: shape (n_simulations, k)
        samples = np.column_stack([
            rng.beta(alphas[i], betas[i], size=n_simulations)
            for i in range(k)
        ])

        # For each simulation, find which arm had the highest sample
        winners = np.argmax(samples, axis=1)

        # Count wins per arm
        win_counts = np.bincount(winners, minlength=k)
        allocation = (win_counts / n_simulations * 100).tolist()

        return allocation

    def select_arm(self, successes: list[int], failures: list[int]) -> int:
        """
        Select which arm to pull next (for real-time assignment).

        Draws one sample from each arm's Beta posterior and returns
        the index of the arm with the highest sampled value.

        Args:
            successes: List of success counts per arm
            failures: List of failure counts per arm

        Returns:
            Index of the selected arm
        """
        k = len(successes)
        if k != len(failures):
            raise ValueError("successes and failures must have the same length")
        if k == 0:
            raise ValueError("Need at least one arm")

        rng = np.random.default_rng()

        samples = np.array([
            rng.beta(successes[i] + 1, failures[i] + 1)
            for i in range(k)
        ])

        return int(np.argmax(samples))


class UCB1:
    """
    Upper Confidence Bound (UCB1) algorithm.

    Selects arm with highest: mean_i + c * sqrt(ln(t) / n_i)
    where t = total pulls, n_i = pulls for arm i, c = exploration constant.

    Deterministic (no randomness), good theoretical regret bounds.
    """

    def __init__(self, exploration_constant: float = 2.0):
        self.c = exploration_constant

    def _compute_ucb_scores(self, means: list[float], counts: list[int]) -> np.ndarray:
        """
        Compute UCB score for each arm.

        Arms with zero pulls get infinite UCB (must be explored first).

        Args:
            means: Current estimated mean reward for each arm
            counts: Number of pulls per arm

        Returns:
            Array of UCB scores
        """
        k = len(means)
        total = sum(counts)
        scores = np.empty(k)

        for i in range(k):
            if counts[i] == 0:
                scores[i] = np.inf
            else:
                exploration_bonus = self.c * np.sqrt(np.log(total) / counts[i])
                scores[i] = means[i] + exploration_bonus

        return scores

    def get_allocation(
        self,
        means: list[float],
        counts: list[int],
    ) -> list[float]:
        """
        Compute allocation based on UCB scores.

        Arms with higher UCB scores receive proportionally more traffic.
        Arms with zero pulls receive equal share of a guaranteed exploration
        budget to ensure they get tried.

        Args:
            means: Current estimated mean reward for each arm
            counts: Number of pulls per arm

        Returns:
            List of allocation percentages (sum to 100)
        """
        k = len(means)
        if k != len(counts):
            raise ValueError("means and counts must have the same length")
        if k == 0:
            raise ValueError("Need at least one arm")

        # If any arm has zero pulls, those arms split traffic equally
        zero_pull_arms = [i for i in range(k) if counts[i] == 0]
        if zero_pull_arms:
            allocation = [0.0] * k
            share = 100.0 / len(zero_pull_arms)
            for i in zero_pull_arms:
                allocation[i] = share
            return allocation

        scores = self._compute_ucb_scores(means, counts)

        # Convert scores to allocation proportionally
        # Shift scores so the minimum is at least 0 to avoid negative allocations
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score

        total_score = np.sum(scores)
        if total_score == 0:
            # All scores are equal -- allocate uniformly
            return [100.0 / k] * k

        allocation = (scores / total_score * 100).tolist()
        return allocation

    def select_arm(self, means: list[float], counts: list[int]) -> int:
        """
        Select the arm with highest UCB score.

        If any arm has zero pulls, it is selected first (round-robin
        through unexplored arms).

        Args:
            means: Current estimated mean reward for each arm
            counts: Number of pulls per arm

        Returns:
            Index of the selected arm
        """
        k = len(means)
        if k != len(counts):
            raise ValueError("means and counts must have the same length")
        if k == 0:
            raise ValueError("Need at least one arm")

        # If any arm has 0 pulls, select the first such arm
        for i in range(k):
            if counts[i] == 0:
                return i

        total = sum(counts)
        scores = np.array([
            means[i] + self.c * np.sqrt(np.log(total) / counts[i])
            for i in range(k)
        ])

        return int(np.argmax(scores))


class EpsilonGreedy:
    """
    Epsilon-Greedy: simple baseline.

    With probability (1-epsilon): pick best arm (exploit)
    With probability epsilon: pick random arm (explore)
    """

    def __init__(self, epsilon: float = 0.1):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.epsilon = epsilon

    def get_allocation(self, means: list[float]) -> list[float]:
        """
        Compute expected allocation percentages.

        The best arm (highest mean) receives:
            (1 - epsilon) * 100  +  (epsilon / k) * 100
        Each other arm receives:
            (epsilon / k) * 100

        Args:
            means: Current estimated mean reward for each arm

        Returns:
            List of allocation percentages (sum to 100)
        """
        k = len(means)
        if k == 0:
            raise ValueError("Need at least one arm")

        best_arm = int(np.argmax(means))
        explore_share = (self.epsilon / k) * 100

        allocation = [explore_share] * k
        allocation[best_arm] += (1 - self.epsilon) * 100

        return allocation

    def select_arm(self, means: list[float]) -> int:
        """
        Select an arm using epsilon-greedy strategy.

        With probability (1-epsilon), pick the arm with the highest mean.
        With probability epsilon, pick a uniformly random arm.

        Args:
            means: Current estimated mean reward for each arm

        Returns:
            Index of the selected arm
        """
        k = len(means)
        if k == 0:
            raise ValueError("Need at least one arm")

        rng = np.random.default_rng()

        if rng.random() < self.epsilon:
            return int(rng.integers(0, k))
        else:
            return int(np.argmax(means))


def compute_cumulative_regret(
    rewards: list[float],
    best_arm_mean: float,
) -> float:
    """
    Compute cumulative regret: sum(best_mean - received_reward).
    Lower is better. Measures how much reward was lost by exploring.

    Args:
        rewards: List of rewards received at each round
        best_arm_mean: The mean reward of the best arm (oracle value)

    Returns:
        Cumulative regret (non-negative float)
    """
    return sum(best_arm_mean - r for r in rewards)


def run_bandit_simulation(
    true_rates: list[float],
    algorithm: str = "thompson",
    n_rounds: int = 10000,
    epsilon: float = 0.1,
) -> BanditResult:
    """
    Simulate a bandit experiment with known true rates.
    Useful for comparing algorithms and for demo purposes.

    Runs the chosen algorithm for n_rounds, pulling arms and observing
    Bernoulli rewards drawn from the true_rates. Tracks cumulative
    regret and final allocation.

    Args:
        true_rates: True conversion rate for each arm
        algorithm: "thompson" | "ucb1" | "epsilon_greedy"
        n_rounds: Number of rounds to simulate
        epsilon: Epsilon for epsilon-greedy (ignored for other algorithms)

    Returns:
        BanditResult with final allocations and regret
    """
    k = len(true_rates)
    if k == 0:
        raise ValueError("Need at least one arm")
    if not all(0.0 <= r <= 1.0 for r in true_rates):
        raise ValueError("All rates must be between 0 and 1")
    if algorithm not in ("thompson", "ucb1", "epsilon_greedy"):
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'thompson', 'ucb1', or 'epsilon_greedy'")

    rng = np.random.default_rng(seed=42)

    successes = [0] * k
    failures = [0] * k
    counts = [0] * k
    total_rewards: list[float] = []
    best_mean = max(true_rates)

    # Initialize bandit
    if algorithm == "thompson":
        bandit = ThompsonSampling()
    elif algorithm == "ucb1":
        bandit = UCB1()
    else:
        bandit = EpsilonGreedy(epsilon=epsilon)

    for _ in range(n_rounds):
        # Select arm
        if algorithm == "thompson":
            arm = bandit.select_arm(successes, failures)
        elif algorithm == "ucb1":
            means = [
                successes[i] / counts[i] if counts[i] > 0 else 0.0
                for i in range(k)
            ]
            arm = bandit.select_arm(means, counts)
        else:
            means = [
                successes[i] / counts[i] if counts[i] > 0 else 0.0
                for i in range(k)
            ]
            arm = bandit.select_arm(means)

        # Observe reward (Bernoulli draw)
        reward = float(rng.random() < true_rates[arm])
        total_rewards.append(reward)

        # Update stats
        counts[arm] += 1
        if reward > 0:
            successes[arm] += 1
        else:
            failures[arm] += 1

    # Compute final metrics
    estimated_means = [
        successes[i] / counts[i] if counts[i] > 0 else 0.0
        for i in range(k)
    ]
    best_arm_index = int(np.argmax(estimated_means))
    cumulative_regret = compute_cumulative_regret(total_rewards, best_mean)

    # Compute final recommended allocation
    if algorithm == "thompson":
        allocation = bandit.get_allocation(successes, failures)
    elif algorithm == "ucb1":
        allocation = bandit.get_allocation(estimated_means, counts)
    else:
        allocation = bandit.get_allocation(estimated_means)

    arm_names = [f"arm_{i}" for i in range(k)]

    return BanditResult(
        arm_names=arm_names,
        recommended_allocation=allocation,
        estimated_means=estimated_means,
        total_pulls=counts,
        best_arm_index=best_arm_index,
        cumulative_regret=cumulative_regret,
    )
