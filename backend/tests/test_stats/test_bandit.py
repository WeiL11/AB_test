"""
Unit tests for the Multi-Armed Bandit module.

Tests Thompson Sampling, UCB1, Epsilon-Greedy algorithms, the
compute_cumulative_regret helper, and the run_bandit_simulation
end-to-end function.
"""

import numpy as np
import pytest

from app.stats.bandit import (
    BanditResult,
    EpsilonGreedy,
    ThompsonSampling,
    UCB1,
    compute_cumulative_regret,
    run_bandit_simulation,
)


# --------------------------------------------------------------------------- #
#  ThompsonSampling -- get_allocation
# --------------------------------------------------------------------------- #

class TestThompsonSamplingAllocation:
    def test_allocation_sums_to_100(self):
        ts = ThompsonSampling()
        alloc = ts.get_allocation([10, 5], [90, 95])
        assert sum(alloc) == pytest.approx(100.0, abs=0.1)

    def test_better_arm_gets_more_traffic(self):
        """Arm with higher success rate should get more allocation."""
        ts = ThompsonSampling()
        # Arm 0: 10% success, Arm 1: 1% success
        alloc = ts.get_allocation([100, 10], [900, 990])
        assert alloc[0] > alloc[1], (
            f"Arm 0 (10%) should beat Arm 1 (1%): {alloc}"
        )

    def test_three_arms_allocation(self):
        ts = ThompsonSampling()
        alloc = ts.get_allocation([50, 30, 10], [50, 70, 90])
        assert len(alloc) == 3
        assert sum(alloc) == pytest.approx(100.0, abs=0.1)
        # Best arm (50%) should get the most
        assert alloc[0] > alloc[1]
        assert alloc[0] > alloc[2]

    def test_equal_arms_roughly_equal(self):
        """With identical stats, allocation should be roughly uniform."""
        ts = ThompsonSampling()
        alloc = ts.get_allocation([50, 50, 50], [50, 50, 50])
        for a in alloc:
            assert 20.0 < a < 50.0  # roughly 33% each

    def test_single_arm(self):
        """Single arm should get 100% allocation."""
        ts = ThompsonSampling()
        alloc = ts.get_allocation([10], [90])
        assert len(alloc) == 1
        assert alloc[0] == pytest.approx(100.0, abs=0.1)

    def test_mismatched_lengths_raises(self):
        ts = ThompsonSampling()
        with pytest.raises(ValueError, match="same length"):
            ts.get_allocation([10, 20], [90])

    def test_empty_arms_raises(self):
        ts = ThompsonSampling()
        with pytest.raises(ValueError, match="at least one"):
            ts.get_allocation([], [])


# --------------------------------------------------------------------------- #
#  ThompsonSampling -- select_arm
# --------------------------------------------------------------------------- #

class TestThompsonSamplingSelectArm:
    def test_select_arm_returns_valid_index(self):
        ts = ThompsonSampling()
        arm = ts.select_arm([50, 30, 20], [50, 70, 80])
        assert 0 <= arm <= 2

    def test_select_arm_returns_int(self):
        ts = ThompsonSampling()
        arm = ts.select_arm([10, 20], [90, 80])
        assert isinstance(arm, int)

    def test_select_arm_with_no_data(self):
        """With uniform prior (no data), any arm is valid."""
        ts = ThompsonSampling()
        arm = ts.select_arm([0, 0], [0, 0])
        assert arm in (0, 1)


# --------------------------------------------------------------------------- #
#  UCB1 -- select_arm
# --------------------------------------------------------------------------- #

class TestUCB1SelectArm:
    def test_selects_unexplored_arm(self):
        """Arms with zero pulls should be selected first."""
        ucb = UCB1()
        arm = ucb.select_arm([0.5, 0.3], [10, 0])
        assert arm == 1  # Arm 1 has 0 pulls

    def test_selects_first_unexplored_arm(self):
        """When multiple arms have 0 pulls, the first one is selected."""
        ucb = UCB1()
        arm = ucb.select_arm([0.5, 0.3, 0.4], [10, 0, 0])
        assert arm == 1  # First zero-pull arm

    def test_selects_best_when_all_explored(self):
        """With sufficient exploration, UCB should prefer the better arm."""
        ucb = UCB1(exploration_constant=0.1)  # Low exploration
        arm = ucb.select_arm([0.9, 0.1], [1000, 1000])
        assert arm == 0  # Much better mean

    def test_returns_int(self):
        ucb = UCB1()
        arm = ucb.select_arm([0.5, 0.3], [100, 100])
        assert isinstance(arm, int)

    def test_mismatched_lengths_raises(self):
        ucb = UCB1()
        with pytest.raises(ValueError, match="same length"):
            ucb.select_arm([0.5], [10, 20])

    def test_empty_arms_raises(self):
        ucb = UCB1()
        with pytest.raises(ValueError, match="at least one"):
            ucb.select_arm([], [])


# --------------------------------------------------------------------------- #
#  UCB1 -- get_allocation
# --------------------------------------------------------------------------- #

class TestUCB1Allocation:
    def test_allocation_sums_to_100(self):
        ucb = UCB1()
        alloc = ucb.get_allocation([0.5, 0.3], [100, 100])
        assert sum(alloc) == pytest.approx(100.0, abs=0.1)

    def test_unexplored_arms_get_all_traffic(self):
        """Arms with zero pulls should share 100% of allocation."""
        ucb = UCB1()
        alloc = ucb.get_allocation([0.5, 0.3, 0.0], [100, 100, 0])
        # Only the zero-pull arm gets traffic
        assert alloc[2] == pytest.approx(100.0, abs=0.1)
        assert alloc[0] == pytest.approx(0.0, abs=0.1)

    def test_multiple_unexplored_split_equally(self):
        """Multiple zero-pull arms should split traffic equally."""
        ucb = UCB1()
        alloc = ucb.get_allocation([0.5, 0.0, 0.0], [100, 0, 0])
        assert alloc[1] == pytest.approx(50.0, abs=0.1)
        assert alloc[2] == pytest.approx(50.0, abs=0.1)

    def test_better_arm_gets_more_traffic(self):
        """With equal pulls, arm with higher mean should get more traffic."""
        ucb = UCB1()
        alloc = ucb.get_allocation([0.9, 0.1], [1000, 1000])
        assert alloc[0] > alloc[1]


# --------------------------------------------------------------------------- #
#  UCB1 -- exploration constant
# --------------------------------------------------------------------------- #

class TestUCB1ExplorationConstant:
    def test_custom_exploration_constant(self):
        ucb = UCB1(exploration_constant=0.5)
        assert ucb.c == 0.5

    def test_default_exploration_constant(self):
        ucb = UCB1()
        assert ucb.c == 2.0


# --------------------------------------------------------------------------- #
#  EpsilonGreedy -- get_allocation
# --------------------------------------------------------------------------- #

class TestEpsilonGreedyAllocation:
    def test_best_arm_gets_most_traffic(self):
        eg = EpsilonGreedy(epsilon=0.1)
        alloc = eg.get_allocation([0.5, 0.3, 0.1])
        # Best arm gets (1-eps)*100 + (eps/3)*100 = 90 + 3.33 = 93.33
        assert alloc[0] > 90.0

    def test_allocation_sums_to_100(self):
        eg = EpsilonGreedy(epsilon=0.1)
        alloc = eg.get_allocation([0.5, 0.3])
        assert sum(alloc) == pytest.approx(100.0, abs=0.1)

    def test_non_best_arms_get_explore_share(self):
        """Non-best arms should each get epsilon/k * 100."""
        eg = EpsilonGreedy(epsilon=0.1)
        alloc = eg.get_allocation([0.5, 0.3, 0.1])
        expected_explore = (0.1 / 3) * 100  # ~3.33%
        assert alloc[1] == pytest.approx(expected_explore, abs=0.1)
        assert alloc[2] == pytest.approx(expected_explore, abs=0.1)

    def test_epsilon_zero_all_to_best(self):
        """With epsilon=0, all traffic goes to the best arm."""
        eg = EpsilonGreedy(epsilon=0.0)
        alloc = eg.get_allocation([0.5, 0.3])
        assert alloc[0] == pytest.approx(100.0, abs=0.1)
        assert alloc[1] == pytest.approx(0.0, abs=0.1)

    def test_epsilon_one_uniform(self):
        """With epsilon=1.0, traffic should be uniform across all arms."""
        eg = EpsilonGreedy(epsilon=1.0)
        alloc = eg.get_allocation([0.5, 0.3, 0.1])
        for a in alloc:
            assert a == pytest.approx(100.0 / 3, abs=0.1)

    def test_single_arm(self):
        eg = EpsilonGreedy(epsilon=0.1)
        alloc = eg.get_allocation([0.5])
        assert alloc[0] == pytest.approx(100.0, abs=0.1)

    def test_empty_arms_raises(self):
        eg = EpsilonGreedy(epsilon=0.1)
        with pytest.raises(ValueError, match="at least one"):
            eg.get_allocation([])


# --------------------------------------------------------------------------- #
#  EpsilonGreedy -- select_arm
# --------------------------------------------------------------------------- #

class TestEpsilonGreedySelectArm:
    def test_select_arm_returns_valid_index(self):
        eg = EpsilonGreedy(epsilon=0.1)
        arm = eg.select_arm([0.5, 0.3, 0.1])
        assert 0 <= arm <= 2

    def test_select_arm_greedy_picks_best(self):
        """With epsilon=0, should always pick the best arm."""
        eg = EpsilonGreedy(epsilon=0.0)
        # Deterministic -- always picks the best
        for _ in range(10):
            arm = eg.select_arm([0.1, 0.9, 0.3])
            assert arm == 1


# --------------------------------------------------------------------------- #
#  EpsilonGreedy -- validation
# --------------------------------------------------------------------------- #

class TestEpsilonGreedyValidation:
    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedy(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon"):
            EpsilonGreedy(epsilon=1.5)


# --------------------------------------------------------------------------- #
#  compute_cumulative_regret
# --------------------------------------------------------------------------- #

class TestCumulativeRegret:
    def test_zero_regret_when_always_best(self):
        """If every reward equals the best arm mean, regret is zero."""
        regret = compute_cumulative_regret([1.0, 1.0, 1.0], 1.0)
        assert regret == pytest.approx(0.0)

    def test_positive_regret_when_suboptimal(self):
        """Regret = sum(best_mean - reward) for each round."""
        regret = compute_cumulative_regret([0.5, 0.5, 0.5], 1.0)
        assert regret == pytest.approx(1.5)

    def test_regret_with_mixed_rewards(self):
        regret = compute_cumulative_regret([0.0, 1.0, 0.5], 1.0)
        # (1-0) + (1-1) + (1-0.5) = 1.0 + 0.0 + 0.5 = 1.5
        assert regret == pytest.approx(1.5)

    def test_empty_rewards(self):
        regret = compute_cumulative_regret([], 1.0)
        assert regret == pytest.approx(0.0)

    def test_regret_is_non_negative_for_bounded_rewards(self):
        """When rewards never exceed best_arm_mean, regret should be non-negative."""
        rewards = [0.2, 0.8, 0.5, 0.3]
        regret = compute_cumulative_regret(rewards, 1.0)
        assert regret >= 0.0


# --------------------------------------------------------------------------- #
#  run_bandit_simulation -- Thompson Sampling
# --------------------------------------------------------------------------- #

class TestBanditSimulationThompson:
    def test_finds_best_arm(self):
        """Thompson sampling should identify the arm with the highest rate."""
        result = run_bandit_simulation(
            [0.1, 0.5, 0.3], algorithm="thompson", n_rounds=5000
        )
        assert result.best_arm_index == 1  # 50% rate arm

    def test_allocation_sums_to_100(self):
        result = run_bandit_simulation(
            [0.2, 0.3], algorithm="thompson", n_rounds=1000
        )
        assert sum(result.recommended_allocation) == pytest.approx(100.0, abs=0.1)

    def test_total_pulls_sum_to_n_rounds(self):
        n_rounds = 1000
        result = run_bandit_simulation(
            [0.2, 0.3], algorithm="thompson", n_rounds=n_rounds
        )
        assert sum(result.total_pulls) == n_rounds

    def test_total_pulls_length_matches_arms(self):
        result = run_bandit_simulation(
            [0.1, 0.2, 0.3, 0.4], algorithm="thompson", n_rounds=500
        )
        assert len(result.total_pulls) == 4
        assert len(result.recommended_allocation) == 4
        assert len(result.estimated_means) == 4

    def test_arm_names_generated(self):
        result = run_bandit_simulation(
            [0.1, 0.2, 0.3], algorithm="thompson", n_rounds=100
        )
        assert result.arm_names == ["arm_0", "arm_1", "arm_2"]

    def test_cumulative_regret_present(self):
        result = run_bandit_simulation(
            [0.2, 0.5], algorithm="thompson", n_rounds=1000
        )
        assert result.cumulative_regret is not None
        assert result.cumulative_regret >= 0.0

    def test_best_arm_gets_most_pulls(self):
        """With enough rounds, the best arm should receive the most pulls."""
        result = run_bandit_simulation(
            [0.1, 0.5, 0.2], algorithm="thompson", n_rounds=5000
        )
        assert result.total_pulls[1] > result.total_pulls[0]
        assert result.total_pulls[1] > result.total_pulls[2]


# --------------------------------------------------------------------------- #
#  run_bandit_simulation -- UCB1
# --------------------------------------------------------------------------- #

class TestBanditSimulationUCB1:
    def test_ucb1_finds_best_arm(self):
        result = run_bandit_simulation(
            [0.1, 0.5, 0.3], algorithm="ucb1", n_rounds=5000
        )
        assert result.best_arm_index == 1

    def test_ucb1_allocation_sums_to_100(self):
        result = run_bandit_simulation(
            [0.2, 0.3], algorithm="ucb1", n_rounds=1000
        )
        assert sum(result.recommended_allocation) == pytest.approx(100.0, abs=0.1)

    def test_ucb1_total_pulls(self):
        result = run_bandit_simulation(
            [0.2, 0.3], algorithm="ucb1", n_rounds=500
        )
        assert sum(result.total_pulls) == 500


# --------------------------------------------------------------------------- #
#  run_bandit_simulation -- Epsilon-Greedy
# --------------------------------------------------------------------------- #

class TestBanditSimulationEpsilonGreedy:
    def test_epsilon_greedy_finds_best_arm(self):
        result = run_bandit_simulation(
            [0.1, 0.5, 0.3], algorithm="epsilon_greedy", n_rounds=5000, epsilon=0.1
        )
        assert result.best_arm_index == 1

    def test_epsilon_greedy_total_pulls(self):
        result = run_bandit_simulation(
            [0.2, 0.4], algorithm="epsilon_greedy", n_rounds=1000
        )
        assert sum(result.total_pulls) == 1000


# --------------------------------------------------------------------------- #
#  run_bandit_simulation -- validation
# --------------------------------------------------------------------------- #

class TestBanditSimulationValidation:
    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            run_bandit_simulation([], algorithm="thompson", n_rounds=100)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            run_bandit_simulation([1.5, 0.3], algorithm="thompson")

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            run_bandit_simulation([-0.1, 0.3], algorithm="thompson")

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_bandit_simulation([0.2, 0.3], algorithm="random")


# --------------------------------------------------------------------------- #
#  run_bandit_simulation -- result structure
# --------------------------------------------------------------------------- #

class TestBanditSimulationResult:
    def test_result_type(self):
        result = run_bandit_simulation([0.2, 0.3], algorithm="thompson", n_rounds=100)
        assert isinstance(result, BanditResult)

    def test_result_is_frozen(self):
        result = run_bandit_simulation([0.2, 0.3], algorithm="thompson", n_rounds=100)
        with pytest.raises(AttributeError):
            result.best_arm_index = 0  # type: ignore[misc]

    def test_estimated_means_in_valid_range(self):
        """Estimated means should be in [0, 1] for Bernoulli rewards."""
        result = run_bandit_simulation(
            [0.2, 0.5, 0.8], algorithm="thompson", n_rounds=2000
        )
        for m in result.estimated_means:
            assert 0.0 <= m <= 1.0
