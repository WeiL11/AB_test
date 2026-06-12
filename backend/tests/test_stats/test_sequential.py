"""
Unit tests for the Group Sequential Testing module.

Tests the GroupSequentialTest class and the sequential_z_test convenience
function, covering O'Brien-Fleming and Pocock alpha-spending functions,
boundary behaviour, rejection logic, and edge cases.
"""

import math

import numpy as np
import pytest
from scipy import stats as sp_stats

from app.stats.sequential import GroupSequentialTest, SequentialResult, sequential_z_test


# --------------------------------------------------------------------------- #
#  GroupSequentialTest -- O'Brien-Fleming spending
# --------------------------------------------------------------------------- #

class TestGroupSequentialOBF:
    """O'Brien-Fleming boundaries should be very strict early and relax later."""

    def test_obrien_fleming_boundaries_decrease(self):
        """O'Brien-Fleming boundaries should decrease with more information."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        boundaries = gst.compute_all_boundaries()
        assert len(boundaries) == 5
        # Boundaries should decrease: strict early, relaxed late
        for i in range(1, len(boundaries)):
            assert boundaries[i][1] <= boundaries[i - 1][1], (
                f"Boundary at look {i+1} ({boundaries[i][1]:.4f}) should be "
                f"<= boundary at look {i} ({boundaries[i-1][1]:.4f})"
            )

    def test_obrien_fleming_early_boundary_very_strict(self):
        """At 20% information, OBF boundary should be very high (>3.5)."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=1, z_statistic=2.0)
        assert result.boundary > 3.5, (
            f"First OBF boundary should exceed 3.5, got {result.boundary:.4f}"
        )
        assert not result.can_reject  # z=2.0 well below a boundary >3.5

    def test_obrien_fleming_final_boundary_reasonable(self):
        """At the final look with 5 analyses, boundary should be around 2.0."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=5, z_statistic=0.0)
        # The final OBF boundary is typically around 2.02-2.04 for K=5
        assert 1.9 < result.boundary < 2.3, (
            f"Final OBF boundary should be near 2.0, got {result.boundary:.4f}"
        )

    def test_single_analysis_boundary_near_1_96(self):
        """With exactly 1 analysis, the boundary should be ~1.96 (standard z)."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=1, spending="obrien_fleming")
        result = gst.analyze(analysis_number=1, z_statistic=0.0)
        assert result.boundary == pytest.approx(1.96, abs=0.05)

    def test_reject_when_z_exceeds_boundary(self):
        """Should reject when |z_statistic| exceeds the boundary."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        # At the last analysis, boundary is around 2.0; z=5.0 is way above.
        result = gst.analyze(analysis_number=5, z_statistic=5.0)
        assert result.can_reject
        assert result.recommendation == "stop_reject"

    def test_reject_with_negative_z_exceeding_boundary(self):
        """Should also reject for large negative z (two-sided test)."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=5, z_statistic=-5.0)
        assert result.can_reject
        assert result.recommendation == "stop_reject"

    def test_continue_when_z_below_boundary(self):
        """Should recommend 'continue' when z is below boundary at non-final look."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=2, z_statistic=1.0)
        assert not result.can_reject
        assert result.recommendation == "continue"

    def test_stop_futility_at_final_analysis_without_rejection(self):
        """At the final look with z below boundary, recommendation is stop_futility."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=5, z_statistic=0.5)
        assert not result.can_reject
        assert result.recommendation == "stop_futility"

    def test_alpha_spent_at_final_look_equals_alpha(self):
        """Cumulative alpha spent at 100% information should equal the total alpha."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        result = gst.analyze(analysis_number=5, z_statistic=0.0)
        assert result.alpha_spent == pytest.approx(0.05, abs=0.005)

    def test_alpha_spent_increases_with_looks(self):
        """Cumulative alpha spent should increase monotonically."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
        spent_values = []
        for k in range(1, 6):
            result = gst.analyze(analysis_number=k, z_statistic=0.0)
            spent_values.append(result.alpha_spent)
        for i in range(1, len(spent_values)):
            assert spent_values[i] >= spent_values[i - 1]

    def test_information_fraction_defaults(self):
        """Default information fraction should be analysis_number / n_analyses."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        result = gst.analyze(analysis_number=3, z_statistic=1.0)
        assert result.information_fraction == pytest.approx(0.6, abs=0.01)

    def test_custom_information_fraction(self):
        """Should honour a custom information fraction."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        result = gst.analyze(analysis_number=2, z_statistic=1.0, information_fraction=0.5)
        assert result.information_fraction == pytest.approx(0.5, abs=0.001)


# --------------------------------------------------------------------------- #
#  GroupSequentialTest -- Pocock spending
# --------------------------------------------------------------------------- #

class TestGroupSequentialPocock:
    """Pocock boundaries should be roughly constant across looks."""

    def test_pocock_boundaries_roughly_constant(self):
        """Pocock boundaries should be roughly constant across looks."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="pocock")
        boundaries = gst.compute_all_boundaries()
        boundary_values = [b[1] for b in boundaries]
        # Range should be small (within ~0.5 of each other)
        assert max(boundary_values) - min(boundary_values) < 0.6, (
            f"Pocock boundaries should be roughly constant, got range "
            f"{max(boundary_values) - min(boundary_values):.4f}: {boundary_values}"
        )

    def test_pocock_boundaries_above_standard_z(self):
        """Pocock boundaries should all exceed ~1.96 (paying for multiple looks)."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="pocock")
        boundaries = gst.compute_all_boundaries()
        for t, b in boundaries:
            assert b > 1.96, (
                f"Pocock boundary at t={t:.2f} should exceed 1.96, got {b:.4f}"
            )

    def test_pocock_alpha_spent_at_end(self):
        """At the final analysis, cumulative alpha should equal the total alpha."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="pocock")
        result = gst.analyze(analysis_number=5, z_statistic=0.0)
        assert result.alpha_spent == pytest.approx(0.05, abs=0.005)


# --------------------------------------------------------------------------- #
#  GroupSequentialTest -- compute_all_boundaries
# --------------------------------------------------------------------------- #

class TestComputeAllBoundaries:
    def test_length_matches_n_analyses(self):
        for k in (1, 3, 5, 10):
            gst = GroupSequentialTest(alpha=0.05, n_analyses=k)
            boundaries = gst.compute_all_boundaries()
            assert len(boundaries) == k

    def test_information_fractions_equally_spaced(self):
        gst = GroupSequentialTest(alpha=0.05, n_analyses=4)
        boundaries = gst.compute_all_boundaries()
        fractions = [b[0] for b in boundaries]
        expected = [0.25, 0.50, 0.75, 1.00]
        for f, e in zip(fractions, expected):
            assert f == pytest.approx(e, abs=0.001)

    def test_boundaries_are_positive(self):
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        boundaries = gst.compute_all_boundaries()
        for _, b in boundaries:
            assert b > 0.0


# --------------------------------------------------------------------------- #
#  GroupSequentialTest -- SequentialResult dataclass
# --------------------------------------------------------------------------- #

class TestSequentialResult:
    def test_result_has_all_fields(self):
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        result = gst.analyze(analysis_number=1, z_statistic=1.5)
        assert isinstance(result, SequentialResult)
        assert result.analysis_number == 1
        assert result.total_analyses == 5
        assert isinstance(result.information_fraction, float)
        assert isinstance(result.z_statistic, float)
        assert isinstance(result.boundary, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.alpha_spent, float)
        assert isinstance(result.can_reject, bool)
        assert result.recommendation in ("stop_reject", "stop_futility", "continue")

    def test_p_value_is_two_sided(self):
        """p-value should be 2 * (1 - Phi(|z|))."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=3)
        result = gst.analyze(analysis_number=2, z_statistic=1.96)
        expected_p = 2.0 * sp_stats.norm.sf(1.96)
        assert result.p_value == pytest.approx(expected_p, abs=0.001)

    def test_result_is_frozen(self):
        """SequentialResult should be immutable."""
        gst = GroupSequentialTest(alpha=0.05, n_analyses=3)
        result = gst.analyze(analysis_number=1, z_statistic=0.0)
        with pytest.raises(AttributeError):
            result.can_reject = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
#  GroupSequentialTest -- validation / edge cases
# --------------------------------------------------------------------------- #

class TestGroupSequentialValidation:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            GroupSequentialTest(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            GroupSequentialTest(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            GroupSequentialTest(alpha=-0.1)

    def test_invalid_n_analyses_raises(self):
        with pytest.raises(ValueError, match="n_analyses"):
            GroupSequentialTest(n_analyses=0)
        with pytest.raises(ValueError, match="n_analyses"):
            GroupSequentialTest(n_analyses=-1)

    def test_unknown_spending_raises(self):
        with pytest.raises(ValueError, match="spending"):
            GroupSequentialTest(spending="invalid")

    def test_analysis_number_out_of_range_raises(self):
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        with pytest.raises(ValueError, match="analysis_number"):
            gst.analyze(analysis_number=0, z_statistic=1.0)
        with pytest.raises(ValueError, match="analysis_number"):
            gst.analyze(analysis_number=6, z_statistic=1.0)

    def test_invalid_information_fraction_raises(self):
        gst = GroupSequentialTest(alpha=0.05, n_analyses=5)
        with pytest.raises(ValueError, match="information_fraction"):
            gst.analyze(analysis_number=1, z_statistic=1.0, information_fraction=0.0)
        with pytest.raises(ValueError, match="information_fraction"):
            gst.analyze(analysis_number=1, z_statistic=1.0, information_fraction=1.5)


# --------------------------------------------------------------------------- #
#  sequential_z_test convenience function
# --------------------------------------------------------------------------- #

class TestSequentialZTest:
    def test_with_significant_difference(self):
        """Should detect a large mean difference at the final analysis."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 5000)
        treatment = np.random.normal(110, 15, 5000)
        result = sequential_z_test(control, treatment, analysis_number=5, n_analyses=5)
        assert result.can_reject
        assert result.recommendation == "stop_reject"

    def test_with_no_difference(self):
        """Should not reject with no real difference, especially at early look."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 500)
        treatment = np.random.normal(100, 15, 500)
        result = sequential_z_test(control, treatment, analysis_number=1, n_analyses=5)
        assert not result.can_reject

    def test_z_statistic_direction(self):
        """z-statistic should be positive when treatment mean > control mean."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 2000)
        treatment = np.random.normal(110, 10, 2000)
        result = sequential_z_test(control, treatment, analysis_number=3, n_analyses=5)
        assert result.z_statistic > 0

    def test_with_max_sample_size(self):
        """Information fraction should reflect current vs. max sample size."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 500)
        treatment = np.random.normal(100, 10, 500)
        result = sequential_z_test(
            control, treatment,
            analysis_number=3,
            n_analyses=5,
            max_sample_size=1000,
        )
        # info_fraction = min(n_c, n_t) / max_sample_size = 500/1000 = 0.5
        assert result.information_fraction == pytest.approx(0.5, abs=0.01)

    def test_result_type(self):
        """Should return a SequentialResult."""
        np.random.seed(42)
        control = np.random.normal(0, 1, 100)
        treatment = np.random.normal(0, 1, 100)
        result = sequential_z_test(control, treatment, analysis_number=1, n_analyses=3)
        assert isinstance(result, SequentialResult)

    def test_too_few_observations_raises(self):
        """Should raise if either group has fewer than 2 observations."""
        with pytest.raises(ValueError, match="at least 2"):
            sequential_z_test(np.array([1.0]), np.array([2.0, 3.0]),
                              analysis_number=1, n_analyses=3)
        with pytest.raises(ValueError, match="at least 2"):
            sequential_z_test(np.array([1.0, 2.0]), np.array([3.0]),
                              analysis_number=1, n_analyses=3)

    def test_non_1d_array_raises(self):
        """Should raise if arrays are not 1-D."""
        with pytest.raises(ValueError, match="1-D"):
            sequential_z_test(
                np.array([[1, 2], [3, 4]]),
                np.array([5.0, 6.0]),
                analysis_number=1,
                n_analyses=3,
            )

    def test_pocock_spending_option(self):
        """Should accept 'pocock' spending function."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 200)
        treatment = np.random.normal(100, 10, 200)
        result = sequential_z_test(
            control, treatment,
            analysis_number=2,
            n_analyses=5,
            spending="pocock",
        )
        assert isinstance(result, SequentialResult)

    def test_different_alpha(self):
        """Should respect a custom alpha level."""
        np.random.seed(42)
        control = np.random.normal(100, 10, 200)
        treatment = np.random.normal(100, 10, 200)
        result = sequential_z_test(
            control, treatment,
            analysis_number=1,
            n_analyses=3,
            alpha=0.01,
        )
        assert isinstance(result, SequentialResult)
        # Boundary for alpha=0.01 should be wider than for alpha=0.05
        result_05 = sequential_z_test(
            control, treatment,
            analysis_number=1,
            n_analyses=3,
            alpha=0.05,
        )
        assert result.boundary > result_05.boundary
