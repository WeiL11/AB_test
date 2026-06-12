import pytest
from app.stats.power_analysis import (
    required_sample_size,
    minimum_detectable_effect_func,
    compute_power,
    power_curve,
    estimate_duration,
)


class TestRequiredSampleSize:
    def test_binomial_reasonable_size(self):
        """10% baseline, 1pp MDE should need ~14k-16k per arm."""
        result = required_sample_size(0.10, 0.01)
        assert 10000 < result.required_sample_size_per_variant < 20000

    def test_larger_effect_needs_fewer_samples(self):
        small = required_sample_size(0.10, 0.01)
        large = required_sample_size(0.10, 0.05)
        assert large.required_sample_size_per_variant < small.required_sample_size_per_variant

    def test_higher_power_needs_more_samples(self):
        low_power = required_sample_size(0.10, 0.02, power=0.80)
        high_power = required_sample_size(0.10, 0.02, power=0.95)
        assert high_power.required_sample_size_per_variant > low_power.required_sample_size_per_variant

    def test_continuous_metric(self):
        result = required_sample_size(0, 0.5, metric_type="continuous", variance=1.0)
        assert result.required_sample_size_per_variant > 0

    def test_total_is_double_per_variant(self):
        result = required_sample_size(0.10, 0.02)
        assert result.total_sample_size == result.required_sample_size_per_variant * 2


class TestMDE:
    def test_round_trip_consistency(self):
        """n = sample_size(MDE=0.02) then MDE(n) should be approximately 0.02."""
        result = required_sample_size(0.10, 0.02)
        mde = minimum_detectable_effect_func(
            result.required_sample_size_per_variant, 0.10
        )
        assert mde == pytest.approx(0.02, abs=0.002)


class TestComputePower:
    def test_power_at_design_point(self):
        """Power should be ~0.80 at the designed sample size and MDE."""
        result = required_sample_size(0.10, 0.02)
        power = compute_power(result.required_sample_size_per_variant, 0.10, 0.02)
        assert power == pytest.approx(0.80, abs=0.02)

    def test_power_increases_with_sample_size(self):
        p1 = compute_power(1000, 0.10, 0.02)
        p2 = compute_power(5000, 0.10, 0.02)
        assert p2 > p1

    def test_power_increases_with_effect_size(self):
        p1 = compute_power(5000, 0.10, 0.01)
        p2 = compute_power(5000, 0.10, 0.05)
        assert p2 > p1


class TestPowerCurve:
    def test_returns_correct_number_of_points(self):
        curve = power_curve(0.10, 5000, n_points=10)
        assert len(curve) == 10

    def test_power_is_monotonically_increasing(self):
        curve = power_curve(0.10, 5000)
        powers = [p.power for p in curve]
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1] - 0.001  # allow tiny floating-point noise


class TestEstimateDuration:
    def test_basic_calculation(self):
        days = estimate_duration(5000, 1000, n_variants=2)
        assert days == 10  # 5000*2 / 1000

    def test_partial_allocation(self):
        days = estimate_duration(5000, 1000, n_variants=2, allocation_pct=50.0)
        assert days == 20  # 10000 / (1000 * 0.5)
