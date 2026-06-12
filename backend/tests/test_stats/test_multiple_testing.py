"""
Comprehensive tests for the Multiple Testing Correction module.

Tests cover Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg correction methods,
plus the apply_correction dispatcher. Verifies:
- Adjusted p-value computation
- Capping at 1.0
- Significance determination
- Empty input handling
- Metric name preservation
- Relative power of methods (BH >= Holm >= Bonferroni)
- Monotonicity of adjusted p-values
- Method dispatch and aliases
- Error handling for unknown methods
"""
import pytest

from app.stats.multiple_testing import (
    bonferroni,
    holm,
    benjamini_hochberg,
    apply_correction,
    CorrectedResult,
)


class TestBonferroni:
    """Tests for the Bonferroni correction method."""

    def test_multiplies_by_m(self):
        """Bonferroni should multiply each p-value by the number of tests."""
        results = bonferroni([0.01, 0.04, 0.03])
        assert results[0].adjusted_p_value == pytest.approx(0.03)
        assert results[1].adjusted_p_value == pytest.approx(0.12)
        assert results[2].adjusted_p_value == pytest.approx(0.09)

    def test_caps_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        results = bonferroni([0.5, 0.8])
        assert results[0].adjusted_p_value == 1.0
        assert results[1].adjusted_p_value == 1.0

    def test_significance_with_correction(self):
        """Only p-values whose adjusted values are < alpha should be significant."""
        results = bonferroni([0.01, 0.04, 0.10])
        # p=0.01 * 3 = 0.03 < 0.05 => significant
        assert results[0].is_significant
        # p=0.04 * 3 = 0.12 > 0.05 => not significant
        assert not results[1].is_significant
        # p=0.10 * 3 = 0.30 > 0.05 => not significant
        assert not results[2].is_significant

    def test_empty_list(self):
        """Empty input should return empty list."""
        results = bonferroni([])
        assert results == []

    def test_single_test_unchanged(self):
        """With only one test, adjusted p-value equals original."""
        results = bonferroni([0.03])
        assert results[0].adjusted_p_value == pytest.approx(0.03)
        assert results[0].original_p_value == pytest.approx(0.03)

    def test_metric_names_preserved(self):
        """Metric names should be attached to the correct results."""
        results = bonferroni([0.01, 0.04], metric_names=["ctr", "revenue"])
        assert results[0].metric_name == "ctr"
        assert results[1].metric_name == "revenue"

    def test_metric_names_none_by_default(self):
        """When metric_names is not provided, metric_name should be None."""
        results = bonferroni([0.01, 0.04])
        assert results[0].metric_name is None
        assert results[1].metric_name is None

    def test_method_field_is_bonferroni(self):
        """The method field should be 'bonferroni'."""
        results = bonferroni([0.01])
        assert results[0].method == "bonferroni"

    def test_original_p_values_preserved(self):
        """Original p-values should be preserved unchanged."""
        p_values = [0.01, 0.05, 0.10]
        results = bonferroni(p_values)
        for r, p in zip(results, p_values):
            assert r.original_p_value == p

    def test_custom_alpha(self):
        """Custom alpha should change significance decisions."""
        # With alpha=0.10: 0.01 * 2 = 0.02 < 0.10 => significant
        # With alpha=0.01: 0.01 * 2 = 0.02 > 0.01 => not significant
        results_lenient = bonferroni([0.01, 0.04], alpha=0.10)
        results_strict = bonferroni([0.01, 0.04], alpha=0.01)
        assert results_lenient[0].is_significant
        assert not results_strict[0].is_significant

    def test_all_significant(self):
        """When all p-values are very small, all should remain significant."""
        results = bonferroni([0.001, 0.005, 0.010], alpha=0.05)
        assert all(r.is_significant for r in results)

    def test_none_significant(self):
        """When all p-values are large, none should be significant."""
        results = bonferroni([0.5, 0.6, 0.9])
        assert not any(r.is_significant for r in results)


class TestHolm:
    """Tests for the Holm-Bonferroni step-down procedure."""

    def test_more_powerful_than_bonferroni(self):
        """Holm should reject at least as many hypotheses as Bonferroni."""
        p_values = [0.001, 0.013, 0.04, 0.06, 0.10]
        bon = bonferroni(p_values)
        hol = holm(p_values)
        n_bon_sig = sum(1 for r in bon if r.is_significant)
        n_holm_sig = sum(1 for r in hol if r.is_significant)
        assert n_holm_sig >= n_bon_sig

    def test_adjusted_p_monotonic_in_sorted_order(self):
        """After sorting by original p-value, adjusted p-values should be non-decreasing."""
        p_values = [0.04, 0.01, 0.001, 0.10]
        results = holm(p_values)
        # Sort by original p-value
        sorted_results = sorted(results, key=lambda r: r.original_p_value)
        adj = [r.adjusted_p_value for r in sorted_results]
        for i in range(1, len(adj)):
            assert adj[i] >= adj[i - 1] - 1e-10, (
                f"Monotonicity violated at index {i}: {adj[i]} < {adj[i-1]}"
            )

    def test_single_test_equals_original(self):
        """With one test, Holm should return the original p-value."""
        results = holm([0.03])
        assert results[0].adjusted_p_value == pytest.approx(0.03)

    def test_empty_list(self):
        """Empty input should return empty list."""
        results = holm([])
        assert results == []

    def test_method_field_is_holm(self):
        """The method field should be 'holm'."""
        results = holm([0.01])
        assert results[0].method == "holm"

    def test_holm_step_down_logic(self):
        """
        Verify Holm step-down with a known example.
        Sorted p-values: [0.001, 0.01, 0.04, 0.10] with m=4
        Step 1: 0.001 * 4 = 0.004
        Step 2: max(0.004, 0.01 * 3) = max(0.004, 0.03) = 0.03
        Step 3: max(0.03, 0.04 * 2) = max(0.03, 0.08) = 0.08
        Step 4: max(0.08, 0.10 * 1) = max(0.08, 0.10) = 0.10
        """
        p_values = [0.04, 0.10, 0.001, 0.01]
        results = holm(p_values)

        # Map back: index 0 had p=0.04 => adjusted=0.08
        # index 1 had p=0.10 => adjusted=0.10
        # index 2 had p=0.001 => adjusted=0.004
        # index 3 had p=0.01 => adjusted=0.03
        assert results[2].adjusted_p_value == pytest.approx(0.004, rel=1e-10)
        assert results[3].adjusted_p_value == pytest.approx(0.03, rel=1e-10)
        assert results[0].adjusted_p_value == pytest.approx(0.08, rel=1e-10)
        assert results[1].adjusted_p_value == pytest.approx(0.10, rel=1e-10)

    def test_caps_at_one(self):
        """Holm-adjusted p-values should be capped at 1.0."""
        results = holm([0.5, 0.8])
        for r in results:
            assert r.adjusted_p_value <= 1.0

    def test_metric_names_preserved(self):
        """Metric names should be attached to the correct results."""
        results = holm([0.01, 0.04], metric_names=["ctr", "revenue"])
        assert results[0].metric_name == "ctr"
        assert results[1].metric_name == "revenue"

    def test_preserves_original_p_values(self):
        """Original p-values should be preserved unchanged."""
        p_values = [0.01, 0.05, 0.10]
        results = holm(p_values)
        for r, p in zip(results, p_values):
            assert r.original_p_value == p


class TestBenjaminiHochberg:
    """Tests for the Benjamini-Hochberg FDR procedure."""

    def test_less_conservative_than_bonferroni(self):
        """BH should reject at least as many hypotheses as Bonferroni."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05, 0.10, 0.20]
        bon = bonferroni(p_values)
        bh = benjamini_hochberg(p_values)
        n_bon_sig = sum(1 for r in bon if r.is_significant)
        n_bh_sig = sum(1 for r in bh if r.is_significant)
        assert n_bh_sig >= n_bon_sig

    def test_adjusted_p_capped_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        results = benjamini_hochberg([0.5, 0.9])
        for r in results:
            assert r.adjusted_p_value <= 1.0

    def test_preserves_original_p_values(self):
        """Original p-values should be preserved unchanged."""
        p_values = [0.01, 0.05, 0.10]
        results = benjamini_hochberg(p_values)
        for r, p in zip(results, p_values):
            assert r.original_p_value == p

    def test_empty_list(self):
        """Empty input should return empty list."""
        results = benjamini_hochberg([])
        assert results == []

    def test_single_test_unchanged(self):
        """With one test, BH should return the original p-value."""
        results = benjamini_hochberg([0.03])
        assert results[0].adjusted_p_value == pytest.approx(0.03)

    def test_method_field_is_benjamini_hochberg(self):
        """The method field should be 'benjamini_hochberg'."""
        results = benjamini_hochberg([0.01])
        assert results[0].method == "benjamini_hochberg"

    def test_bh_step_up_logic(self):
        """
        Verify BH step-up with a known example.
        Sorted p-values: [0.001, 0.01, 0.04, 0.10] with m=4
        Step (from bottom):
          rank 4: 0.10 * 4/4 = 0.10
          rank 3: min(0.10, 0.04 * 4/3) = min(0.10, 0.0533) = 0.0533
          rank 2: min(0.0533, 0.01 * 4/2) = min(0.0533, 0.02) = 0.02
          rank 1: min(0.02, 0.001 * 4/1) = min(0.02, 0.004) = 0.004
        """
        p_values = [0.04, 0.10, 0.001, 0.01]
        results = benjamini_hochberg(p_values)

        assert results[2].adjusted_p_value == pytest.approx(0.004, rel=1e-10)
        assert results[3].adjusted_p_value == pytest.approx(0.02, rel=1e-10)
        assert results[0].adjusted_p_value == pytest.approx(4 / 3 * 0.04, rel=1e-10)
        assert results[1].adjusted_p_value == pytest.approx(0.10, rel=1e-10)

    def test_adjusted_p_monotonic_in_sorted_order(self):
        """After sorting by original p, adjusted p-values should be non-decreasing."""
        p_values = [0.04, 0.01, 0.001, 0.10, 0.05]
        results = benjamini_hochberg(p_values)
        sorted_results = sorted(results, key=lambda r: r.original_p_value)
        adj = [r.adjusted_p_value for r in sorted_results]
        for i in range(1, len(adj)):
            assert adj[i] >= adj[i - 1] - 1e-10

    def test_metric_names_preserved(self):
        """Metric names should be attached to the correct results."""
        results = benjamini_hochberg(
            [0.01, 0.04, 0.10], metric_names=["ctr", "revenue", "sessions"]
        )
        assert results[0].metric_name == "ctr"
        assert results[1].metric_name == "revenue"
        assert results[2].metric_name == "sessions"


class TestApplyCorrection:
    """Tests for the apply_correction dispatcher function."""

    def test_dispatches_bonferroni(self):
        """apply_correction with method='bonferroni' should use Bonferroni."""
        p = [0.01, 0.04]
        results = apply_correction(p, method="bonferroni")
        assert results[0].method == "bonferroni"
        # Verify adjustment is correct (p * m)
        assert results[0].adjusted_p_value == pytest.approx(0.02)
        assert results[1].adjusted_p_value == pytest.approx(0.08)

    def test_dispatches_holm(self):
        """apply_correction with method='holm' should use Holm."""
        p = [0.01, 0.04]
        results = apply_correction(p, method="holm")
        assert results[0].method == "holm"

    def test_dispatches_benjamini_hochberg(self):
        """apply_correction with method='benjamini_hochberg' should use BH."""
        p = [0.01, 0.04]
        results = apply_correction(p, method="benjamini_hochberg")
        assert results[0].method == "benjamini_hochberg"

    def test_fdr_bh_alias(self):
        """'fdr_bh' should be an alias for 'benjamini_hochberg'."""
        p = [0.01, 0.04]
        bh1 = apply_correction(p, method="fdr_bh")
        bh2 = apply_correction(p, method="benjamini_hochberg")
        for r1, r2 in zip(bh1, bh2):
            assert r1.adjusted_p_value == r2.adjusted_p_value

    def test_bh_alias(self):
        """'bh' should also be an alias for 'benjamini_hochberg'."""
        p = [0.01, 0.04]
        bh1 = apply_correction(p, method="bh")
        bh2 = apply_correction(p, method="benjamini_hochberg")
        for r1, r2 in zip(bh1, bh2):
            assert r1.adjusted_p_value == r2.adjusted_p_value

    def test_unknown_method_raises(self):
        """Unknown correction method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_correction([0.01], method="unknown")

    def test_default_method_is_holm(self):
        """Default method should be Holm."""
        p = [0.01, 0.04]
        results = apply_correction(p)
        assert results[0].method == "holm"

    def test_passes_alpha_through(self):
        """Custom alpha should be passed to the underlying method."""
        p = [0.01, 0.04]
        results_strict = apply_correction(p, method="bonferroni", alpha=0.01)
        results_lenient = apply_correction(p, method="bonferroni", alpha=0.10)
        # With alpha=0.01: 0.01*2=0.02 > 0.01, not significant
        assert not results_strict[0].is_significant
        # With alpha=0.10: 0.01*2=0.02 < 0.10, significant
        assert results_lenient[0].is_significant

    def test_passes_metric_names_through(self):
        """Metric names should be forwarded to the underlying method."""
        p = [0.01, 0.04]
        results = apply_correction(
            p, method="bonferroni", metric_names=["ctr", "revenue"]
        )
        assert results[0].metric_name == "ctr"
        assert results[1].metric_name == "revenue"


class TestCorrectedResultStructure:
    """Tests for the CorrectedResult dataclass structure."""

    def test_result_is_corrected_result(self):
        """Results should be CorrectedResult instances."""
        results = bonferroni([0.01])
        assert isinstance(results[0], CorrectedResult)

    def test_frozen_dataclass(self):
        """CorrectedResult should be immutable."""
        results = bonferroni([0.01])
        with pytest.raises(AttributeError):
            results[0].adjusted_p_value = 0.5

    def test_all_fields_present(self):
        """All expected fields should be present."""
        result = bonferroni([0.01], metric_names=["ctr"])[0]
        assert hasattr(result, "original_p_value")
        assert hasattr(result, "adjusted_p_value")
        assert hasattr(result, "is_significant")
        assert hasattr(result, "method")
        assert hasattr(result, "metric_name")
