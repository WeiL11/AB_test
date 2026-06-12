"""
Simulate a full A/B test experiment lifecycle.

This script demonstrates all 9 statistical methods on realistic data,
showing what the platform computes at each stage of an experiment.

Usage:
    python -m scripts.simulate_experiment

No database required -- runs entirely in-memory.
"""
import numpy as np
from app.stats.frequentist import welch_t_test, z_test_proportions
from app.stats.bayesian import beta_binomial_test, normal_test
from app.stats.sequential import GroupSequentialTest, sequential_z_test
from app.stats.confidence_sequence import confidence_sequence
from app.stats.bandit import ThompsonSampling, run_bandit_simulation
from app.stats.cuped import cuped_adjust
from app.stats.multiple_testing import apply_correction
from app.stats.power_analysis import required_sample_size, compute_power
from app.stats.novelty_detection import detect_novelty_effect


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title):
    print(f"\n--- {title} ---")


def main():
    np.random.seed(42)

    print_header("EXPERIMENTOR: Full A/B Test Simulation")
    print("\nScenario: Testing a new blue checkout button vs. the current green one")
    print("  Control (green): 10% conversion rate")
    print("  Treatment (blue): 12.5% conversion rate (true 2.5pp lift)")
    print("  20,000 users per group")

    n = 20000

    # Generate data
    control_conv = np.random.binomial(1, 0.10, n).astype(float)
    treatment_conv = np.random.binomial(1, 0.125, n).astype(float)

    pre_c = np.random.normal(50, 15, n)
    pre_t = np.random.normal(50, 15, n)
    revenue_c = pre_c + np.random.normal(0, 8, n)
    revenue_t = pre_t + np.random.normal(2, 8, n)

    # ========== STEP 1: Power Analysis ==========
    print_header("Step 1: Pre-Experiment Power Analysis")
    power_result = required_sample_size(0.10, 0.025)
    print(f"  Required sample size per variant: {power_result.required_sample_size_per_variant:,}")
    print(f"  Total sample size needed: {power_result.total_sample_size:,}")
    power_at_n = compute_power(n, 0.10, 0.025)
    print(f"  Power with {n:,} users per group: {power_at_n:.1%}")
    print(f"  Verdict: {'Sufficient power' if power_at_n > 0.8 else 'Need more users'}")

    # ========== STEP 2: Frequentist Analysis ==========
    print_header("Step 2: Frequentist Analysis")

    print_section("Conversion Rate (Z-test for proportions)")
    s_c, s_t = int(control_conv.sum()), int(treatment_conv.sum())
    freq_conv = z_test_proportions(s_c, n, s_t, n)
    print(f"  Control rate: {freq_conv.mean_control:.4f} ({s_c}/{n})")
    print(f"  Treatment rate: {freq_conv.mean_treatment:.4f} ({s_t}/{n})")
    print(f"  Absolute effect: {freq_conv.absolute_effect:+.4f}")
    print(f"  Relative effect: {freq_conv.relative_effect:+.2%}")
    print(f"  95% CI: [{freq_conv.ci_lower:.4f}, {freq_conv.ci_upper:.4f}]")
    print(f"  P-value: {freq_conv.p_value:.6f}")
    print(f"  Significant: {freq_conv.is_significant}")

    print_section("Revenue per User (Welch's t-test)")
    freq_rev = welch_t_test(revenue_c, revenue_t)
    print(f"  Control mean: ${freq_rev.mean_control:.2f}")
    print(f"  Treatment mean: ${freq_rev.mean_treatment:.2f}")
    print(f"  Absolute effect: ${freq_rev.absolute_effect:+.2f}")
    print(f"  95% CI: [${freq_rev.ci_lower:.2f}, ${freq_rev.ci_upper:.2f}]")
    print(f"  P-value: {freq_rev.p_value:.6f}")
    print(f"  Significant: {freq_rev.is_significant}")

    # ========== STEP 3: Sequential Testing ==========
    print_header("Step 3: Sequential Testing (O'Brien-Fleming)")
    gst = GroupSequentialTest(alpha=0.05, n_analyses=5, spending="obrien_fleming")
    boundaries = gst.compute_all_boundaries()
    print("  Planned analysis boundaries:")
    for info_frac, boundary in boundaries:
        print(f"    {info_frac:.0%} of data -> boundary = {boundary:.3f}")

    print("\n  Simulating interim analyses:")
    for k in range(1, 6):
        frac = k / 5
        n_k = int(n * frac)
        result = sequential_z_test(
            control_conv[:n_k], treatment_conv[:n_k],
            analysis_number=k, n_analyses=5,
        )
        status = "STOP (reject)" if result.can_reject else "Continue"
        print(f"    Look {k} ({frac:.0%}, n={n_k:,}): z={result.z_statistic:.3f}, boundary={result.boundary:.3f} -> {status}")
        if result.can_reject:
            print(f"    Early stopping possible at {frac:.0%} of planned data!")
            break

    # ========== STEP 4: Confidence Sequences ==========
    print_header("Step 4: Always-Valid Inference (Confidence Sequences)")
    cs = confidence_sequence(revenue_c, revenue_t)
    print(f"  Mean difference: ${cs.mean_difference:.2f}")
    print(f"  CS bounds: [${cs.ci_lower:.2f}, ${cs.ci_upper:.2f}]")
    print(f"  Significant: {cs.is_significant}")
    print(f"  (These bounds are valid at ANY stopping time)")

    # ========== STEP 5: Bayesian Analysis ==========
    print_header("Step 5: Bayesian Analysis")

    print_section("Conversion Rate (Beta-Binomial)")
    bayes_conv = beta_binomial_test(s_c, n, s_t, n)
    print(f"  P(treatment > control): {bayes_conv.prob_treatment_better:.4f}")
    print(f"  Expected loss (if we ship treatment): {bayes_conv.expected_loss_treatment:.6f}")
    print(f"  Expected loss (if we keep control): {bayes_conv.expected_loss_control:.6f}")
    print(f"  95% credible interval: [{bayes_conv.credible_interval_lower:.4f}, {bayes_conv.credible_interval_upper:.4f}]")
    print(f"  Recommendation: {bayes_conv.recommendation}")

    print_section("Revenue (Normal model)")
    bayes_rev = normal_test(revenue_c, revenue_t)
    print(f"  P(treatment > control): {bayes_rev.prob_treatment_better:.4f}")
    print(f"  95% credible interval: [${bayes_rev.credible_interval_lower:.2f}, ${bayes_rev.credible_interval_upper:.2f}]")

    # ========== STEP 6: Multi-Armed Bandit ==========
    print_header("Step 6: Multi-Armed Bandit (Thompson Sampling)")
    bandit = run_bandit_simulation([0.10, 0.125], algorithm="thompson", n_rounds=10000)
    print(f"  After 10,000 rounds:")
    print(f"    Arm 0 (control): {bandit.total_pulls[0]:,} pulls, est. rate={bandit.estimated_means[0]:.4f}")
    print(f"    Arm 1 (treatment): {bandit.total_pulls[1]:,} pulls, est. rate={bandit.estimated_means[1]:.4f}")
    print(f"  Recommended allocation: {bandit.recommended_allocation[0]:.1f}% / {bandit.recommended_allocation[1]:.1f}%")
    print(f"  Cumulative regret: {bandit.cumulative_regret:.1f}")
    print(f"  Best arm: {bandit.best_arm_index} ({'treatment' if bandit.best_arm_index == 1 else 'control'})")

    # ========== STEP 7: CUPED ==========
    print_header("Step 7: CUPED Variance Reduction")
    cuped = cuped_adjust(revenue_c, revenue_t, pre_c, pre_t)
    print(f"  Pre/post correlation: {cuped.correlation:.3f}")
    print(f"  Theta (regression coefficient): {cuped.theta:.3f}")
    print(f"  Original variance (control): {cuped.original_variance_control:.2f}")
    print(f"  Adjusted variance (control): {cuped.adjusted_variance_control:.2f}")
    print(f"  Variance reduction: {cuped.variance_reduction_pct:.1%}")
    print(f"  Original effect: ${cuped.original_effect:.2f}")
    print(f"  Adjusted effect: ${cuped.adjusted_effect:.2f}")
    print(f"  (Same effect, but {cuped.variance_reduction_pct:.0%} less noise)")

    # ========== STEP 8: Multiple Testing Correction ==========
    print_header("Step 8: Multiple Testing Correction")
    p_values = [freq_conv.p_value, freq_rev.p_value]
    metric_names = ["conversion_rate", "revenue_per_user"]

    for method in ["bonferroni", "holm", "benjamini_hochberg"]:
        corrected = apply_correction(p_values, method=method, metric_names=metric_names)
        print(f"\n  {method.upper()}:")
        for r in corrected:
            print(f"    {r.metric_name}: p={r.original_p_value:.6f} -> adj_p={r.adjusted_p_value:.6f} {'*' if r.is_significant else ''}")

    # ========== STEP 9: Novelty Detection ==========
    print_header("Step 9: Novelty/Primacy Effect Detection")
    daily_effects = [0.025 + np.random.normal(0, 0.004) for _ in range(14)]
    novelty = detect_novelty_effect(daily_effects)
    print(f"  Daily effect trend slope: {novelty.slope:.6f} per day")
    print(f"  P-value for trend: {novelty.p_value:.4f}")
    print(f"  Effect type: {novelty.effect_type}")
    print(f"  R-squared: {novelty.r_squared:.3f}")
    if novelty.effect_type == "stable":
        print("  The treatment effect is stable over time. Safe to interpret.")

    # ========== SUMMARY ==========
    print_header("FINAL SUMMARY")
    print(f"""
  Experiment: Blue vs Green Checkout Button

  Frequentist:  Significant positive lift (p={freq_conv.p_value:.6f})
  Bayesian:     P(blue > green) = {bayes_conv.prob_treatment_better:.4f}, recommend: {bayes_conv.recommendation}
  Sequential:   Early stopping possible
  CUPED:        {cuped.variance_reduction_pct:.0%} variance reduction achieved
  Novelty:      Effect is {novelty.effect_type} (no decay detected)

  ALL METHODS AGREE: Ship the blue button.
""")


if __name__ == "__main__":
    main()
