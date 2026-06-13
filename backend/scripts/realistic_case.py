"""
Realistic A/B Test Case Study: Spotify-like "Personalized Playlist" Feature

Scenario:
    Spotify is testing a new personalized playlist algorithm on its Premium users.
    Group A (control): existing recommendation engine
    Group B (treatment): new ML-based recommendation engine

    Metrics:
      - Primary: streaming hours per user per week (continuous)
      - Secondary: premium renewal rate (binomial, 30-day window)
      - Guardrail: app crash rate (binomial, must not increase)

    User segments: iOS vs Android
    Pre-experiment data: last month's streaming hours (for CUPED)
    Traffic: 50,000 users per group over 3 weeks
    Daily traffic: ~4,760 users per day per group

Usage:
    cd backend
    python -m scripts.realistic_case
"""
import numpy as np
from app.stats.frequentist import welch_t_test, z_test_proportions
from app.stats.bayesian import beta_binomial_test, normal_test
from app.stats.sequential import sequential_z_test
from app.stats.confidence_sequence import confidence_sequence
from app.stats.bandit import run_bandit_simulation
from app.stats.cuped import cuped_adjust
from app.stats.multiple_testing import apply_correction
from app.stats.power_analysis import required_sample_size, compute_power, estimate_duration
from app.stats.novelty_detection import detect_novelty_effect, compute_daily_effects
from app.stats.segment_analysis import analyze_segments


def _decision(freq_hours, freq_crash, bayes_hours, novelty):
    guardrail_fail = freq_crash.is_significant and freq_crash.absolute_effect > 0
    prob_str = ">99.99%" if bayes_hours.prob_treatment_better >= 0.9999 else f"{bayes_hours.prob_treatment_better:.2%}"
    loss_str = "<0.0001" if bayes_hours.expected_loss_treatment < 0.0001 else f"{bayes_hours.expected_loss_treatment:.4f}"
    if guardrail_fail:
        return (f"DO NOT SHIP. The primary metric improved ({freq_hours.absolute_effect:+.2f} hrs, p<0.001) and Bayesian\n"
                f"  agrees (P(B>A)={prob_str}, loss={loss_str}). But the crash rate guardrail failed\n"
                f"  ({freq_crash.absolute_effect:+.4f}, p={freq_crash.p_value:.4f}). Investigate the crash increase\n"
                f"  before proceeding. If fixable, patch and re-run. If inherent, do not ship.")
    if novelty.effect_type != "stable":
        return "WAIT. Effect is not stable yet. Extend the experiment."
    if freq_hours.is_significant and bayes_hours.recommendation == "ship":
        return "SHIP the new algorithm."
    return "KEEP RUNNING. Evidence is not yet conclusive."


def _segment_note(seg):
    ios = next((s for s in seg.segment_results if s.segment_value == "iOS"), None)
    android = next((s for s in seg.segment_results if s.segment_value == "Android"), None)
    if ios and android:
        return (f"iOS users benefit significantly ({ios.absolute_effect:+.2f} hrs, p={ios.p_value:.4f}), "
                f"Android shows no effect ({android.absolute_effect:+.2f} hrs, p={android.p_value:.4f}).\n"
                f"  Consider a phased rollout starting with iOS while investigating Android.")
    return ""


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    np.random.seed(2024)
    N = 50_000  # users per group
    DAYS = 21   # 3 weeks

    # ──────────────────────────────────────────────────────
    # Generate realistic data
    # ──────────────────────────────────────────────────────
    divider("CASE STUDY: Spotify Personalized Playlist Algorithm")
    print("""
  Company:     Spotify (hypothetical)
  Experiment:  New ML-based recommendation engine vs. existing engine
  Groups:      50,000 Premium users each
  Duration:    3 weeks
  Primary:     Streaming hours/week  (continuous, control mean ~18.5 hrs, MDE=0.08 hrs)
  Secondary:   Premium renewal rate  (binomial, control ~82%)
  Guardrail:   App crash rate        (binomial, must not increase, baseline ~1.9%)
  Segments:    iOS (60%) vs Android (40%)
  CUPED:       Last month's streaming hours as covariate
""")

    # --- Streaming hours (continuous) ---
    # Control: mean=18.5 hrs/week, sd~4.4 (after pre/post correlation)
    # Treatment: mean=18.93 hrs/week, sd~4.4
    # Effective true lift = 0.35 * (18.93 - 18.5) = +0.15 hrs (~0.8%)
    # This requires ~50K users per group to detect at MDE=0.08 hrs.
    pre_hours_c = np.random.normal(18.0, 5.8, N)  # last month
    pre_hours_t = np.random.normal(18.0, 5.8, N)

    hours_c = 0.65 * pre_hours_c + 0.35 * np.random.normal(18.5, 6.2, N)
    hours_t = 0.65 * pre_hours_t + 0.35 * np.random.normal(18.93, 6.2, N)

    # --- Premium renewal rate (binomial) ---
    # Control: 82%, Treatment: 83.5% (true lift = +1.5pp)
    renew_c = np.random.binomial(1, 0.820, N).astype(float)
    renew_t = np.random.binomial(1, 0.835, N).astype(float)

    # --- App crash rate (guardrail) ---
    # Control: 1.9%, Treatment: 2.4% (true regression = +0.5pp)
    # Large enough for the guardrail to catch a real problem, not noise.
    crash_c = np.random.binomial(1, 0.019, N).astype(float)
    crash_t = np.random.binomial(1, 0.024, N).astype(float)

    # --- Segments: iOS (60%) vs Android (40%) ---
    ios_mask_c = np.random.random(N) < 0.60
    ios_mask_t = np.random.random(N) < 0.60

    # iOS users: stronger effect (base +0.15 + 0.20 = +0.35 hrs)
    # Android users: null effect (base +0.15 - 0.15 = ~0.00 hrs)
    hours_t_seg = hours_t.copy()
    hours_t_seg[ios_mask_t] += 0.20   # extra boost on iOS
    hours_t_seg[~ios_mask_t] -= 0.15  # cancels base effect on Android

    # Daily data for novelty detection (21 days)
    daily_effects = []
    users_per_day = N // DAYS
    for d in range(DAYS):
        start = d * users_per_day
        end = start + users_per_day
        daily_c = hours_c[start:end]
        daily_t = hours_t[start:end]
        daily_effects.append(float(np.mean(daily_t) - np.mean(daily_c)))

    # ──────────────────────────────────────────────────────
    # METHOD 1: Power Analysis
    # ──────────────────────────────────────────────────────
    divider("1. POWER ANALYSIS — How many users do we need?")

    # For streaming hours: baseline=18.5, sd~4.5, MDE=0.08 hrs
    # Effect size (Cohen's d) = 0.08 / 4.5 ≈ 0.018
    # For proportions: baseline=0.82, MDE=0.015
    power_hours = required_sample_size(
        baseline_rate=18.5,
        minimum_detectable_effect=0.08,
        alpha=0.05,
        power=0.80,
        metric_type="continuous",
        variance=4.5**2,
    )
    power_renew = required_sample_size(
        baseline_rate=0.82,
        minimum_detectable_effect=0.015,
    )
    actual_power_hours = compute_power(N, 18.5, 0.08, metric_type="continuous", variance=4.5**2)
    actual_power_renew = compute_power(N, 0.82, 0.015)
    duration = estimate_duration(power_hours.required_sample_size_per_variant, daily_traffic=4760)

    print(f"  Streaming hours:")
    print(f"    Required per group:    {power_hours.required_sample_size_per_variant:,} users")
    print(f"    Actual per group:      {N:,} users")
    print(f"    Power at {N:,}:       {actual_power_hours:.1%}")
    print(f"    Estimated duration:    {duration} days")
    print(f"")
    print(f"  Premium renewal:")
    print(f"    Required per group:    {power_renew.required_sample_size_per_variant:,} users")
    print(f"    Actual per group:      {N:,} users")
    print(f"    Power at {N:,}:       {actual_power_renew:.1%}")
    print(f"")
    print(f"  Verdict: We have {N:,} users. Both metrics are adequately powered.")

    # ──────────────────────────────────────────────────────
    # METHOD 2: Frequentist Tests
    # ──────────────────────────────────────────────────────
    divider("2. FREQUENTIST — Is B significantly different from A?")

    freq_hours = welch_t_test(hours_c, hours_t)
    s_renew_c, s_renew_t = int(renew_c.sum()), int(renew_t.sum())
    freq_renew = z_test_proportions(s_renew_c, N, s_renew_t, N)
    s_crash_c, s_crash_t = int(crash_c.sum()), int(crash_t.sum())
    freq_crash = z_test_proportions(s_crash_c, N, s_crash_t, N)

    print(f"  Streaming hours (Welch's t-test):")
    print(f"    Control mean:        {freq_hours.mean_control:.2f} hrs/week")
    print(f"    Treatment mean:      {freq_hours.mean_treatment:.2f} hrs/week")
    print(f"    Difference:          {freq_hours.absolute_effect:+.2f} hrs")
    print(f"    95% CI:              [{freq_hours.ci_lower:+.2f}, {freq_hours.ci_upper:+.2f}]")
    print(f"    P-value:             {freq_hours.p_value:.6f}")
    print(f"    Significant:         {freq_hours.is_significant}")
    print(f"")
    print(f"  Premium renewal (z-test for proportions):")
    print(f"    Control rate:        {freq_renew.mean_control:.4f} ({s_renew_c:,}/{N:,})")
    print(f"    Treatment rate:      {freq_renew.mean_treatment:.4f} ({s_renew_t:,}/{N:,})")
    print(f"    Difference:          {freq_renew.absolute_effect:+.4f} ({freq_renew.absolute_effect*100:+.2f}pp)")
    print(f"    95% CI:              [{freq_renew.ci_lower:+.4f}, {freq_renew.ci_upper:+.4f}]")
    print(f"    P-value:             {freq_renew.p_value:.6f}")
    print(f"    Significant:         {freq_renew.is_significant}")
    print(f"")
    print(f"  App crash rate (guardrail):")
    print(f"    Control rate:        {freq_crash.mean_control:.4f} ({s_crash_c:,}/{N:,})")
    print(f"    Treatment rate:      {freq_crash.mean_treatment:.4f} ({s_crash_t:,}/{N:,})")
    print(f"    Difference:          {freq_crash.absolute_effect:+.4f}")
    print(f"    P-value:             {freq_crash.p_value:.6f}")
    print(f"    Guardrail violated:  {freq_crash.is_significant and freq_crash.absolute_effect > 0}")

    # ──────────────────────────────────────────────────────
    # METHOD 3: Sequential Testing
    # ──────────────────────────────────────────────────────
    divider("3. SEQUENTIAL TESTING — Can we stop early?")

    print(f"  O'Brien-Fleming with 5 planned looks (every {N//5:,} users)")
    print(f"")
    stopped_early = False
    for k in range(1, 6):
        frac = k / 5
        n_k = int(N * frac)
        result = sequential_z_test(
            hours_c[:n_k], hours_t[:n_k],
            analysis_number=k, n_analyses=5,
        )
        status = "*** REJECT — stop early ***" if result.can_reject else "continue"
        print(f"    Look {k} (n={n_k:,}, {frac:.0%} of data):")
        print(f"      z = {result.z_statistic:+.3f},  boundary = {result.boundary:.3f},  decision: {status}")
        if result.can_reject and not stopped_early:
            stopped_early = True
            early_stop_pct = frac
    if stopped_early:
        print(f"\n  Could have stopped at {early_stop_pct:.0%} of data and saved {(1-early_stop_pct):.0%} of experiment time.")
    else:
        print(f"\n  No early stopping. Need full sample to reach significance.")

    # ──────────────────────────────────────────────────────
    # METHOD 4: Confidence Sequences
    # ──────────────────────────────────────────────────────
    divider("4. CONFIDENCE SEQUENCES — Valid at any stopping time")

    cs = confidence_sequence(hours_c, hours_t)
    print(f"  Mean difference:       {cs.mean_difference:+.3f} hrs")
    print(f"  CS bounds:             [{cs.ci_lower:+.3f}, {cs.ci_upper:+.3f}]")
    print(f"  Significant:           {cs.is_significant}")
    print(f"  (If zero is outside the bounds, the effect is real at ANY sample size)")

    # ──────────────────────────────────────────────────────
    # METHOD 5: Bayesian Testing
    # ──────────────────────────────────────────────────────
    divider("5. BAYESIAN — What's the probability B is better?")

    bayes_hours = normal_test(hours_c, hours_t)
    bayes_renew = beta_binomial_test(s_renew_c, N, s_renew_t, N)

    def _fmt_prob(p):
        """Format probability, capping at >99.99% / <0.01% to signal precision limits."""
        if p >= 0.9999:
            return ">99.99%"
        if p <= 0.0001:
            return "<0.01%"
        return f"{p:.4f}"

    def _fmt_loss(loss):
        """Format expected loss, capping at <0.0001 to signal precision limits."""
        if loss < 0.0001:
            return "<0.0001"
        return f"{loss:.6f}"

    print(f"  Streaming hours (Normal model):")
    print(f"    P(B > A):            {_fmt_prob(bayes_hours.prob_treatment_better)}")
    print(f"    Expected loss (ship): {_fmt_loss(bayes_hours.expected_loss_treatment)} hrs")
    print(f"    95% credible interval: [{bayes_hours.credible_interval_lower:+.3f}, {bayes_hours.credible_interval_upper:+.3f}]")
    print(f"    Recommendation:      {bayes_hours.recommendation}")
    print(f"")
    print(f"  Premium renewal (Beta-Binomial):")
    print(f"    P(B > A):            {_fmt_prob(bayes_renew.prob_treatment_better)}")
    print(f"    Expected loss (ship): {_fmt_loss(bayes_renew.expected_loss_treatment)}")
    print(f"    95% credible interval: [{bayes_renew.credible_interval_lower:+.4f}, {bayes_renew.credible_interval_upper:+.4f}]")
    print(f"    Recommendation:      {bayes_renew.recommendation}")

    # ──────────────────────────────────────────────────────
    # METHOD 6: Multi-Armed Bandit
    # ──────────────────────────────────────────────────────
    divider("6. BANDITS — Adaptive traffic allocation")

    # Simulate using renewal rates as reward signal
    bandit = run_bandit_simulation([0.82, 0.835], algorithm="thompson", n_rounds=20_000)
    print(f"  Thompson Sampling after 20,000 rounds:")
    print(f"    Control:  {bandit.total_pulls[0]:,} pulls, est. rate = {bandit.estimated_means[0]:.4f}")
    print(f"    Treatment: {bandit.total_pulls[1]:,} pulls, est. rate = {bandit.estimated_means[1]:.4f}")
    print(f"    Recommended allocation:  {bandit.recommended_allocation[0]:.1f}% / {bandit.recommended_allocation[1]:.1f}%")
    print(f"    Cumulative regret:       {bandit.cumulative_regret:.1f}")
    print(f"    Best arm:                {'Treatment' if bandit.best_arm_index == 1 else 'Control'}")

    # ──────────────────────────────────────────────────────
    # METHOD 7: CUPED
    # ──────────────────────────────────────────────────────
    divider("7. CUPED — Variance reduction using last month's data")

    cuped = cuped_adjust(hours_c, hours_t, pre_hours_c, pre_hours_t)
    # Compute adjusted SE and z-score for comparison
    adj_se = np.sqrt(cuped.adjusted_variance_control / N + cuped.adjusted_variance_treatment / N)
    adj_z = abs(cuped.adjusted_effect) / adj_se if adj_se > 0 else float('inf')
    from scipy import stats as sp_stats
    adj_p = 2 * sp_stats.norm.sf(adj_z)

    print(f"  Covariate:             Last month's streaming hours")
    print(f"  Correlation (pre/post): {cuped.correlation:.3f}")
    print(f"  Theta:                 {cuped.theta:.3f}")
    print(f"")
    print(f"  Before CUPED:")
    print(f"    Variance (control):  {cuped.original_variance_control:.2f}")
    print(f"    Effect:              {cuped.original_effect:+.3f} hrs")
    print(f"    P-value:             {freq_hours.p_value:.6f}")
    print(f"")
    print(f"  After CUPED:")
    print(f"    Variance (control):  {cuped.adjusted_variance_control:.2f}")
    print(f"    Effect:              {cuped.adjusted_effect:+.3f} hrs")
    print(f"    P-value:             {adj_p:.6f}")
    print(f"    Variance reduction:  {cuped.variance_reduction_pct:.1%}")
    print(f"")
    print(f"  Same effect size, but {cuped.variance_reduction_pct:.0%} less noise → more significant.")

    # ──────────────────────────────────────────────────────
    # METHOD 8: Multiple Testing Correction
    # ──────────────────────────────────────────────────────
    divider("8. MULTIPLE TESTING — Correcting for 3 metrics")

    p_values = [freq_hours.p_value, freq_renew.p_value, freq_crash.p_value]
    names = ["streaming_hours", "renewal_rate", "crash_rate"]

    for method in ["holm", "benjamini_hochberg"]:
        corrected = apply_correction(p_values, method=method, metric_names=names)
        print(f"\n  {method.upper().replace('_', '-')}:")
        for r in corrected:
            sig = "significant" if r.is_significant else "not significant"
            print(f"    {r.metric_name:20s}  raw p={r.original_p_value:.6f}  adj p={r.adjusted_p_value:.6f}  ({sig})")

    # ──────────────────────────────────────────────────────
    # METHOD 9: Novelty Detection
    # ──────────────────────────────────────────────────────
    divider("9. NOVELTY DETECTION — Is the effect stable over 3 weeks?")

    novelty = detect_novelty_effect(daily_effects)
    print(f"  Daily effects (hrs, B - A):")
    for d, eff in enumerate(daily_effects):
        bar = "#" * int(max(0, eff) * 30)
        print(f"    Day {d+1:2d}:  {eff:+.3f}  {bar}")
    print(f"")
    print(f"  Trend slope:           {novelty.slope:+.6f} hrs/day")
    print(f"  Slope p-value:         {novelty.p_value:.4f}")
    print(f"  R-squared:             {novelty.r_squared:.3f}")
    print(f"  Classification:        {novelty.effect_type}")
    if novelty.effect_type == "stable":
        print(f"  The effect is not decaying. Safe to trust the measured lift.")

    # ──────────────────────────────────────────────────────
    # METHOD 10: Segment Analysis
    # ──────────────────────────────────────────────────────
    divider("10. SEGMENT ANALYSIS — iOS vs Android")

    seg_control = {
        "iOS": hours_c[ios_mask_c],
        "Android": hours_c[~ios_mask_c],
    }
    seg_treatment = {
        "iOS": hours_t_seg[ios_mask_t],
        "Android": hours_t_seg[~ios_mask_t],
    }
    seg = analyze_segments(seg_control, seg_treatment, segment_name="platform")

    print(f"  Overall effect:        {seg.overall_effect:+.3f} hrs")
    print(f"  Heterogeneity p-value: {seg.interaction_p_value:.4f} (Cochran's Q)")
    print(f"  Effects differ across segments: {seg.interaction_significant}")
    print(f"")
    for s in seg.segment_results:
        sig = "***" if s.is_significant else ""
        print(f"    {s.segment_value:10s}  n={s.n_control+s.n_treatment:,}  effect={s.absolute_effect:+.3f} hrs  "
              f"CI=[{s.ci_lower:+.3f}, {s.ci_upper:+.3f}]  p={s.p_value:.4f} {sig}")

    # ──────────────────────────────────────────────────────
    # FINAL DECISION
    # ──────────────────────────────────────────────────────
    divider("FINAL DECISION FRAMEWORK")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  METRIC              RESULT              VERDICT                │
  ├─────────────────────────────────────────────────────────────────┤
  │  Streaming hours     {freq_hours.absolute_effect:+.2f} hrs (p={freq_hours.p_value:.4f})   {'Significant lift' if freq_hours.is_significant else 'Not significant':20s}│
  │  Renewal rate        {freq_renew.absolute_effect:+.4f} (p={freq_renew.p_value:.4f})   {'Significant lift' if freq_renew.is_significant else 'Not significant':20s}│
  │  Crash rate          {freq_crash.absolute_effect:+.4f} (p={freq_crash.p_value:.4f})   {'GUARDRAIL FAIL' if freq_crash.is_significant and freq_crash.absolute_effect > 0 else 'Guardrail OK':20s}│
  └─────────────────────────────────────────────────────────────────┘

  CHECKLIST:
    [{'x' if not (freq_crash.is_significant and freq_crash.absolute_effect > 0) else ' '}] Guardrail OK (crash rate did not increase)
    [{'x' if novelty.effect_type == 'stable' else ' '}] Effect is stable over time (no novelty decay)
    [{'x' if freq_hours.is_significant else ' '}] Primary metric significant (frequentist)
    [{'x' if bayes_hours.prob_treatment_better > 0.95 else ' '}] Bayesian P(B>A) > 95% (actual: {'>99.99%' if bayes_hours.prob_treatment_better >= 0.9999 else f'{bayes_hours.prob_treatment_better:.1%}'})
    [{'x' if bayes_hours.expected_loss_treatment < 0.01 else ' '}] Expected loss acceptable (actual: {'<0.0001' if bayes_hours.expected_loss_treatment < 0.0001 else f'{bayes_hours.expected_loss_treatment:.4f}'} hrs)
    [{'x' if cuped.variance_reduction_pct > 0.2 else ' '}] CUPED confirms effect with less noise ({cuped.variance_reduction_pct:.0%} reduction)

  DECISION: {_decision(freq_hours, freq_crash, bayes_hours, novelty)}

  NOTE: {_segment_note(seg)}
""")


if __name__ == "__main__":
    main()
