# Decision Tree: What To Do When Methods Disagree

In the simulation, all 10 methods agreed. In real experiments, they often don't. This document maps every realistic conflict pattern to a concrete decision.

---

## The Methods at a Glance

Not all 10 methods vote equally. They serve different roles:

```
DECISION MAKERS (these directly say ship or don't ship)
  ├── Frequentist test        → "Is the effect statistically significant?"
  ├── Bayesian test           → "What's the probability B is better, and what's the risk?"
  └── Guardrail metrics       → "Did we break anything important?"

ADJUSTERS (these modify how you interpret the decision makers)
  ├── Sequential / CS         → "Can I trust the result given when I stopped?"
  ├── Multiple testing        → "Can I trust the result given how many metrics I tested?"
  ├── CUPED                   → "Is the effect estimate less noisy with variance reduction?"
  └── Novelty detection       → "Will this effect last?"

INFORMATIONAL (these guide strategy, not the ship decision)
  ├── Power analysis          → "Did I have enough data to detect a real effect?"
  ├── Bandits                 → "How should I allocate traffic going forward?"
  └── Segment analysis        → "Does the effect differ across user groups?"
```

---

## The Decision Tree

Start here after running all methods on your experiment.

```
┌─────────────────────────────────────────────────────────────┐
│                    START HERE                                │
│          Did any GUARDRAIL metric regress?                   │
└─────────────────┬──────────────────────┬────────────────────┘
                  │                      │
                 YES                     NO
                  │                      │
                  ▼                      ▼
        ┌─────────────────┐    ┌─────────────────────────────┐
        │   DON'T SHIP    │    │ Is the novelty detector     │
        │                 │    │ showing a decaying effect?   │
        │ A guardrail     │    └──────────┬─────────┬────────┘
        │ regression is a │               │         │
        │ hard veto, even │             DECAY     STABLE
        │ if all other    │               │      or GROWING
        │ methods say     │               │         │
        │ "ship".         │               ▼         ▼
        └─────────────────┘    ┌──────────────┐  ┌──────────────────────┐
                               │ WAIT & RERUN │  │ Did you peek at      │
                               │              │  │ results before the   │
                               │ Run 1-2 more │  │ planned end date?    │
                               │ weeks. If    │  └────┬────────────┬────┘
                               │ effect is    │       │            │
                               │ still there, │      YES           NO
                               │ re-enter     │       │            │
                               │ this tree.   │       ▼            ▼
                               └──────────────┘  ┌─────────┐  ┌──────────────────┐
                                                 │ Did you │  │ Do Frequentist   │
                                                 │ use     │  │ and Bayesian     │
                                                 │ Seq or  │  │ agree?           │
                                                 │ CS?     │  └───┬──────────┬───┘
                                                 └──┬───┬──┘      │          │
                                                   YES  NO        │          │
                                                    │    │       YES         NO
                                                    ▼    ▼        │          │
                                              ┌────────┐ │        ▼          ▼
                                              │ Trust  │ │   ┌────────┐  (See conflict
                                              │ the    │ │   │ SHIP   │   table below)
                                              │ seq/CS │ │   │ or     │
                                              │ result │ │   │ DON'T  │
                                              └────────┘ │   │ SHIP   │
                                                         │   └────────┘
                                                         ▼
                                                  ┌──────────────┐
                                                  │ YOUR RESULT  │
                                                  │ IS INVALID   │
                                                  │              │
                                                  │ You peeked   │
                                                  │ without      │
                                                  │ correction.  │
                                                  │ Re-run with  │
                                                  │ sequential   │
                                                  │ or extend    │
                                                  │ experiment.  │
                                                  └──────────────┘
```

---

## Conflict Table: What To Do When Specific Methods Disagree

### Conflict 1: Frequentist says significant, Bayesian says "keep running"

```
Frequentist:  p = 0.03  → significant
Bayesian:     P(B>A) = 0.91, expected loss = 0.003  → below threshold, keep running
```

**What's happening:** The frequentist test found enough evidence to reject H0, but the Bayesian model isn't confident enough (P(B>A) < 0.95) or the expected loss is too high.

**Who to trust:** The Bayesian result, for two reasons:
1. The Bayesian thresholds (95% probability, 0.1% loss) are deliberately more conservative than frequentist alpha=0.05.
2. Expected loss directly quantifies business risk. A significant p-value doesn't.

**Decision:** Keep running. The effect is probably real but the risk of being wrong is still too high.

---

### Conflict 2: Bayesian says "ship", Frequentist says not significant

```
Frequentist:  p = 0.08  → not significant at alpha=0.05
Bayesian:     P(B>A) = 0.96, expected loss = 0.0003  → ship
```

**What's happening:** The Bayesian posterior is concentrated enough to be confident, but the frequentist CI just barely crosses zero.

**Who to trust:** Depends on your organization's statistical framework:
- **Regulated / conservative environment:** Respect the frequentist result. Keep running until p < 0.05.
- **Move-fast product team:** The Bayesian result is actionable. P(B>A) = 96% with tiny expected loss is a strong signal.

**Decision:** If in doubt, keep running. Both will converge with more data.

---

### Conflict 3: Primary metric significant, but multiple testing correction kills it

```
Before correction:  conversion p=0.04, revenue p=0.03, engagement p=0.12
After Holm:         conversion adj_p=0.08, revenue adj_p=0.06, engagement adj_p=0.12
```

**What's happening:** Individually, conversion and revenue are significant. After adjusting for testing 3 metrics, neither survives.

**Who to trust:** Depends on how many metrics you have:
- **2-5 metrics (primary + guardrails):** Respect the Holm correction. You have few tests, and FWER control matters.
- **10+ metrics (many secondary/exploratory):** Use Benjamini-Hochberg instead of Holm. FDR control is the right tradeoff -- you accept some false discoveries among many tests.
- **1 pre-registered primary metric:** You don't need correction for the primary metric at all. Correction is for the *family* of tests. If you declared one primary metric before the experiment, test it at alpha=0.05 without adjustment.

**Decision:** If your single pre-registered primary metric is significant before correction, you can ship. Apply correction to secondary/exploratory metrics.

---

### Conflict 4: Significant effect, but novelty detection shows decay

```
Frequentist:  p = 0.001, +15% lift
Novelty:      slope = -0.8%/day, p = 0.02  → decaying effect
```

**What's happening:** The effect is real *right now*, but it's shrinking every day. Users are reacting to the novelty of change, not the quality of the new design.

**Who to trust:** The novelty detector overrides the significance test on the question of "should we ship for the long term?"

**Decision:**
1. **Don't ship yet.** The current lift is inflated.
2. Run 1-2 more weeks until the effect stabilizes.
3. Re-analyze. If the stabilized effect is still significant, ship.
4. If the effect decays to zero, don't ship -- it was novelty, not improvement.

---

### Conflict 5: CUPED-adjusted result disagrees with unadjusted result

```
Unadjusted:    effect = +$1.20,  p = 0.08  → not significant
CUPED-adjusted: effect = +$1.15,  p = 0.01  → significant
```

**What's happening:** CUPED reduced the noise (variance) without changing the effect size much. With less noise, the same effect became detectable.

**Who to trust:** The CUPED-adjusted result, as long as:
- The covariate (pre-experiment data) was measured *before* the experiment started.
- The correlation between covariate and metric is real (rho > 0.3).

CUPED is provably unbiased. It doesn't inflate effects. It just removes predictable user-level noise. This is the equivalent of having a larger sample size.

**Decision:** Trust the CUPED result. Ship.

---

### Conflict 6: Sequential test says "stop early", fixed-horizon test says "not yet"

```
Sequential (at 40% of data):  z = 3.2, boundary = 3.1  → reject, stop early
Fixed-horizon (at 40%):       p = 0.07  → not significant
```

**What's happening:** The sequential boundary at 40% is stricter than alpha=0.05, but the z-statistic cleared it. The fixed-horizon test at 40% of data doesn't have enough power yet.

**Who to trust:** The sequential result. That's the whole point of sequential testing -- it's designed to give valid answers at interim looks. The fixed-horizon test at 40% is *underpowered by design* (you planned for 100% of data).

**Decision:** You can stop early. The sequential boundary controls the overall Type I error at exactly 5% across all planned looks.

---

### Conflict 7: Overall effect is null, but segment analysis finds a subgroup effect

```
Overall:   effect = +0.2%, p = 0.71  → not significant
Mobile:    effect = +8.3%, p = 0.003  → significant
Desktop:   effect = -5.1%, p = 0.02  → significant
```

**What's happening:** The treatment helps mobile users and hurts desktop users. The overall average washes out to near zero.

**Who to trust:** This is a real finding, but treat it with caution:
1. **Was the segment pre-registered?** If you declared "we'll look at mobile vs. desktop" before the experiment, this is credible.
2. **Was it discovered post-hoc?** If you sliced by 20 segments and found one significant result, it's likely a false discovery (see Method 7, multiple testing).
3. **Does the Cochran's Q test confirm heterogeneity?** If Q is significant, the effects genuinely differ across segments.

**Decision:**
- If pre-registered segment + significant Q-test → Ship for mobile only (or customize the experience by platform).
- If post-hoc discovery → Run a follow-up experiment specifically on mobile users to confirm.

---

### Conflict 8: Bandit says one arm is best, but frequentist test is not significant

```
Thompson Sampling:  98% traffic to arm B after 5,000 rounds
Frequentist:        p = 0.12  → not significant
```

**What's happening:** The bandit quickly learned that B converts better and routed traffic accordingly. But because the bandit sends almost all traffic to B, the control group (A) has very few observations, making the frequentist test underpowered.

**This is expected.** Bandits and frequentist tests have fundamentally different goals:
- Bandit: Minimize regret *during* the experiment (maximize revenue now).
- Frequentist: Maximize statistical certainty *after* the experiment.

**Decision:** Don't use the frequentist p-value from a bandit experiment -- the unequal sample sizes violate its assumptions. Instead:
- Use the bandit's allocation as the decision: if it's sending 98% to B, B is almost certainly better.
- If you need formal statistical significance, run a separate fixed-allocation experiment after the bandit identifies the likely winner.

---

### Conflict 9: Power analysis says underpowered, but you got a significant result

```
Power analysis:  Need 10,000 per group, you have 3,000
Frequentist:     p = 0.04  → significant
```

**What's happening:** You don't have enough data to reliably detect the *minimum* effect you planned for, but you got a significant result anyway.

**Who to trust:** The significant result is valid -- underpowered experiments can still detect large effects. But be cautious:
1. **The effect size is probably inflated.** In underpowered studies, the only effects that clear the significance bar are the ones that happen to be large by random chance ("winner's curse" / Type M error).
2. **The true effect is likely smaller** than what you measured.

**Decision:** Treat the direction as real, but don't trust the magnitude. If the effect size matters for your decision (e.g., "is this a 2% lift or a 10% lift?"), keep running until you reach the planned sample size.

---

## Quick Reference: Priority Order

When methods conflict, resolve in this order:

```
Priority 1: GUARDRAILS
  └─ Any guardrail regression → DON'T SHIP (hard veto, overrides everything)

Priority 2: VALIDITY CHECKS
  ├─ Did you peek without sequential/CS correction? → Result is INVALID
  └─ Is the novelty detector showing decay? → WAIT for effect to stabilize

Priority 3: STATISTICAL SIGNIFICANCE
  ├─ Frequentist + Bayesian agree → Follow their recommendation
  ├─ Bayesian says ship, Frequentist borderline → Probably safe to ship
  ├─ Frequentist says significant, Bayesian says keep running → Keep running
  └─ CUPED-adjusted significant, unadjusted not → Trust CUPED

Priority 4: CONTEXT & NUANCE
  ├─ Segment effects found → Confirm if pre-registered or post-hoc
  ├─ Bandit vs. frequentist disagree → Don't mix their conclusions
  └─ Underpowered but significant → Direction is real, magnitude is inflated
```

---

## One-Page Flowchart Summary

```
                         ┌──────────────┐
                         │  Guardrail   │
                         │  regressed?  │
                         └───┬──────┬───┘
                            YES     NO
                             │      │
                             ▼      ▼
                        DON'T   ┌──────────┐
                        SHIP    │ Novelty  │
                                │ decay?   │
                                └──┬────┬──┘
                                 YES    NO
                                  │     │
                                  ▼     ▼
                              WAIT   ┌──────────┐
                              &      │ Peeked   │
                              RERUN  │ without  │
                                     │ seq/CS?  │
                                     └──┬────┬──┘
                                      YES    NO
                                       │     │
                                       ▼     ▼
                                   INVALID ┌─────────────────┐
                                           │ Freq + Bayesian │
                                           │ agree?          │
                                           └──┬──────────┬───┘
                                             YES         NO
                                              │          │
                                              ▼          ▼
                                         FOLLOW     KEEP RUNNING
                                         THEIR      (collect more
                                         ANSWER     data until they
                                                    converge)
```
