# Experimentor

A/B testing platform with 4 core testing methods and 6 supporting techniques. Python/FastAPI backend, React frontend, 303 tests.

Group A sees the current experience. Group B sees the change. Both groups generate data. This platform determines whether the difference is real or noise.

## Case Study

Spotify (hypothetical) tests a new ML-based recommendation engine against the existing one. 50,000 Premium users per group, 3 weeks. Three metrics tracked: streaming hours (primary — does the new algorithm increase engagement?), premium renewal rate (secondary — does it help retention?), and app crash rate (guardrail — does it break anything?).

```bash
cd backend && python -m scripts.realistic_case   # run it yourself, no database needed
```

### Decision: Do Not Ship

The new algorithm improves streaming hours and renewal rates, but **increases app crash rate by +0.68 percentage points**. Crash rate is a guardrail metric — any statistically significant regression is a hard veto, regardless of how much the primary metric improved.

### Setup: Power Analysis (Method 8)

Before running the experiment, we compute how many users we need.

- **Minimum detectable effect (MDE):** 0.08 hrs/week — the smallest improvement worth detecting
- **Required sample size:** ~50,000 per group at 80% power, alpha=0.05
- **Actual sample size:** 50,000 per group — the experiment is properly powered (not over- or under-powered)
- **Why this matters:** If we used too few users, we might miss a real effect (false negative). If we used too many, we waste resources and every tiny difference looks significant. Here the sample matches the requirement.

### What the frequentist tests found (Method 1)

Welch's t-test for streaming hours, z-test for proportions on the other two.

- **Streaming hours:** Control 18.20 hrs/wk vs Treatment 18.33 hrs/wk → **+0.14 hrs (p<0.001)**. Statistically significant lift. The 95% CI is [+0.08, +0.19], meaning the true effect is somewhere in that range.
- **Premium renewal:** Control 81.9% vs Treatment 83.6% → **+1.68pp (p<0.001)**. Statistically significant lift.
- **App crash rate:** Control 1.81% vs Treatment 2.49% → **+0.68pp (p<0.001)**. Statistically significant regression. This is the guardrail metric — it blocks the launch.

### Could we have stopped early? Sequential testing (Method 3)

O'Brien-Fleming with 5 planned looks (every 10,000 users).

- **Look 1 (20% of data):** z=2.18, boundary=4.38 → continue. The boundary is strict early on to prevent premature decisions.
- **Look 2 (40% of data):** z=3.39, boundary=3.10 → **reject. Could stop here.**
- **Looks 3-5:** All reject with increasing confidence.
- **Bottom line:** We could have stopped at 40% of the data and saved 60% of the experiment's runtime. The effect was large enough to clear the strict early boundary.

### Is the result valid at any stopping time? Confidence sequences (Method 4)

Unlike fixed-horizon CIs, confidence sequences are valid no matter when you look.

- **Mean difference:** +0.14 hrs
- **CS bounds:** [+0.02, +0.25] — the entire interval is above zero
- **Why this matters:** If someone peeked at results on day 5 or day 18, the CS would still be valid. A regular 95% CI is only valid at the pre-planned end date. CS is wider (that's the price of anytime validity), but here both bounds are still above zero.

### What's the probability B is actually better? Bayesian testing (Method 2)

Instead of "reject or don't reject," Bayesian testing gives a direct probability.

- **P(B > A): >99.99%** — there is near-certainty that B streams more than A. (Displayed as >99.99% because 100,000 Monte Carlo draws found zero cases where A > B — the ">99.99%" signals this is a precision limit, not exact certainty.)
- **Expected loss if we ship B:** <0.0001 hrs — if B is actually worse, we lose almost nothing.
- **Recommendation:** Ship (but the guardrail overrides this).
- **Why both frequentist and Bayesian:** Frequentist says "the data is unlikely under the null hypothesis." Bayesian says "given the data, here's the probability B is better and the risk if we're wrong." They answer different questions. When both agree, you can be confident.

### Can we reduce noise? CUPED variance reduction (Method 5)

Uses last month's streaming hours as a covariate to remove predictable user-level noise.

- **Correlation between pre- and post-experiment data:** 0.87 — users who streamed a lot last month tend to stream a lot this month, regardless of the experiment.
- **Variance reduction:** 75% — CUPED removes that predictable component, leaving only the experiment's effect.
- **Effect before CUPED:** +0.14 hrs (p<0.001). **After CUPED:** +0.15 hrs (p<0.001).
- **Why this matters:** The effect size barely changed (CUPED is unbiased), but the p-value dropped because there's less noise. If the effect had been borderline (e.g., p=0.08 without CUPED), the noise reduction could push it to significant (e.g., p=0.01). It didn't change the conclusion here, but it increases confidence in the estimate.

### Do we need to correct for testing 3 metrics? Multiple testing correction (Method 6)

We tested 3 metrics simultaneously. Each test at alpha=0.05 means a 14% chance of at least one false positive across the family.

- **Holm correction:** All three metrics remain significant after adjustment. Streaming hours adj_p<0.001, renewal adj_p<0.001, crash rate adj_p<0.001.
- **Benjamini-Hochberg:** Same result — all significant.
- **Why this matters:** If only one metric had been borderline significant (e.g., p=0.04), correction might push it above 0.05 and change the decision. Here, all p-values are so small that correction doesn't change anything.

### Is the effect real or just novelty? Novelty detection (Method 7)

Users might behave differently simply because something changed, not because it's better. This wears off.

- **Daily effect over 21 days:** Ranges from -0.08 to +0.37 hrs, with no systematic trend.
- **Trend slope:** +0.002 hrs/day (p=0.55) — flat. Not decaying, not growing.
- **Classification:** Stable.
- **Why this matters:** If the slope had been negative (e.g., -0.05 hrs/day, p<0.05), the lift would be shrinking and might disappear after a month. We'd need to wait longer before deciding. Here, the effect is steady across all 3 weeks.

### Can we shift traffic to the winner? Multi-armed bandits (Method 9)

Thompson Sampling simulated with renewal rate as the reward signal.

- **After 20,000 rounds:** 95.4% of traffic routed to Treatment, 4.6% to Control.
- **Cumulative regret:** 24.0 — the total cost of exploring suboptimal arms.
- **Why this matters:** Bandits minimize regret *during* the experiment by shifting traffic to the winner early. But the unequal allocation makes the frequentist test unreliable (Control has very few observations). Use bandits for allocation decisions, not for statistical inference.

### Does the effect differ across user groups? Segment analysis (Method 10)

Split by platform: iOS (60% of users) vs Android (40%).

- **iOS:** +0.37 hrs (p<0.001) — strong, significant improvement.
- **Android:** -0.06 hrs (p=0.17) — no effect. Not significant.
- **Cochran's Q test for heterogeneity:** p<0.001 — the effects genuinely differ across platforms.
- **The overall +0.14 hrs is misleading.** It's a weighted average: 60% x (+0.37) + 40% x (-0.06) = +0.20. The "overall significant result" masks the fact that Android users get no benefit at all. A manager looking only at the top-line number would miss this.

### Recommended next steps

1. **Investigate the crash rate increase.** Is it caused by the new recommendation algorithm itself, or by a correlated deployment (e.g., a new media player component shipped alongside it)?
2. **If the crash issue is fixable,** patch it and re-run the experiment with the same 50K sample size.
3. **Consider an iOS-only rollout,** since Android shows no benefit. This also reduces the crash rate exposure to a smaller population while the team investigates.
4. **If crash rate is inherent to the new algorithm,** do not ship — no amount of engagement lift justifies a degraded user experience.

## Methods

4 core testing methods + 6 supporting techniques.

| | Method | Role | Question it answers |
|---|---|---|---|
| 1 | Welch's t-test | Core | Is the difference between A and B statistically significant? |
| 2 | Bayesian testing | Core | What is P(B > A), and what do I lose if I'm wrong? |
| 3 | Sequential testing | Core | Can I check results early without inflating false positives? |
| 4 | Confidence sequences | Core | Can I monitor continuously without pre-planning when to look? |
| 5 | CUPED | Adjust | Can I reduce noise using pre-experiment data? |
| 6 | Multiple testing | Adjust | How do I correct for testing many metrics at once? |
| 7 | Novelty detection | Adjust | Is the effect decaying over time? |
| 8 | Power analysis | Plan | How many users do I need before starting? |
| 9 | Multi-armed bandits | Allocate | Can I shift traffic to the winner during the experiment? |
| 10 | Segment analysis | Explore | Does the effect differ across user groups? |

## When Methods Disagree

| Priority | Situation | Action |
|---|---|---|
| 1 | Guardrail metric regressed | Don't ship. Hard veto. |
| 2 | Novelty decay detected | Wait for stabilization, re-analyze |
| 3 | Peeked without sequential/CS correction | Result invalid. Extend or re-run |
| 4 | Frequentist and Bayesian disagree | Collect more data. They converge |
| 5 | CUPED-adjusted vs unadjusted disagree | Trust CUPED (removes noise, not bias) |
| 6 | Bandit vs frequentist disagree | Don't mix. Use bandit for allocation, frequentist for inference |
| 7 | Segment effect found post-hoc | Treat as hypothesis. Confirm in follow-up experiment |

Default: collect more data.

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0 (async), PostgreSQL 16, Redis 7 |
| Stats | NumPy, SciPy — pure functions, no DB dependencies |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, TanStack Query, Recharts |
| Infra | Docker Compose, 303 tests (pytest) |

## Quick Start

```bash
git clone https://github.com/WeiL11/AB_test.git && cd AB_test

# Docker (full stack)
docker compose up -d
docker compose exec backend python -m scripts.seed_demo

# Or local
docker compose up -d postgres redis
cd backend && pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
pytest  # 303 tests

# Frontend
cd frontend && npm install && npm run dev
```

## API

All under `/api/v1`. Swagger at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/experiments/` | Create experiment with variants and metrics |
| `GET` | `/experiments/{id}/results` | Run analysis, return results |
| `POST` | `/events/batch` | Ingest up to 1,000 events |
| `GET` | `/assign/{exp_id}/{user_id}` | Deterministic variant assignment (SHA-256) |
| `POST` | `/power/sample-size` | Sample size calculator |

Full CRUD for experiments and events — see `/docs` for all 12 endpoints.

## Tests

303 tests, no mocking. Synthetic data with known properties verifies math correctness. Modules with multiple algorithms (bandits: 3, Bayesian: 2, multiple testing: 3, sequential: 2) have the most tests because each algorithm needs its own correctness and edge case coverage.

| Module | Count | Why | Covers |
|---|---|---|---|
| Bandits | 56 | 3 algorithms x ~15 + shared | Thompson/UCB1/Epsilon-Greedy, regret, simulation |
| Bayesian | 46 | 2 models x ~20 + validation | Posterior, P(B>A), expected loss, ROPE, recommendations |
| Multiple testing | 42 | 3 methods x ~10 + dispatcher | Bonferroni/Holm/BH, step logic, dispatching |
| Sequential | 35 | 2 boundaries + z-test | O'Brien-Fleming/Pocock, alpha spending, early stopping |
| Confidence sequences | 30 | 1 algorithm + over-time | Narrowing, detection, any-time validity |
| Novelty detection | 30 | 3 patterns + daily helper | Decay/growth/stable, weighted regression |
| CUPED | 20 | 1 method, many properties | Variance reduction, unbiasedness, effect preservation |
| Frequentist | 14 | 2 tests + chi-squared | Welch's t-test, z-test, chi-squared |
| Power analysis | 13 | 3 functions | Sample size, power curves, duration |
| Integration | 12 | 1 per method + agreement | End-to-end: all methods on same dataset |
| Assignment | 5 | 1 algorithm | Deterministic SHA-256 hashing, uniform distribution |

## Structure

```
backend/app/
  stats/        10 pure-function modules (frequentist, sequential, bayesian, ...)
  api/          REST endpoints
  services/     Business logic (analysis, assignment, ingestion)
  models/       SQLAlchemy ORM (Experiment, Variant, Metric, Event)

frontend/src/
  pages/        Dashboard, ExperimentPage, CreateExperiment, PowerAnalysis
  components/   CI charts, metrics table, power calculator
```

## Docs

- [Architecture](docs/ARCHITECTURE.md) — system design, schema, design decisions, scaling
- [Statistical Methods](docs/STATISTICAL_METHODS.md) — mathematical foundations for all 10 methods
- [Decision Tree](docs/DECISION_TREE.md) — conflict resolution for 9 disagreement scenarios

## License

MIT
