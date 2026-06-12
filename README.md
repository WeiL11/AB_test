# Experimentor

Full-stack A/B testing platform. 10 statistical methods, 303 tests, FastAPI + React.

---

## The Setup

Group A sees the current experience. Group B sees the change. Both groups generate data -- conversions, revenue, session duration. The question: is the difference between A and B real, or noise?

Each method below answers a different part of that question.

---

## What Each Method Tells You

**1. Welch's t-test** -- Compares the means of A and B. Tells you whether the observed difference is statistically significant without assuming both groups have equal variance. Outputs a p-value and confidence interval.

**2. Sequential testing** -- Same comparison, but lets you check results at pre-planned intervals without inflating false positives. Uses O'Brien-Fleming alpha-spending to keep the overall Type I error at exactly 5% across all looks.

**3. Confidence sequences** -- Same guarantee as sequential testing, but you can check at any time without pre-planning. Wider intervals early on; narrows as data accumulates.

**4. Bayesian testing** -- Instead of a p-value, gives you P(B > A) directly. Also computes expected loss: "if I ship B and I'm wrong, what's the cost?" Ship requires both P(B > A) > 0.95 and expected loss < 0.1%.

**5. Multi-armed bandits** -- Shifts traffic toward the better-performing variant during the experiment. Thompson Sampling, UCB1, and Epsilon-Greedy. Reduces opportunity cost but produces unbalanced samples.

**6. CUPED** -- Adjusts for pre-experiment user behavior to reduce variance. A user who spent $200 last month will spend more this month regardless of group assignment. Removing that noise is equivalent to doubling sample size.

**7. Multiple testing correction** -- If you test 20 metrics, there's a 64% chance one shows significance by chance. Holm and Benjamini-Hochberg adjust the threshold. Holm for 2-5 key metrics, BH for 10+ exploratory ones.

**8. Power analysis** -- Answers "how many users do I need?" before the experiment starts. Given baseline rate, minimum detectable effect, and desired power, computes required sample size and estimated duration.

**9. Novelty detection** -- Checks whether the effect is decaying over time. If lift is +15% in week 1 and +3% in week 3, users are reacting to change, not improvement.

**10. Segment analysis** -- Runs the test within each user segment independently. Detects when an overall null result hides opposite effects in subgroups (e.g., +12% mobile, -8% desktop). Uses Cochran's Q to test for heterogeneity.

---

## When Methods Disagree

In practice, they won't always align. Resolution order:

**1. Guardrail regression is a hard veto.** If any guardrail metric (latency, crash rate, revenue) regresses significantly, don't ship -- regardless of what the primary metric shows.

**2. Novelty decay overrides significance.** A decaying effect means the current estimate is inflated. Wait for the trend to stabilize, then re-analyze.

**3. Peeking without correction invalidates the result.** If you checked results multiple times without sequential testing or confidence sequences, the p-value is unreliable. Extend the experiment or re-run with proper stopping rules.

**4. When frequentist and Bayesian disagree,** it usually means the evidence is borderline. Frequentist significant but Bayesian uncertain (P(B>A) < 0.95) means the business risk is still too high. Bayesian confident but frequentist not significant (p = 0.07) means the effect is likely real but hasn't crossed the classical threshold. In both cases: collect more data. They converge with sufficient sample size.

**5. CUPED-adjusted results take precedence** over unadjusted when the covariate is valid (measured pre-experiment, correlation > 0.3). CUPED doesn't inflate effects -- it removes noise.

**6. Bandit results and frequentist results don't mix.** Bandits produce unbalanced allocations by design. Don't run a t-test on bandit-allocated data. Use bandit allocation as the decision signal, or run a separate fixed-split experiment for formal inference.

**7. Segment findings require confirmation.** A significant subgroup effect is trustworthy if the segment was pre-registered or Cochran's Q confirms heterogeneity. Otherwise, treat it as a hypothesis for a follow-up experiment.

**The default when uncertain: collect more data.** With sufficient sample size, all valid methods converge on the same answer.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0 (async), PostgreSQL 16, Redis 7 |
| Stats engine | NumPy, SciPy (pure functions, no DB dependencies) |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, TanStack Query, Recharts |
| Infra | Docker Compose, 303 unit tests (pytest) |

---

## Quick Start

```bash
git clone https://github.com/WeiL11/AB_test.git
cd AB_test

# Option A: Docker
docker compose up -d
docker compose exec backend python -m scripts.seed_demo

# Option B: Local
cd backend && pip install -e ".[dev]"
docker compose up -d postgres redis
uvicorn app.main:app --reload --port 8000

# Run tests
pytest

# Frontend
cd frontend && npm install && npm run dev
```

**Run the simulation** (no database needed):
```bash
cd backend && python -m scripts.simulate_experiment
```

---

## API

All endpoints under `/api/v1`. Full Swagger docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/experiments/` | Create experiment with variants and metrics |
| `GET` | `/experiments/` | List experiments (paginated, filterable) |
| `GET` | `/experiments/{id}` | Get experiment details |
| `POST` | `/experiments/{id}/start` | Start experiment |
| `POST` | `/experiments/{id}/stop` | Stop experiment |
| `GET` | `/experiments/{id}/results` | Run analysis, return results |
| `POST` | `/events/batch` | Ingest up to 1,000 events |
| `GET` | `/assign/{exp_id}/{user_id}` | Deterministic variant assignment |
| `POST` | `/power/sample-size` | Sample size calculator |

---

## Tests (303)

Every statistical method is tested independently. No mocking -- tests generate synthetic data with known properties and verify the math produces correct results.

| Module | Tests | What's verified |
|---|---|---|
| Frequentist | 14 | Welch's t-test, z-test, chi-squared: significance, CIs, edge cases |
| Sequential | 35 | O'Brien-Fleming/Pocock boundaries, alpha spending, early stopping |
| Confidence sequences | 33 | Width narrows over time, detects effects, valid at all stopping times |
| Bayesian | 48 | Posterior accuracy, P(B>A), expected loss, ROPE, ship/don't-ship logic |
| Bandits | 54 | Thompson/UCB1/Epsilon-Greedy allocation, regret, simulation correctness |
| CUPED | 20 | Variance reduction, effect preservation, unbiasedness |
| Multiple testing | 33 | Bonferroni/Holm/BH adjustments, step-down/step-up logic, dispatching |
| Novelty detection | 26 | Decay/growth/stable classification, weighted regression, daily effects |
| Power analysis | 11 | Sample size, MDE round-trip, power curves, duration estimation |
| Assignment | 5 | Deterministic hashing, uniform distribution, cross-experiment independence |
| Integration | 12 | End-to-end: all 10 methods agree on the same synthetic experiment |

---

## Project Structure

```
backend/app/
  api/          # Route handlers (experiments, events, analysis, power, assignment)
  models/       # SQLAlchemy models (Experiment, Variant, Metric, Event, Assignment)
  services/     # Business logic (analysis, assignment, event ingestion)
  stats/        # Statistical engine -- 10 modules, pure functions
    frequentist.py, sequential.py, confidence_sequence.py, bayesian.py,
    bandit.py, cuped.py, multiple_testing.py, power_analysis.py,
    novelty_detection.py, segment_analysis.py

frontend/src/
  pages/        # Dashboard, ExperimentPage, CreateExperiment, PowerAnalysis
  components/   # CI charts, metrics table, status badges, power calculator
  hooks/        # TanStack Query hooks for server state
```

---

## Further Reading

- **[Architecture](docs/ARCHITECTURE.md)** -- System design, database schema, key design decisions with tradeoffs, scaling considerations.
- **[Statistical Methods](docs/STATISTICAL_METHODS.md)** -- Mathematical foundations for all 10 methods.
- **[Decision Tree](docs/DECISION_TREE.md)** -- Detailed conflict resolution for 9 disagreement scenarios.

---

## License

MIT
