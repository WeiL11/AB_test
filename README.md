# Experimentor

A/B testing platform with 4 core testing methods and 6 supporting techniques. Python/FastAPI backend, React frontend, 303 tests.

Group A sees the current experience. Group B sees the change. Both groups generate data. This platform determines whether the difference is real or noise.

## Case Study

Spotify (hypothetical) tests a new recommendation engine. 50,000 users per group, 3 weeks.

```bash
cd backend && python -m scripts.realistic_case   # run it yourself, no database needed
```

### Decision: Do Not Ship

The new algorithm improves streaming hours and renewal rates, but **increases app crash rate**. Crash rate is a guardrail metric — any significant regression is a hard veto regardless of primary metric gains.

### What improved

| Metric | Control | Treatment | Change | p-value | Verdict |
|---|---|---|---|---|---|
| Streaming hours (primary) | 18.20 hrs/wk | 18.33 hrs/wk | **+0.14 hrs** | <0.001 | Significant lift |
| Premium renewal (secondary) | 81.9% | 83.6% | **+1.68pp** | <0.001 | Significant lift |

### What blocked the launch

| Metric | Control | Treatment | Change | p-value | Verdict |
|---|---|---|---|---|---|
| App crash rate (guardrail) | 1.81% | 2.49% | **+0.68pp** | <0.001 | **Regression — veto** |

### Deeper analysis

| Check | Finding | Implication |
|---|---|---|
| Power analysis | Need ~50K per group; we have 50K (80% power at MDE=0.08 hrs) | Experiment is properly powered, not over- or under-powered |
| Early stopping | Could stop at 40% of data (z=3.39 > boundary 3.10) | Effect was detectable early — not a data issue |
| Bayesian | P(B>A) >99.99%, expected loss <0.0001 | The improvement is real, not noise |
| Variance reduction | CUPED reduced noise by 75% using last month's data | Effect estimate is reliable |
| Effect stability | No decay over 21 days (slope p=0.55) | Not a novelty effect |
| Platform segments | iOS: +0.37 hrs (p<0.001), Android: -0.06 hrs (p=0.17) | Effect is iOS-only. The overall +0.14 hrs is a weighted average (60% iOS, 40% Android) masking a null result on Android |

### Recommended next steps

1. Investigate the crash rate increase — is it caused by the new algorithm or a correlated deployment?
2. If the crash issue is fixable, patch and re-run the experiment.
3. Consider an iOS-only rollout, since Android shows no benefit.
4. If crash rate is inherent to the new algorithm, do not ship.

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

303 tests, no mocking. Synthetic data with known properties verifies math correctness.

| Module | Count | Covers |
|---|---|---|
| Bayesian | 48 | Posterior, P(B>A), expected loss, ROPE, recommendations |
| Bandits | 54 | Thompson/UCB1/Epsilon-Greedy, regret, simulation |
| Sequential | 35 | O'Brien-Fleming/Pocock, alpha spending, early stopping |
| Confidence sequences | 33 | Narrowing, detection, any-time validity |
| Multiple testing | 33 | Bonferroni/Holm/BH, step logic, dispatching |
| Novelty detection | 26 | Decay/growth/stable, weighted regression |
| CUPED | 20 | Variance reduction, unbiasedness, effect preservation |
| Frequentist | 14 | Welch's t-test, z-test, chi-squared |
| Integration | 12 | End-to-end: all methods on same dataset |
| Power analysis | 11 | Sample size, MDE, power curves, duration |
| Assignment | 5 | Deterministic hashing, uniform distribution |

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
