# Experimentor

A/B testing platform with 10 statistical methods. Python/FastAPI backend, React frontend, 303 tests.

Group A sees the current experience. Group B sees the change. Both groups generate data. This platform determines whether the difference is real or noise.

## Case Study

Hypothetical Spotify experiment: new recommendation engine vs. existing. 50,000 users per group, 3 weeks. Three metrics — streaming hours (primary), premium renewal rate (secondary), app crash rate (guardrail).

Run it yourself (no database needed):
```bash
cd backend && python -m scripts.realistic_case
```

| Method | Result |
|---|---|
| Power analysis | Need 974/group for hours, 9,955 for renewal. At 50K: 100% power |
| Welch's t-test | Streaming +0.27 hrs, p<0.001. Renewal +1.68pp, p<0.001. **Crash +0.29pp, p=0.001** |
| Sequential | Could stop at 40% of data (z=6.32 > boundary 3.10) |
| Confidence sequences | +0.27 hrs, bounds [+0.265, +0.266], significant at any stopping time |
| Bayesian | P(B>A) = 100%, expected loss = 0, recommendation: ship |
| Thompson Sampling | 88% traffic to treatment after 20K rounds |
| CUPED | 75% variance reduction using last month's data, same effect |
| Multiple testing | All 3 metrics significant after Holm correction |
| Novelty detection | Stable over 21 days, no decay |
| Segment analysis | iOS: +0.45 hrs (p<0.001). Android: -0.03 hrs (not significant) |

**Decision: do not ship.** 9 methods say ship, but the crash rate guardrail failed. Guardrail regression is a hard veto. The effect is also iOS-only — Android shows no improvement.

## Methods

| # | Method | Question it answers |
|---|---|---|
| 1 | Welch's t-test | Is the difference between A and B statistically significant? |
| 2 | Sequential testing | Can I check results early without inflating false positives? |
| 3 | Confidence sequences | Can I monitor continuously without pre-planning when to look? |
| 4 | Bayesian testing | What is P(B > A), and what do I lose if I'm wrong? |
| 5 | Multi-armed bandits | Can I shift traffic to the winner during the experiment? |
| 6 | CUPED | Can I reduce noise using pre-experiment data? |
| 7 | Multiple testing | How do I correct for testing many metrics at once? |
| 8 | Power analysis | How many users do I need before starting? |
| 9 | Novelty detection | Is the effect decaying over time? |
| 10 | Segment analysis | Does the effect differ across user groups? |

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

# Docker
docker compose up -d
docker compose exec backend python -m scripts.seed_demo

# Or local
cd backend && pip install -e ".[dev]"
docker compose up -d postgres redis
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
