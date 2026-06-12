# Experimentor

**A production-grade A/B testing platform with 9 statistical methods, from frequentist hypothesis tests to multi-armed bandits.**

Built to demonstrate the statistical rigor and system design behind platforms like Google Optimize, Statsig, and Eppo. Full-stack application with a FastAPI backend, React frontend, and a statistical engine covering the methods that matter most in industry experimentation.

---

## What Sets This Apart

Most A/B testing platforms implement a t-test and call it done. Experimentor implements the same statistical methods used at Google, Netflix, and Microsoft to run experiments correctly at scale.

| Capability | Basic A/B Tool | Experimentor |
|---|---|---|
| Hypothesis testing | Student's t-test (assumes equal variance) | Welch's t-test (no equal variance assumption) |
| Peeking at results | Inflates false positive rate to ~14.6% with 5 peeks | Group sequential testing with O'Brien-Fleming spending |
| Continuous monitoring | Not supported | Always-valid confidence sequences (valid at any stopping time) |
| Bayesian analysis | P(B>A) only | Full posterior: P(B>A), expected loss, ROPE analysis |
| Multiple metrics | No correction (64% false positive rate with 20 metrics) | Bonferroni, Holm, Benjamini-Hochberg FDR |
| Traffic optimization | Fixed 50/50 split | Thompson Sampling, UCB1, Epsilon-Greedy bandits |
| Variance reduction | None | CUPED (30-50% variance reduction using pre-experiment data) |
| Effect stability | Declares winner on day 1 | Novelty/primacy detection via trend regression |
| Power analysis | Manual calculation | Interactive calculator with power curves |
| Segment analysis | None | Heterogeneous treatment effect detection with Cochran's Q |
| Traffic splitting | Random (non-reproducible) | Deterministic SHA-256 hashing (stateless, verifiable) |
| Test suite | None | 291 unit tests across all statistical methods |

---

## Statistical Methods

### 1. Frequentist Testing
Welch's t-test for continuous metrics, unpooled z-test for proportions. Uses the Welch-Satterthwaite degrees of freedom approximation -- no equal-variance assumption required. Chi-squared independence test for contingency tables.

### 2. Sequential Testing (Group Sequential)
O'Brien-Fleming and Pocock alpha-spending functions via the Lan-DeMets framework. Solves the "peeking problem": checking results 5 times at alpha=0.05 inflates false positives to ~14.6% without correction. Sequential boundaries keep the overall Type I error at exactly 0.05.

### 3. Always-Valid Inference (Confidence Sequences)
Normal-mixture mSPRT confidence sequences that are valid at any sample size, including data-dependent stopping times. Wider than fixed-horizon CIs early on, but guarantee `P(mu in CS) >= 1 - alpha` at ALL times simultaneously.

### 4. Bayesian A/B Testing
Beta-Binomial model for conversion metrics, Normal-Normal conjugate model for continuous metrics. Directly answers business questions: "What is the probability B beats A?" and "If I ship B and I'm wrong, how much do I lose?" (expected loss). Includes Region of Practical Equivalence (ROPE) analysis.

### 5. Multi-Armed Bandits
Thompson Sampling (optimal Bayesian exploration-exploitation), UCB1 (deterministic upper confidence bounds), and Epsilon-Greedy (simple baseline). Dynamically shifts traffic toward better-performing variants to reduce opportunity cost during the experiment.

### 6. CUPED Variance Reduction
Controlled-experiment Using Pre-Experiment Data. Reduces metric variance by `1 - rho^2` (30-50% typical) by regressing out pre-experiment user behavior. The adjustment `Y_adj = Y - theta * (X - E[X])` is provably unbiased since `E[X - E[X]] = 0`.

### 7. Multiple Testing Correction
Bonferroni (controls FWER, most conservative), Holm-Bonferroni step-down (controls FWER, strictly more powerful), and Benjamini-Hochberg (controls FDR, best for 10+ metrics). Without correction, testing 20 metrics at alpha=0.05 gives a 64% chance of at least one false positive.

### 8. Power Analysis
Sample size calculation, minimum detectable effect computation (via Brent's method), power curves, and experiment duration estimation. Supports both binomial and continuous metrics.

### 9. Novelty/Primacy Detection
Weighted linear regression of daily treatment effects over time. Detects decaying effects (novelty -- users react to change, not improvement) and growing effects (primacy -- users learning a new flow). Prevents premature winner declaration.

### 10. Segment Analysis
Runs the A/B test within each user segment independently, then tests for heterogeneous treatment effects using Cochran's Q test with inverse-variance weighting. Applies multiple testing correction across segments. Detects when an overall null result hides a strong subgroup effect.

---

## Tech Stack

**Backend**
- Python 3.12, FastAPI, Uvicorn
- SQLAlchemy 2.0 (async), Alembic
- PostgreSQL 16, Redis 7
- NumPy, SciPy (statistical engine)
- Pydantic v2 (validation and serialization)

**Frontend**
- React 18, TypeScript, Vite
- Tailwind CSS
- TanStack Query (server state management)
- Recharts (data visualization)

**Infrastructure**
- Docker Compose (PostgreSQL 16, Redis 7, backend)
- Dockerfile with multi-stage build

---

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/your-username/ab-testing-platform.git
cd ab-testing-platform

# Start all services
docker compose up -d

# Seed demo data (3 experiments, ~150K events)
docker compose exec backend python -m scripts.seed_demo

# Backend API:   http://localhost:8000
# API docs:      http://localhost:8000/docs
# Frontend:      http://localhost:5173
```

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start PostgreSQL and Redis (or use Docker for just the databases)
docker compose up -d postgres redis

# Run the server
uvicorn app.main:app --reload --port 8000

# Seed demo data
python -m scripts.seed_demo

# Run tests (291 tests)
pytest
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev          # Development server at http://localhost:5173
npm run build        # Production build (~236ms)
```

---

## API Endpoints

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/experiments/` | Create a new experiment with variants and metrics |
| `GET` | `/experiments/` | List experiments (paginated, filterable by status) |
| `GET` | `/experiments/{id}` | Get experiment details |
| `PATCH` | `/experiments/{id}` | Update experiment configuration |
| `POST` | `/experiments/{id}/start` | Start an experiment (draft -> running) |
| `POST` | `/experiments/{id}/stop` | Stop an experiment (running -> completed) |
| `GET` | `/experiments/{id}/results` | Run statistical analysis and return results |
| `POST` | `/events/` | Ingest a single event |
| `POST` | `/events/batch` | Ingest events in batch (up to 1000) |
| `GET` | `/assign/{experiment_id}/{user_id}` | Get deterministic variant assignment |
| `POST` | `/power/sample-size` | Calculate required sample size |

Interactive API documentation is available at `/docs` (Swagger UI) when the server is running.

---

## Project Structure

```
ab-testing-platform/
|-- docker-compose.yml
|-- backend/
|   |-- app/
|   |   |-- main.py                  # FastAPI application, lifespan, CORS
|   |   |-- config.py                # Pydantic settings (DB, Redis, defaults)
|   |   |-- db.py                    # SQLAlchemy async engine, session factory
|   |   |-- api/                     # Route handlers
|   |   |   |-- experiments.py       # CRUD + start/stop
|   |   |   |-- events.py            # Single + batch event ingestion
|   |   |   |-- analysis.py          # Statistical analysis endpoint
|   |   |   |-- power.py             # Power/sample-size calculator
|   |   |   |-- assign.py            # Variant assignment
|   |   |   +-- health.py            # Health check
|   |   |-- models/                  # SQLAlchemy ORM models
|   |   |   |-- experiment.py        # Experiment, Variant, Metric
|   |   |   |-- event.py             # Event, Assignment
|   |   |   +-- result.py            # AnalysisResult (cached results)
|   |   |-- schemas/                 # Pydantic request/response schemas
|   |   |-- services/                # Business logic layer
|   |   |   |-- experiment_service.py
|   |   |   |-- event_service.py
|   |   |   |-- analysis_service.py
|   |   |   +-- assignment_service.py
|   |   +-- stats/                   # Statistical engine (pure functions)
|   |       |-- base.py              # Result dataclasses
|   |       |-- frequentist.py       # Welch's t-test, z-test, chi-squared
|   |       |-- sequential.py        # Group sequential (O'Brien-Fleming, Pocock)
|   |       |-- confidence_sequence.py # Always-valid inference (mSPRT)
|   |       |-- bayesian.py          # Beta-Binomial, Normal-Normal, ROPE
|   |       |-- bandit.py            # Thompson, UCB1, Epsilon-Greedy
|   |       |-- cuped.py             # CUPED variance reduction
|   |       |-- multiple_testing.py  # Bonferroni, Holm, BH-FDR
|   |       |-- power_analysis.py    # Sample size, MDE, power curves
|   |       |-- novelty_detection.py # Novelty/primacy trend detection
|   |       +-- segment_analysis.py  # Heterogeneous treatment effects
|   |-- tests/                       # 291 unit tests
|   |-- scripts/
|   |   +-- seed_demo.py             # Generates 3 experiments with 150K events
|   +-- Dockerfile
+-- frontend/
    +-- src/
        |-- api/                     # API client (TanStack Query)
        |-- components/
        |   |-- analysis/            # CI charts, metrics table, results summary
        |   |-- common/              # MetricCard, StatusBadge, ConfidenceBand
        |   |-- layout/              # Header, Sidebar
        |   +-- power/               # Interactive power calculator
        |-- hooks/                   # useExperiment, useAnalysis
        |-- pages/                   # Dashboard, CreateExperiment, ExperimentPage, PowerAnalysis
        +-- types/                   # TypeScript interfaces
```

---

## How It Works

```
1. CREATE            2. ASSIGN             3. COLLECT            4. ANALYZE
experiment with      users via             events with           on demand via
variants + metrics   deterministic hash    metric values         /results endpoint
       |                   |                    |                      |
       v                   v                    v                      v
  [Experiment]    sha256(exp.user) -> bucket  [Events]         [Stats Engine]
  [Variants]      bucket -> variant          stored in         frequentist |
  [Metrics]       (stateless, repeatable)    PostgreSQL        sequential  |
       |                                         |             bayesian   |
       +-------- stored in PostgreSQL -----------+             bandit     |
                                                               cuped     |
                                                               ...       |
                                                                    |
                                                                    v
                                                            [ExperimentResults]
                                                             per-metric CIs
                                                             p-values
                                                             recommendation
```

**Data flow in detail:**

1. **Experiment Creation** -- An experiment is created with one control variant and one or more treatment variants, each with a traffic percentage. Metrics are defined as primary (drives the ship/no-ship decision), secondary (informational), or guardrail (must not regress).

2. **User Assignment** -- When a user hits the assignment endpoint, `SHA-256(experiment_id + "." + user_id)` is computed and mapped to a bucket in `[0, 9999]`. Variants are sorted by ID and walked in order, accumulating traffic percentages until the bucket is covered. This is fully deterministic: the same user always gets the same variant, with no server-side state required.

3. **Event Ingestion** -- Events are recorded with an experiment ID, variant ID, metric name, and numeric value. Binary outcomes use 0/1 values. Batch ingestion supports up to 1000 events per request. Events are indexed by `(experiment_id, metric_name)` for efficient per-metric queries.

4. **Statistical Analysis** -- The `/results` endpoint fetches all events for the experiment, groups them by variant and metric, and runs the appropriate statistical test. The analysis service computes a recommendation (`ship`, `dont_ship`, `keep_running`, `inconclusive`) based on primary metric significance and guardrail checks.

---

## Further Reading

- **[Architecture Document](docs/ARCHITECTURE.md)** -- System design, database schema, statistical engine architecture, key design decisions with tradeoffs, and scaling considerations.

---

## License

MIT
