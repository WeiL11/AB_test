# Experimentor

You want to test whether a change -- a new button color, a different checkout flow, a redesigned landing page -- actually improves a metric you care about. You split users into two groups: **Group A** sees the original, **Group B** sees the change. You collect data from both groups and compare.

That comparison is where things get tricky. A naive approach gives you wrong answers more often than you'd expect. This platform implements 10 methods to get the comparison right.

---

## How the 10 Methods Work (and Why Each One Matters)

### 1. Is B actually different from A? — Welch's T-Test

You collect a metric (say, revenue per user) from Group A and Group B. You compute the mean of each group. But the means will always differ slightly just from random chance -- even if the change did nothing. The question is whether the gap is big enough to be real.

**The problem with the simple approach:** Student's t-test assumes both groups have the same variance. That's almost never true in A/B tests -- the change itself shifts the distribution. If you use it anyway, your confidence intervals are wrong.

**What we do instead:** Welch's t-test drops the equal-variance assumption. It estimates degrees of freedom from each group's own variance (Welch-Satterthwaite approximation). For proportions (converted or not), we use an unpooled z-test. Both tell you: "given the data from A and B, here's the effect size, a confidence interval, and a p-value."

### 2. Can I check results early without fooling myself? — Sequential Testing

You launch the experiment and look at the dashboard on day 3. Not significant. You look again on day 5, day 7, day 10, day 14. On day 14 you see p=0.04 and declare a winner.

**The problem:** Every time you peek, you give randomness another chance to look significant. Peeking 5 times at alpha=0.05 inflates your real false positive rate to about 14.6%. You think you have 95% confidence, but you actually have 85%.

**What we do instead:** O'Brien-Fleming alpha-spending. Instead of spending your entire alpha=0.05 budget on one test, you pre-plan how many times you'll look and "spend" small slices of alpha at each look. Early looks have very tight boundaries (hard to reject), later looks have looser ones. The total false positive rate across all looks stays at exactly 5%.

### 3. What if I don't know when I'll check? — Confidence Sequences

Sequential testing requires pre-planning how many times you'll look. But what if your team watches a live dashboard and could stop the experiment any time?

**What we do:** Confidence sequences (mixture Sequential Probability Ratio Test). These produce confidence intervals that are valid at *every* sample size simultaneously. You can check every hour, every day, or only once -- the guarantee holds regardless. The tradeoff: the intervals are wider than fixed-horizon CIs early on, so you need more data to reach significance.

### 4. What's the probability B is actually better? — Bayesian Testing

Frequentist tests give you a p-value: "if B were no different from A, how unlikely is the data I observed?" That answers the question backwards. What you really want to know is: "given the data, what's the probability B is better?"

**What we do:** For conversion metrics, we use a Beta-Binomial model. Start with a flat prior (Beta(1,1) -- no opinion), observe conversions and non-conversions, get a posterior distribution over each group's true rate. Then we sample 100,000 times from both posteriors and directly compute:
- **P(B > A):** How often B's sampled rate beats A's.
- **Expected loss:** If you ship B and you're wrong, how much do you lose on average? This prevents the trap where P(B>A) = 96% but B is only 0.001% better when it wins and 5% worse when it loses.
- **ROPE analysis:** Is the difference practically meaningful, or just statistically non-zero?

Ship decision requires *both* P(B>A) > 0.95 *and* expected loss < 0.1%.

### 5. Can we send more traffic to the winner while testing? — Multi-Armed Bandits

With a fixed 50/50 split, half your users get the worse experience for the entire experiment. Bandits dynamically shift traffic toward the variant that's performing better, reducing the total cost of experimentation.

**What the data tells the bandit:** Each group's conversion rate updates a reward estimate. The bandit uses these estimates to decide where to send the next user:
- **Thompson Sampling:** Sample from each variant's Beta posterior, send the user to whichever sampled highest. Naturally balances exploration (trying uncertain options) and exploitation (using the current best).
- **UCB1:** Send users to the variant with the highest `mean + exploration_bonus`. Deterministic, no randomness.
- **Epsilon-Greedy:** Send 90% of traffic to the best variant, 10% random. Simple baseline.

### 6. Can we make the experiment faster? — CUPED Variance Reduction

A user who spent $200 last month is likely to spend more this month than a user who spent $5 -- regardless of which group they're in. This natural variation adds noise to your A/B test, making it harder to detect a real effect.

**What we do:** CUPED (Controlled-experiment Using Pre-Experiment Data) subtracts out this predictable variation. For each user, we adjust: `Y_adjusted = Y - theta * (X - average(X))`, where X is their pre-experiment behavior. This reduces variance by `1 - correlation^2`. With typical correlations of 0.5-0.7, that's 25-50% less noise -- equivalent to doubling your sample size for free. The adjustment is provably unbiased because `average(X - average(X)) = 0` by construction.

### 7. What if I'm testing 20 metrics at once? — Multiple Testing Correction

You measure conversion, revenue, session time, page views, bounce rate, and 15 other metrics. Even if B has zero effect on everything, there's a `1 - 0.95^20 = 64%` chance at least one metric shows p < 0.05 by pure chance.

**What we do:** Adjust the significance threshold:
- **Bonferroni:** Divide alpha by the number of tests. Simple, conservative.
- **Holm (step-down):** Strictly more powerful than Bonferroni -- tests the smallest p-value at alpha/n, the second-smallest at alpha/(n-1), and so on. No reason to ever prefer Bonferroni over Holm.
- **Benjamini-Hochberg:** Controls the *false discovery rate* rather than the chance of any single false positive. Much more lenient, best when you have 10+ exploratory metrics and accept that some "significant" results may be false.

### 8. How many users do I need? — Power Analysis

Before running the experiment: how large does each group need to be to detect an effect of a given size?

**What the data tells you (in advance):** Given your baseline conversion rate (say 10%), how big an improvement you want to detect (say 1 percentage point), and your tolerance for false negatives (typically 20%), the power calculator computes the required sample size per variant. It also plots a *power curve* showing how detection ability changes with effect size, and estimates how many days the experiment will take given your daily traffic.

### 9. Is the effect real or just novelty? — Novelty/Primacy Detection

You launch a new checkout flow. Conversion jumps 15% in week 1, 8% in week 2, 3% in week 3. Users were reacting to the novelty of change, not the quality of the new design. If you stopped on day 7, you'd massively overestimate the long-term effect.

**What we do:** Compute the daily treatment effect (B minus A) for each day, then fit a weighted linear regression over time. If the slope is significantly negative, the effect is decaying (novelty). If significantly positive, it's growing (primacy -- users learning a new flow). If flat, the effect is stable and trustworthy.

### 10. Does B work differently for different users? — Segment Analysis

Overall, B shows no significant effect. But when you split by platform, B is +12% on mobile and -8% on desktop. The overall average hides real heterogeneity.

**What we do:** Run the A/B test independently within each user segment (country, platform, cohort, etc.), then test whether the effects differ across segments using Cochran's Q test with inverse-variance weighting. Multiple testing correction is applied across segments to prevent false discoveries.

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
git clone https://github.com/WeiL11/AB_test.git
cd AB_test

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

# Run tests (303 tests)
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
|   |-- tests/                       # 303 unit tests
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
