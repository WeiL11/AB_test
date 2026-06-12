# Experimentor: A Production-Grade A/B Testing Platform

## Project Overview

A full-stack experimentation platform demonstrating Google L5/L6 interview-level statistical rigor, system design, and engineering quality. This is **not** a toy t-test calculator -- it implements the same methods used at Google, Netflix, Meta, and Spotify.

---

## What Makes This Google Interview-Level

| Dimension | Toy Project | This Project |
|---|---|---|
| **Statistics** | Single t-test, alpha=0.05 | Sequential testing, Bayesian posteriors, CUPED variance reduction, multiple testing correction |
| **Peeking problem** | Ignores it (inflated false positives) | O'Brien-Fleming spending + confidence sequences for continuous monitoring |
| **Optimization** | Fixed 50/50 split | Multi-armed bandit (Thompson Sampling, UCB) with regret tracking |
| **Variance reduction** | None | CUPED -- reduces required sample size by 30-50% |
| **Guardrails** | None | Automatic guardrail metric monitoring with alerting |
| **Segment analysis** | None | Heterogeneous treatment effect detection across user segments |
| **System design** | Monolith with SQLite | Event ingestion, async aggregation, Redis caching, proper API with pagination |
| **Frontend** | Basic table | Interactive dashboard with CI plots, posterior distributions, power calculator |

---

## Technology Stack

### Backend (Python)
| Component | Technology |
|---|---|
| Web framework | FastAPI |
| Statistical computation | NumPy, SciPy, statsmodels |
| Bayesian engine | Conjugate priors (hand-rolled) + optional PyMC |
| Database | PostgreSQL |
| Cache / Pub-Sub | Redis |
| Task queue | Celery + Redis (or arq) |
| ORM | SQLAlchemy 2.0 (async) + Alembic |
| Testing | pytest + hypothesis |
| Containerization | Docker + docker-compose |

### Frontend (React + TypeScript)
| Component | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| Charts | Recharts + D3 for custom statistical plots |
| State management | TanStack Query (React Query) |
| UI components | Tailwind CSS + shadcn/ui |
| Routing | React Router v6 |
| Real-time | Server-Sent Events (SSE) |

---

## Project Structure

```
ab-testing-platform/
├── README.md
├── docker-compose.yml
├── docs/
│   ├── ARCHITECTURE.md               # System design (interview talking point)
│   ├── STATISTICAL_METHODS.md         # Math derivations & references
│   └── API.md
│
├── backend/
│   ├── pyproject.toml
│   ├── alembic/
│   │   └── versions/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── config.py                  # Pydantic BaseSettings
│   │   ├── dependencies.py
│   │   ├── db.py
│   │   │
│   │   ├── api/                       # Thin API layer
│   │   │   ├── experiments.py
│   │   │   ├── events.py
│   │   │   ├── analysis.py
│   │   │   ├── power.py
│   │   │   ├── segments.py
│   │   │   └── health.py
│   │   │
│   │   ├── models/                    # SQLAlchemy ORM
│   │   │   ├── experiment.py
│   │   │   ├── event.py
│   │   │   ├── assignment.py
│   │   │   ├── segment.py
│   │   │   └── result.py
│   │   │
│   │   ├── schemas/                   # Pydantic request/response
│   │   │   ├── experiment.py
│   │   │   ├── event.py
│   │   │   ├── analysis.py
│   │   │   └── power.py
│   │   │
│   │   ├── services/                  # Business logic
│   │   │   ├── experiment_service.py
│   │   │   ├── assignment_service.py
│   │   │   ├── event_service.py
│   │   │   └── analysis_service.py
│   │   │
│   │   ├── stats/                     # THE CORE: Statistical Engine
│   │   │   ├── base.py               # Abstract interfaces
│   │   │   ├── frequentist.py         # Welch's t-test, z-test, chi-squared
│   │   │   ├── sequential.py          # Group sequential (O'Brien-Fleming, Pocock)
│   │   │   ├── confidence_sequence.py # Always-valid inference (mSPRT)
│   │   │   ├── bayesian.py            # Beta-Binomial, Normal posteriors, Thompson Sampling
│   │   │   ├── bandit.py              # Multi-armed bandit (Thompson, UCB1, Epsilon-greedy)
│   │   │   ├── cuped.py              # CUPED variance reduction
│   │   │   ├── multiple_testing.py    # Bonferroni, Holm, Benjamini-Hochberg FDR
│   │   │   ├── power_analysis.py      # Sample size, MDE, power calculation
│   │   │   ├── novelty_detection.py   # Novelty/primacy effect detection
│   │   │   ├── segment_analysis.py    # Heterogeneous treatment effects
│   │   │   └── guardrails.py          # Guardrail metric monitoring
│   │   │
│   │   ├── assignment/                # Traffic splitting
│   │   │   ├── hasher.py             # Deterministic hash-based assignment
│   │   │   └── router.py
│   │   │
│   │   └── workers/                   # Background tasks
│   │       ├── aggregator.py
│   │       ├── analyzer.py
│   │       └── alerter.py
│   │
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_stats/
│   │   │   ├── test_frequentist.py
│   │   │   ├── test_sequential.py
│   │   │   ├── test_bayesian.py
│   │   │   ├── test_bandit.py
│   │   │   ├── test_cuped.py
│   │   │   ├── test_multiple_testing.py
│   │   │   └── test_power_analysis.py
│   │   ├── test_api/
│   │   │   ├── test_experiments.py
│   │   │   ├── test_events.py
│   │   │   └── test_analysis.py
│   │   ├── test_services/
│   │   │   ├── test_assignment.py
│   │   │   └── test_event_service.py
│   │   └── test_integration/
│   │       └── test_full_experiment.py
│   │
│   └── scripts/
│       ├── seed_demo.py
│       └── simulate_experiment.py
│
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/
        │   ├── client.ts
        │   ├── experiments.ts
        │   └── analysis.ts
        ├── components/
        │   ├── layout/
        │   │   ├── Sidebar.tsx
        │   │   └── Header.tsx
        │   ├── experiments/
        │   │   ├── ExperimentList.tsx
        │   │   ├── ExperimentForm.tsx
        │   │   └── ExperimentDetail.tsx
        │   ├── analysis/
        │   │   ├── ResultsSummary.tsx
        │   │   ├── ConfidenceIntervalChart.tsx
        │   │   ├── PosteriorPlot.tsx
        │   │   ├── SequentialMonitor.tsx
        │   │   ├── SegmentBreakdown.tsx
        │   │   ├── GuardrailPanel.tsx
        │   │   └── CumulativeChart.tsx
        │   ├── power/
        │   │   ├── PowerCalculator.tsx
        │   │   └── SampleSizePlot.tsx
        │   └── common/
        │       ├── MetricCard.tsx
        │       ├── StatusBadge.tsx
        │       └── ConfidenceBand.tsx
        ├── pages/
        │   ├── Dashboard.tsx
        │   ├── ExperimentPage.tsx
        │   ├── CreateExperiment.tsx
        │   ├── PowerAnalysis.tsx
        │   └── Settings.tsx
        ├── hooks/
        │   ├── useExperiment.ts
        │   ├── useAnalysis.ts
        │   └── useRealtime.ts
        ├── types/
        │   ├── experiment.ts
        │   └── analysis.ts
        └── utils/
            ├── format.ts
            └── statistics.ts
```

---

## Database Schema

### Core Tables

```sql
-- Experiment definition
CREATE TABLE experiments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    hypothesis      TEXT,                         -- Forces scientific rigor
    status          VARCHAR(20) NOT NULL DEFAULT 'draft',
                    -- draft | running | paused | completed | killed
    allocation_pct  REAL NOT NULL DEFAULT 100.0,
    analysis_type   VARCHAR(20) NOT NULL DEFAULT 'frequentist',
                    -- frequentist | bayesian | sequential | bandit

    -- Sequential testing config
    max_sample_size     INTEGER,
    interim_analyses    INTEGER DEFAULT 5,
    spending_function   VARCHAR(30) DEFAULT 'obrien_fleming',

    -- Bayesian config
    prior_alpha    REAL DEFAULT 1.0,
    prior_beta     REAL DEFAULT 1.0,
    rope_lower     REAL,
    rope_upper     REAL,

    -- Bandit config
    bandit_algorithm VARCHAR(20),
    epsilon         REAL DEFAULT 0.1,

    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at     TIMESTAMPTZ,
    ended_at       TIMESTAMPTZ,
    created_by     VARCHAR(100)
);

-- Variants within an experiment
CREATE TABLE variants (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id  UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name           VARCHAR(100) NOT NULL,
    is_control     BOOLEAN NOT NULL DEFAULT FALSE,
    traffic_pct    REAL NOT NULL,
    description    TEXT,
    UNIQUE(experiment_id, name)
);

-- Metrics to track
CREATE TABLE metrics (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id  UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name           VARCHAR(100) NOT NULL,
    metric_type    VARCHAR(20) NOT NULL,       -- primary | secondary | guardrail
    data_type      VARCHAR(20) NOT NULL,       -- binomial | continuous | count | ratio
    guardrail_direction VARCHAR(10),
    guardrail_threshold REAL,
    cuped_enabled  BOOLEAN DEFAULT FALSE,
    cuped_covariate VARCHAR(100),
    cuped_lookback_days INTEGER DEFAULT 14,
    minimum_detectable_effect REAL,
    UNIQUE(experiment_id, name)
);

-- Deterministic user-to-variant assignments
CREATE TABLE assignments (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   UUID NOT NULL REFERENCES experiments(id),
    user_id         VARCHAR(255) NOT NULL,
    variant_id      UUID NOT NULL REFERENCES variants(id),
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    segments        JSONB DEFAULT '{}',
    UNIQUE(experiment_id, user_id)
);

-- Raw events (append-only)
CREATE TABLE events (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   UUID NOT NULL REFERENCES experiments(id),
    user_id         VARCHAR(255) NOT NULL,
    variant_id      UUID NOT NULL REFERENCES variants(id),
    metric_name     VARCHAR(100) NOT NULL,
    value           DOUBLE PRECISION NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'
);

-- Pre-experiment covariate data (for CUPED)
CREATE TABLE pre_experiment_data (
    user_id         VARCHAR(255) NOT NULL,
    metric_name     VARCHAR(100) NOT NULL,
    value           DOUBLE PRECISION NOT NULL,
    measured_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(user_id, metric_name)
);

-- Cached analysis results
CREATE TABLE analysis_results (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   UUID NOT NULL REFERENCES experiments(id),
    metric_id       UUID NOT NULL REFERENCES metrics(id),
    variant_id      UUID NOT NULL REFERENCES variants(id),
    segment_filter  JSONB DEFAULT '{}',

    -- Frequentist
    sample_size         INTEGER,
    mean                DOUBLE PRECISION,
    variance            DOUBLE PRECISION,
    ci_lower            DOUBLE PRECISION,
    ci_upper            DOUBLE PRECISION,
    relative_lift       DOUBLE PRECISION,
    p_value             DOUBLE PRECISION,
    is_significant      BOOLEAN,

    -- Sequential
    z_statistic         DOUBLE PRECISION,
    sequential_boundary DOUBLE PRECISION,
    can_stop_early      BOOLEAN DEFAULT FALSE,

    -- Bayesian
    posterior_mean      DOUBLE PRECISION,
    posterior_std       DOUBLE PRECISION,
    prob_beat_control   DOUBLE PRECISION,
    expected_loss       DOUBLE PRECISION,
    credible_lower      DOUBLE PRECISION,
    credible_upper      DOUBLE PRECISION,

    -- CUPED
    cuped_adjusted_mean DOUBLE PRECISION,
    cuped_variance_reduction DOUBLE PRECISION,

    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analysis_method VARCHAR(30) NOT NULL
);

-- Multiple testing correction results
CREATE TABLE correction_results (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   UUID NOT NULL REFERENCES experiments(id),
    method          VARCHAR(30) NOT NULL,
    metric_id       UUID NOT NULL REFERENCES metrics(id),
    variant_id      UUID NOT NULL REFERENCES variants(id),
    original_p      DOUBLE PRECISION,
    adjusted_p      DOUBLE PRECISION,
    is_significant  BOOLEAN,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## Statistical Engine -- The Core Differentiator

### 9 Statistical Methods Implemented

#### 1. Frequentist Testing (`stats/frequentist.py`)
- Welch's t-test (NOT Student's -- does not assume equal variances)
- Z-test for proportions (unpooled)
- Chi-squared test for categorical outcomes
- Delta method for ratio metrics

#### 2. Sequential Testing (`stats/sequential.py`)
Solves the **peeking problem**: without sequential testing, peeking at results 5 times inflates false positive rate from 5% to ~14.6%.
- O'Brien-Fleming alpha spending: conservative early, aggressive late
- Pocock alpha spending: constant boundaries
- Lan-DeMets framework for flexible analysis timing

#### 3. Always-Valid Inference (`stats/confidence_sequence.py`)
- Mixture sequential probability ratio test (mSPRT)
- Confidence sequences valid at ALL sample sizes simultaneously
- No need to pre-specify number of interim looks
- Based on Howard et al. (2021)

#### 4. Bayesian Engine (`stats/bayesian.py`)
- Beta-Binomial conjugate for conversion metrics
- Normal-Normal conjugate for continuous metrics
- P(B > A) via Monte Carlo sampling
- Expected loss: E[max(0, theta_A - theta_B)]
- Region of Practical Equivalence (ROPE)

#### 5. Multi-Armed Bandit (`stats/bandit.py`)
- Thompson Sampling: optimal exploration-exploitation via posterior sampling
- UCB1: deterministic upper confidence bound
- Epsilon-Greedy: simple baseline
- Cumulative regret tracking

#### 6. CUPED Variance Reduction (`stats/cuped.py`)
Reduces metric variance by 30-50% using pre-experiment data as covariate.
```
Y_adjusted = Y - theta * (X - E[X])
Var(Y_adjusted) = Var(Y) * (1 - rho^2)
```
If pre/post correlation rho = 0.7, variance reduction = 51%.

#### 7. Multiple Testing Correction (`stats/multiple_testing.py`)
- Bonferroni: alpha / m (conservative)
- Holm-Bonferroni: step-down (uniformly more powerful)
- Benjamini-Hochberg FDR: controls false discovery rate (best for many metrics)

#### 8. Power Analysis (`stats/power_analysis.py`)
- Required sample size given MDE, alpha, power
- Minimum detectable effect given sample size
- Power curves
- Duration estimation from daily traffic

#### 9. Novelty/Primacy Detection (`stats/novelty_detection.py`)
- Weighted linear regression of daily treatment effect over time
- Detects decaying effects (novelty) or growing effects (primacy)

---

## API Design

### Experiment Lifecycle
```
POST   /api/v1/experiments              Create experiment
GET    /api/v1/experiments              List (paginated, filterable)
GET    /api/v1/experiments/{id}         Get details
PATCH  /api/v1/experiments/{id}         Update config
POST   /api/v1/experiments/{id}/start   Start experiment
POST   /api/v1/experiments/{id}/stop    Stop experiment
```

### Traffic Assignment
```
GET    /api/v1/assign/{experiment_id}/{user_id}   Get variant
POST   /api/v1/assign/bulk                         Batch lookup
```

### Event Ingestion
```
POST   /api/v1/events                   Single event
POST   /api/v1/events/batch             Batch (up to 1000)
```

### Analysis
```
GET    /api/v1/experiments/{id}/results              Full results
GET    /api/v1/experiments/{id}/results/sequential    Sequential monitoring
GET    /api/v1/experiments/{id}/results/bayesian      Posteriors
GET    /api/v1/experiments/{id}/results/segments      Segment breakdowns
GET    /api/v1/experiments/{id}/results/guardrails    Guardrail status
GET    /api/v1/experiments/{id}/results/novelty       Novelty detection
GET    /api/v1/experiments/{id}/results/timeseries    Daily effects
POST   /api/v1/experiments/{id}/results/recompute     Force recomputation
```

### Power Analysis
```
POST   /api/v1/power/sample-size       Required sample size
POST   /api/v1/power/mde               Minimum detectable effect
POST   /api/v1/power/duration           Duration estimate
```

### Real-time
```
GET    /api/v1/experiments/{id}/stream  SSE for live updates
```

---

## Frontend Pages

### 1. Dashboard
- Experiment list with status badges (draft/running/completed)
- Summary cards: total experiments, active, decisions made

### 2. Create Experiment (Multi-step Wizard)
1. Basic info (name, hypothesis)
2. Variants (add/remove, traffic percentages)
3. Metrics (primary/secondary/guardrail, CUPED toggle)
4. Analysis settings (frequentist/bayesian/sequential, alpha, power)
5. Review and launch with integrated power calculator

### 3. Experiment Results (Centerpiece)
- **Summary Panel**: status, days running, recommendation (ship/don't ship/keep running)
- **Confidence Interval Chart**: horizontal bar with point estimate and CI band
- **Sequential Monitor**: z-statistic over time with O'Brien-Fleming boundaries
- **Bayesian Posterior Plot**: overlaid posterior distributions, P(B>A) shading
- **Segment Breakdown**: treatment effect per segment with heatmap
- **Guardrail Panel**: red/green indicators per guardrail metric
- **Time Series**: daily cumulative effect with expanding confidence bands
- **Metrics Table**: all metrics with raw and corrected p-values

### 4. Power Calculator (Standalone)
- Interactive sliders for baseline, MDE, alpha, power
- Real-time sample size computation
- Power curve visualization
- Duration estimator

---

## Implementation Phases

### Phase 1: Foundation
Core statistical engine + minimal API + demo data.
- Project skeleton, Docker Compose, Alembic migrations
- Database models
- `stats/frequentist.py` + `stats/power_analysis.py`
- Hash-based traffic splitting
- Basic CRUD API + event ingestion
- Demo data seeder
- Unit tests for statistical functions

### Phase 2: Advanced Statistics
The methods that separate this from toy projects.
- Sequential testing (O'Brien-Fleming)
- Always-valid inference (confidence sequences)
- Bayesian engine (Beta-Binomial, expected loss)
- Multi-armed bandit (Thompson Sampling)
- CUPED variance reduction
- Multiple testing correction
- Novelty/primacy detection
- Segment analysis + guardrails
- Property-based statistical tests

### Phase 3: Frontend Dashboard
Interactive dashboard showcasing the statistical engine.
- React + TypeScript + Vite + Tailwind + shadcn/ui
- Dashboard, Create Experiment, Results pages
- D3 custom visualizations (posteriors, sequential boundaries)
- Power Calculator page
- SSE for live updates

### Phase 4: Production Polish
Details that show production thinking.
- Background workers (periodic recomputation, guardrail alerting)
- Redis caching
- Batch ingestion, pagination, error handling
- `ARCHITECTURE.md` (doubles as system design interview answer)
- `STATISTICAL_METHODS.md` (mathematical derivations)
- Integration tests, simulation scripts
- README with architecture diagram and demo walkthrough

---

## Interview Talking Points

1. **Welch's over Student's t-test**: Does not assume equal variances -- treatment groups often have different variance.
2. **Hash-based assignment**: `sha256(experiment_id + user_id) % 10000` -- deterministic, stateless, verifiable.
3. **CUPED as a "free lunch"**: 30-50% variance reduction using existing pre-experiment data, zero bias.
4. **Always-valid over group sequential**: No need to pre-specify interim looks, at the cost of slightly wider CIs.
5. **Expected loss, not just P(B>A)**: Captures magnitude of risk, not just probability direction.
6. **BH-FDR over Bonferroni**: Controls false discovery rate (appropriate) vs family-wise error rate (too conservative for 50+ metrics).
7. **Supporting both Bayesian and Frequentist**: Different use cases require different tools -- this mirrors Google/Netflix practice.

---

## References

- Deng et al. (2013) -- CUPED variance reduction
- Johari et al. (2017) -- Peeking at A/B Tests / Always-valid inference
- Howard et al. (2021) -- Time-uniform confidence sequences
- Lan & DeMets (1983) -- Alpha spending for sequential designs
- Thompson (1933) -- Thompson Sampling
- Benjamini & Hochberg (1995) -- FDR control
