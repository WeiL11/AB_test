# Architecture

System design document for Experimentor, a full-stack A/B testing platform. This document covers the high-level architecture, data flow, database design, statistical engine organization, traffic splitting algorithm, key design decisions with tradeoffs, and scaling considerations.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Database Design](#database-design)
4. [Statistical Engine Architecture](#statistical-engine-architecture)
5. [Traffic Splitting](#traffic-splitting)
6. [Key Design Decisions](#key-design-decisions)
7. [Scaling Considerations](#scaling-considerations)

---

## System Overview

```
+------------------+       +-------------------+       +------------------+
|                  |       |                   |       |                  |
|  React Frontend  | ----> |   FastAPI Backend  | ----> |   PostgreSQL 16  |
|  (Vite + TS)     |       |   (Uvicorn)        |       |                  |
|                  |       |                   |       +------------------+
+------------------+       +--------+----------+
                                    |                  +------------------+
                                    +----------------> |   Redis 7        |
                                    |                  |   (cache/future) |
                                    |                  +------------------+
                                    |
                           +--------v----------+
                           |                   |
                           |  Statistical       |
                           |  Engine            |
                           |  (NumPy + SciPy)   |
                           |                   |
                           +-------------------+
```

The system follows a layered architecture:

```
+-----------------------------------------------------------------------+
|                           API Layer (FastAPI)                          |
|  experiments.py | events.py | analysis.py | power.py | assign.py      |
+-----------------------------------------------------------------------+
                                    |
+-----------------------------------------------------------------------+
|                         Service Layer                                  |
|  experiment_service | event_service | analysis_service | assignment    |
+-----------------------------------------------------------------------+
                           |                    |
              +------------+           +--------+--------+
              |                        |                 |
+-------------v---------+  +-----------v---+  +----------v----------+
|   Data Access Layer   |  |  Stats Engine |  |  Assignment Engine  |
|   SQLAlchemy 2.0      |  |  Pure funcs   |  |  SHA-256 hashing    |
|   (async)             |  |  (stateless)  |  |  (stateless)        |
+-----------------------+  +---------------+  +---------------------+
              |
+-------------v---------+
|   PostgreSQL 16       |
|   experiments         |
|   variants            |
|   metrics             |
|   events              |
|   assignments         |
|   analysis_results    |
+-----------------------+
```

**Key architectural properties:**

- **Stateless computation.** The statistical engine and assignment engine are pure functions with no side effects. Given the same inputs, they always produce the same outputs. This makes them trivially testable (291 unit tests) and horizontally scalable.

- **Async I/O throughout.** SQLAlchemy 2.0 async with asyncpg, FastAPI's native async handlers. Database queries never block the event loop.

- **Separation of concerns.** API routes handle HTTP concerns (validation, serialization, error codes). Services handle business logic (state machine transitions, authorization). Stats modules handle mathematics (no database awareness).

---

## Data Flow

### Experiment Lifecycle

An experiment goes through a well-defined state machine:

```
                  create
                    |
                    v
               +---------+
               |  DRAFT  |
               +----+----+
                    |
                    | start (validates: has control, traffic sums to 100)
                    v
               +---------+
               | RUNNING |<--------+
               +----+----+        |
                    |              |
                    | stop         | (future: resume)
                    v              |
             +-----------+         |
             | COMPLETED |---------+
             +-----------+
```

### End-to-End Data Flow

```
Phase 1: SETUP                  Phase 2: COLLECTION              Phase 3: ANALYSIS
========================        ========================         ========================

POST /experiments/              GET /assign/{exp}/{user}         GET /experiments/{id}/results
  |                               |                                |
  v                               v                                v
Create Experiment               SHA-256(exp_id.user_id)          Fetch experiment + metrics
  - name, hypothesis              |                                |
  - variants (control +           v                                v
    treatments with               bucket = hash % 10000          For each metric:
    traffic %)                    |                                |
  - metrics (primary,             v                                v
    secondary, guardrail)       Walk variants by sorted ID       Fetch events grouped by
  |                             accumulate traffic_pct            variant
  v                             return first match                |
Store in PostgreSQL               |                                v
                                  v                              Run statistical test
POST /{id}/start                Store assignment record          (welch_t_test or
  |                                                              z_test_proportions)
  v                             POST /events/                      |
Validate state + config           |                                v
Set status = running              v                              Compute recommendation
Set started_at                  Validate experiment exists        (ship / dont_ship /
                                Store event:                      keep_running / inconclusive)
                                  - experiment_id                  |
                                  - variant_id                     v
                                  - metric_name                  Return ExperimentResults
                                  - value (0/1 for binary,         - per-metric CIs
                                    continuous otherwise)          - p-values
                                  - timestamp                      - effect sizes
                                  - metadata (JSON)                - recommendation
```

### Analysis Pipeline Detail

When the `/results` endpoint is called:

```
1. Load experiment with variants + metrics (eager loading via selectinload)
2. Identify control variant (is_control = true)
3. For each metric:
   a. Query events WHERE experiment_id AND metric_name, grouped by variant_id
   b. If metric is binomial:
        - Count successes (sum of 1s) and total observations per variant
        - Run z_test_proportions(successes_c, n_c, successes_t, n_t)
   c. If metric is continuous:
        - Collect raw observation arrays per variant
        - Run welch_t_test(control_values, treatment_values)
   d. Package into MetricResult (effect, CI, p-value, significance)
4. Compute recommendation:
   a. If any guardrail metric is significantly negative -> "dont_ship"
   b. If primary metric is significantly positive -> "ship"
   c. If primary metric is significantly negative -> "dont_ship"
   d. If experiment is completed but no significance -> "inconclusive"
   e. Otherwise -> "keep_running"
5. Return ExperimentResults with all metrics + recommendation
```

---

## Database Design

### Entity-Relationship Diagram

```
+------------------+       +------------------+       +------------------+
|   experiments    |       |    variants      |       |    metrics       |
+------------------+       +------------------+       +------------------+
| id          UUID | PK    | id          UUID | PK    | id          UUID | PK
| name     VARCHAR |       | experiment_id FK |------>| experiment_id FK |
| description TEXT |<------| name     VARCHAR |       | name     VARCHAR |
| hypothesis  TEXT |  1:N  | is_control  BOOL |       | metric_type  STR |
| status   VARCHAR |       | traffic_pct  FLT |       | data_type    STR |
| allocation_pct   |       | description TEXT |       | mde         FLOAT|
| analysis_type    |       +------------------+       | cuped_enabled    |
| max_sample_size  |              |                   | cuped_covariate  |
| interim_analyses |              |                   | guardrail_dir    |
| spending_func    |              |                   | guardrail_thresh |
| prior_alpha      |              |                   +------------------+
| prior_beta       |              |
| created_at    TS |              |
| started_at    TS |              |
| ended_at     TS  |              |
+------------------+              |
        |                         |
        |    +--------------------+--------------------+
        |    |                                         |
        v    v                                         v
+------------------+                          +------------------+
|     events       |                          |   assignments    |
+------------------+                          +------------------+
| id       BIGINT  | PK                      | id       BIGINT  | PK
| experiment_id FK |                          | experiment_id FK |
| user_id  VARCHAR |                          | user_id  VARCHAR |
| variant_id   FK  |                          | variant_id   FK  |
| metric_name  STR |                          | assigned_at   TS |
| value     FLOAT  |                          | segments    JSON |
| timestamp     TS |                          +------------------+
| metadata    JSON |                          UNIQUE(experiment_id, user_id)
+------------------+
INDEX(experiment_id, metric_name)
INDEX(timestamp)
```

### Table Design Rationale

**experiments** -- The core entity. Stores configuration for all analysis types (frequentist, bayesian, sequential, bandit) in a single table. This is a deliberate choice over STI or separate tables: experiment configuration is read far more than written, and the column overhead is minimal. The `status` field enforces the state machine (draft -> running -> completed).

**variants** -- Separated from experiments because an experiment can have 2+ variants. Each variant has a `traffic_pct` (0-100) and an `is_control` flag. The UNIQUE constraint on `(experiment_id, name)` prevents duplicate variant names within an experiment.

**metrics** -- Also separated via 1:N because an experiment typically has 3-10 metrics. The `metric_type` (primary/secondary/guardrail) drives the recommendation logic. The `data_type` (binomial/continuous/count/ratio) determines which statistical test to use. CUPED configuration is per-metric because not all metrics have suitable covariates.

**events** -- The high-volume table. Uses BIGINT auto-increment primary key instead of UUID for write performance (B-tree friendly sequential inserts vs. random UUID inserts). The composite index on `(experiment_id, metric_name)` is critical: every analysis query filters by both columns. The timestamp index supports time-windowed queries for novelty detection.

**assignments** -- Records which variant each user was assigned to. The UNIQUE constraint on `(experiment_id, user_id)` enforces one assignment per user per experiment. The `segments` JSON column stores user attributes (country, platform, cohort) for segment analysis without requiring a separate table or schema migration for each new attribute.

**analysis_results** -- Caches computed analysis results to avoid re-running expensive statistical computations on every page load. Keyed by `(experiment_id, metric_id, variant_id)`.

### Indexing Strategy

The two indexes on the `events` table are the most important:

1. **`idx_events_experiment_metric(experiment_id, metric_name)`** -- Every analysis query does `SELECT value FROM events WHERE experiment_id = ? AND metric_name = ?`. This composite index makes it an index-only scan.

2. **`idx_events_timestamp(timestamp)`** -- Used for novelty detection (daily bucketing) and potential time-range queries. Without this index, novelty detection would require a sequential scan.

The assignments table's UNIQUE constraint on `(experiment_id, user_id)` doubles as an index for the lookup-or-insert pattern in the assignment service.

---

## Statistical Engine Architecture

The stats package is organized as independent, pure-function modules with no database dependencies:

```
stats/
|-- base.py                 # Shared result dataclasses (FrequentistResult, PowerResult)
|-- frequentist.py          # welch_t_test(), z_test_proportions(), chi_squared_test()
|-- sequential.py           # GroupSequentialTest class, sequential_z_test()
|-- confidence_sequence.py  # confidence_sequence(), compute_cs_over_time()
|-- bayesian.py             # beta_binomial_test(), normal_test()
|-- bandit.py               # ThompsonSampling, UCB1, EpsilonGreedy classes
|-- cuped.py                # cuped_adjust()
|-- multiple_testing.py     # bonferroni(), holm(), benjamini_hochberg()
|-- power_analysis.py       # required_sample_size(), compute_power(), power_curve()
|-- novelty_detection.py    # detect_novelty_effect(), compute_daily_effects()
+-- segment_analysis.py     # analyze_segments() + heterogeneity test
```

### Design Pattern: Frozen Dataclasses

Every statistical method returns a frozen (immutable) dataclass:

```python
@dataclass(frozen=True)
class FrequentistResult:
    mean_control: float
    mean_treatment: float
    absolute_effect: float
    relative_effect: Optional[float]
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    ...
```

**Why frozen dataclasses?**
- **Immutability** prevents accidental mutation of results during downstream processing.
- **Hashability** allows results to be used as dictionary keys or cached in sets.
- **Self-documenting** -- the field names and types serve as documentation.
- **Serialization** -- trivially converted to dicts via `dataclasses.asdict()` for JSON responses.

### Composition

The modules compose cleanly:

```
segment_analysis.py
    |-- calls frequentist.welch_t_test() per segment
    |-- calls frequentist.z_test_proportions() for binomial segments
    +-- calls multiple_testing.apply_correction() on segment p-values

analysis_service.py (service layer)
    |-- calls frequentist.welch_t_test() or z_test_proportions()
    +-- produces MetricResult -> ExperimentResults

cuped.py
    +-- returns CUPEDResult with adjusted values
        (caller then passes adjusted arrays to any test)
```

### Pure Functions with NumPy/SciPy

The entire statistical engine depends only on NumPy and SciPy. No database imports, no HTTP imports, no framework dependencies. This makes the stats package:

- **Unit-testable in isolation** (291 tests, no fixtures or mocking needed)
- **Reusable** outside the web application (e.g., Jupyter notebooks, CLI tools)
- **Verifiable** against textbook formulas and reference implementations

---

## Traffic Splitting

### Algorithm

```python
def assign_variant(experiment_id, user_id, variants):
    hash_input = f"{experiment_id}.{user_id}".encode("utf-8")
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    bucket = int(hash_hex, 16) % 10000

    cumulative = 0.0
    for variant in sorted(variants, key=lambda v: str(v["id"])):
        cumulative += variant["traffic_pct"] * 100
        if bucket < cumulative:
            return variant

    return sorted_variants[-1]  # fallback
```

### Properties

**Deterministic.** `SHA-256(experiment_id + "." + user_id)` always produces the same hash. The same user will always be assigned to the same variant for the same experiment. No database lookup required, no race conditions, no server-side state.

**Uniform distribution.** SHA-256 is a cryptographic hash with excellent uniformity. Modulo 10000 introduces negligible bias (2^256 mod 10000 bias is ~10^-73). The 10000-bucket resolution supports traffic splits as granular as 0.01%.

**Cross-experiment independence.** Because the experiment ID is part of the hash input, user `U` might be in control for experiment A and treatment for experiment B. This prevents systematic assignment patterns across experiments.

**Sorted-by-ID variant walking.** Variants are sorted by their UUID before accumulating traffic percentages. This ensures that if a new variant is added or removed, the assignment boundary shifts predictably rather than depending on insertion order.

### Why Hash-Based Over Random Assignment

| Property | Hash-Based | Random + Lookup |
|---|---|---|
| Consistency | Same user always gets same variant | Requires database lookup |
| Statelessness | No state needed | Must persist assignment |
| Debuggability | Given (exp_id, user_id), anyone can verify | Must query the database |
| Scalability | O(1) compute, no I/O | O(1) DB lookup, but requires connection |
| Re-assignability | Cannot change assignment without changing ID | Can update DB record |

The tradeoff: hash-based assignment cannot be overridden for individual users (e.g., for QA testing). This is acceptable for this system. Production platforms like Google Optimize use hash-based assignment with a manual override table for special cases.

---

## Key Design Decisions

### 1. Welch's t-test vs. Student's t-test

**Decision:** Welch's t-test (does not assume equal variances).

**Rationale:** Student's t-test assumes `sigma_control = sigma_treatment`. In A/B tests, this assumption is almost always wrong: the treatment changes the metric distribution, which changes its variance. Welch's t-test uses the Welch-Satterthwaite degrees of freedom approximation, which is slightly conservative (fewer degrees of freedom) but never anticonservative. The power loss is negligible with sample sizes typical in A/B testing (1000+), and the test is valid regardless of variance heterogeneity.

**Tradeoff:** Welch's t-test has slightly less power than Student's when variances truly are equal. At typical A/B testing sample sizes, this difference is <0.1% power -- not worth the risk of an invalid test.

### 2. Hash-Based vs. Random Assignment

**Decision:** Deterministic SHA-256 hashing.

**Rationale:** See the Traffic Splitting section above. The key advantage is statelessness: assignment can be computed at any layer (client SDK, CDN edge, backend) without a database round-trip. This is the same approach used by Google, Meta, and Netflix.

**Tradeoff:** Cannot override individual assignments without changing the user ID or experiment ID. Production systems solve this with an override table checked before the hash.

### 3. CUPED as a "Free Lunch"

**Decision:** Implement CUPED variance reduction as an opt-in per-metric feature.

**Rationale:** CUPED reduces variance by `1 - rho^2` where `rho` is the Pearson correlation between the pre-experiment covariate and the experiment metric. With typical correlations of 0.5-0.7, this means 25-50% variance reduction -- equivalent to doubling the sample size for free. The adjustment `Y_adj = Y - theta * (X - E[X])` is provably unbiased because `E[X - E[X]] = 0` regardless of the treatment assignment.

**Why it is not the default:** CUPED requires pre-experiment data for each user, which is not always available (e.g., new users, new metrics). It also requires that the covariate `X` is measured before the experiment starts -- using data from during the experiment would introduce bias. The `cuped_covariate` field in the Metric model explicitly names what pre-experiment data to use.

**Tradeoff:** Adds complexity to the data pipeline (must join pre-experiment data). If the covariate has low correlation with the metric, CUPED provides minimal benefit but adds computational cost.

### 4. Always-Valid Confidence Sequences vs. Group Sequential Testing

**Decision:** Implement both.

**Rationale:** They solve the same problem (valid inference under continuous monitoring) but have different tradeoffs:

| Property | Group Sequential (O'Brien-Fleming) | Confidence Sequences (mSPRT) |
|---|---|---|
| Planning required | Must pre-specify number of looks | No pre-specification needed |
| Stopping flexibility | Can only look at pre-planned times | Can look at any time |
| Statistical power | Higher (tighter bounds at planned looks) | Lower (wider bounds due to any-time guarantee) |
| Implementation complexity | Medium | Medium |
| When to use | Known experiment duration, planned check-ins | Unknown duration, continuous dashboards |

O'Brien-Fleming is appropriate when the experiment has a planned duration and the team will check results at scheduled intervals. Confidence sequences are appropriate when the team monitors a real-time dashboard and might stop the experiment at any time.

**Tradeoff:** Supporting both adds code but does not add user-facing complexity -- the choice is made once per experiment via `analysis_type`.

### 5. Expected Loss vs. P(B>A) for Bayesian Decisions

**Decision:** Use both P(B>A) AND expected loss as decision criteria.

**Rationale:** P(B>A) alone is insufficient for decision-making. Consider: P(B>A) = 96% sounds great, but if B is only 0.001% better when it wins and 5% worse when it loses, the expected loss of shipping B is high. Expected loss = `E[max(0, A - B)]` directly quantifies the business cost of a wrong decision.

The decision rule requires BOTH conditions:
```
Ship if: P(B > A) > 0.95 AND expected_loss < 0.001
```

This means: "We are 95% sure B is better, AND even if we are wrong, we lose less than 0.1%."

**Tradeoff:** The dual threshold can be confusing for users unfamiliar with Bayesian decision theory. The recommendation engine translates this into plain language ("ship", "dont_ship", "keep_running").

### 6. Benjamini-Hochberg FDR vs. Bonferroni

**Decision:** Support all three (Bonferroni, Holm, BH-FDR) with Holm as default.

**Rationale:**
- **Bonferroni** (`alpha_adj = alpha / m`) controls FWER but is overly conservative for many metrics. With 20 metrics, each individual test must pass at alpha = 0.0025.
- **Holm** (step-down) controls FWER and is uniformly more powerful than Bonferroni -- there is no reason to ever prefer Bonferroni over Holm except for simplicity.
- **BH-FDR** controls the false discovery rate, not FWER. It is much more powerful but allows some false positives among rejections. Best for exploratory analysis with many secondary metrics.

**When to use each:**
- Primary + guardrail metrics (2-5 tests): Holm (FWER control matters, few enough tests that power loss is small)
- Many secondary/exploratory metrics (10+): BH-FDR (FWER is too conservative; controlling FDR at 5% is the right tradeoff)

**Tradeoff:** BH-FDR assumes independence or positive dependence among test statistics. In practice, A/B test metrics (conversion, revenue, engagement) are often positively correlated, so this assumption holds.

### 7. Supporting Both Bayesian and Frequentist

**Decision:** Support both paradigms as first-class citizens.

**Rationale:** Different teams and organizations have different statistical cultures. Frequentist testing is the industry standard, well-understood by product managers, and required by regulatory frameworks. Bayesian testing directly answers business questions ("probability B is better") and handles early stopping naturally.

Rather than forcing a philosophical choice, the platform lets the user choose `analysis_type` at experiment creation time. This mirrors how mature experimentation platforms (Statsig, Eppo) work in practice.

**Tradeoff:** Maintaining two analysis pipelines doubles the testing surface. However, since the statistical engine is pure functions, each module is tested independently.

---

## Scaling Considerations

The current architecture handles single-team experimentation well. Here is what would change at Google/Netflix/Microsoft scale (millions of users, thousands of concurrent experiments, billions of events per day).

### Event Ingestion

**Current:** Events are written directly to PostgreSQL via INSERT statements. Batch endpoint supports up to 1000 events per request.

**At scale:**
```
Current:
  Client -> API -> PostgreSQL (direct INSERT)

At scale:
  Client -> API -> Kafka/Kinesis -> Stream Processor -> ClickHouse/BigQuery
                        |
                        v
                   Real-time aggregation
                   (per-variant, per-metric, per-hour)
```

- **Message queue (Kafka/Kinesis)** decouples ingestion from storage. The API returns 200 immediately after enqueuing, and events are written in bulk by a consumer.
- **Columnar store (ClickHouse, BigQuery, Druid)** replaces PostgreSQL for event storage. Row-oriented PostgreSQL is not designed for analytical workloads over billions of rows.
- **Pre-aggregation** computes running sums, counts, and sum-of-squares per (experiment, variant, metric, time_bucket). Analysis reads aggregates, not raw events. This reduces analysis query time from O(n_events) to O(n_variants * n_metrics * n_time_buckets).

### Statistical Computation

**Current:** Analysis is computed on demand when the `/results` endpoint is called. Events are fetched from PostgreSQL and passed to NumPy.

**At scale:**
- **Pre-aggregated sufficient statistics.** For Welch's t-test, you only need `(n, sum, sum_of_squares)` per variant per metric -- not the raw events. These can be maintained incrementally as events arrive.
- **Incremental Bayesian updates.** The Beta-Binomial posterior `Beta(alpha + successes, beta + failures)` can be updated incrementally without re-processing historical events.
- **Materialized analysis results** cached in Redis or a results table, recomputed on a schedule (e.g., every hour) rather than on demand.
- **Async computation pipeline.** Long-running analyses (CUPED with millions of users, segment analysis across hundreds of segments) would run as background tasks (Celery/Temporal) rather than in the request path.

### Traffic Splitting

**Current:** Hash-based assignment is already horizontally scalable (pure computation, no I/O).

**At scale:**
- **Consistent hashing with salt rotation** allows gradual rollouts: change the salt to re-randomize a fraction of users.
- **Mutual exclusion layers** ensure users are not in conflicting experiments (e.g., two experiments changing the same page). Google's experiment infrastructure uses "layers" where each layer is a separate hash space.
- **Feature flags integration.** Assignment decisions are cached in a feature flag service (LaunchDarkly-like) for sub-millisecond client-side resolution.

### Database

**Current:** Single PostgreSQL 16 instance with two indexes on the events table.

**At scale:**
- **Partition events by experiment_id** (or by time). Each experiment's events are in a separate partition, making per-experiment queries efficient and old experiment data easy to archive.
- **Read replicas** for analysis queries to avoid impacting event ingestion throughput.
- **Separate OLTP and OLAP stores.** PostgreSQL for experiment metadata (low-volume, transactional). ClickHouse/BigQuery for events (high-volume, analytical).

```
OLTP (PostgreSQL)                    OLAP (ClickHouse)
+------------------+                 +---------------------------+
| experiments      |                 | events (partitioned       |
| variants         |                 |   by experiment_id,       |
| metrics          |                 |   ordered by timestamp)   |
| assignments      |                 |                           |
| analysis_results |                 | pre_aggregated_stats      |
+------------------+                 |   (n, sum, sum_sq per     |
                                     |    variant/metric/hour)   |
                                     +---------------------------+
```

### Frontend

**Current:** TanStack Query with polling for results updates.

**At scale:**
- **WebSocket/SSE** for real-time results streaming as analysis is recomputed.
- **Cached analysis snapshots** served from Redis/CDN to avoid hitting the analysis pipeline on every dashboard load.
- **Pagination and virtualization** for experiment lists with thousands of active experiments.

### Multi-Tenancy

**Current:** Single-tenant (no authentication, no organization scoping).

**At scale:**
- **Row-level security** with organization/team scoping on all tables.
- **Rate limiting** per organization for event ingestion.
- **Experiment quotas** to prevent resource exhaustion.
- **Audit logging** for experiment creation, state transitions, and result access.

---

## Summary

The architecture makes deliberate choices that favor correctness and testability at the current scale while maintaining clear paths to horizontal scaling. The statistical engine's pure-function design means it can be extracted into a library, called from a streaming pipeline, or wrapped in a microservice without any changes. The hash-based assignment algorithm is already production-scale. The main scaling bottleneck is event storage and analysis computation, both of which have well-understood solutions (pre-aggregation, columnar stores, incremental statistics).
