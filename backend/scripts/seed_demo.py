"""
Seed script that generates realistic demo data for the A/B testing platform.

Usage: python -m scripts.seed_demo

Creates 3 demo experiments with ~100K total events:
1. "Checkout Button Color" - binomial (conversion rate) - significant result
2. "Homepage Redesign" - continuous (revenue per user) - not significant
3. "Search Algorithm v2" - binomial (click-through rate) - significant result

This uses numpy to generate realistic data and prints the data as JSON
that can be posted to the API, OR can directly insert via SQLAlchemy.
"""
import asyncio
import uuid
import numpy as np
from datetime import datetime, timezone, timedelta


# Generate demo data as dicts (API-compatible format)
def generate_demo_data():
    np.random.seed(42)
    experiments = []
    all_events = []

    # Experiment 1: Checkout Button Color (significant positive result)
    exp1_id = str(uuid.uuid4())
    control_id = str(uuid.uuid4())
    treatment_id = str(uuid.uuid4())

    experiments.append({
        "id": exp1_id,
        "name": "Checkout Button Color Test",
        "description": "Testing if a blue checkout button outperforms the green one",
        "hypothesis": "Blue checkout button will increase conversion rate by at least 2%",
        "analysis_type": "frequentist",
        "variants": [
            {"id": control_id, "name": "control_green", "is_control": True, "traffic_pct": 50.0},
            {"id": treatment_id, "name": "variant_blue", "is_control": False, "traffic_pct": 50.0},
        ],
        "metrics": [
            {"name": "conversion_rate", "metric_type": "primary", "data_type": "binomial"},
            {"name": "revenue_per_user", "metric_type": "secondary", "data_type": "continuous"},
        ],
    })

    # Generate 20,000 users per variant
    n_users = 20000
    base_time = datetime.now(timezone.utc) - timedelta(days=14)

    # Control: 10% conversion, Treatment: 12.5% conversion (significant lift)
    for i in range(n_users):
        user_id = f"user_{i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))

        # Control conversion
        converted = 1.0 if np.random.random() < 0.10 else 0.0
        all_events.append({
            "experiment_id": exp1_id, "user_id": user_id,
            "variant_id": control_id, "metric_name": "conversion_rate",
            "value": converted, "timestamp": ts.isoformat(),
        })
        # Control revenue
        revenue = np.random.exponential(25.0) if converted else 0.0
        all_events.append({
            "experiment_id": exp1_id, "user_id": user_id,
            "variant_id": control_id, "metric_name": "revenue_per_user",
            "value": round(revenue, 2), "timestamp": ts.isoformat(),
        })

    for i in range(n_users):
        user_id = f"user_{n_users + i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))

        # Treatment conversion (12.5% - significant lift)
        converted = 1.0 if np.random.random() < 0.125 else 0.0
        all_events.append({
            "experiment_id": exp1_id, "user_id": user_id,
            "variant_id": treatment_id, "metric_name": "conversion_rate",
            "value": converted, "timestamp": ts.isoformat(),
        })
        revenue = np.random.exponential(27.0) if converted else 0.0
        all_events.append({
            "experiment_id": exp1_id, "user_id": user_id,
            "variant_id": treatment_id, "metric_name": "revenue_per_user",
            "value": round(revenue, 2), "timestamp": ts.isoformat(),
        })

    # Experiment 2: Homepage Redesign (NOT significant)
    exp2_id = str(uuid.uuid4())
    ctrl2_id = str(uuid.uuid4())
    treat2_id = str(uuid.uuid4())

    experiments.append({
        "id": exp2_id,
        "name": "Homepage Redesign v2",
        "description": "Testing new homepage layout with larger hero image",
        "hypothesis": "New layout will increase time on site by 10%",
        "analysis_type": "frequentist",
        "variants": [
            {"id": ctrl2_id, "name": "control_original", "is_control": True, "traffic_pct": 50.0},
            {"id": treat2_id, "name": "variant_redesign", "is_control": False, "traffic_pct": 50.0},
        ],
        "metrics": [
            {"name": "time_on_site", "metric_type": "primary", "data_type": "continuous"},
        ],
    })

    n2 = 15000
    for i in range(n2):
        user_id = f"user_hp_{i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))
        # Control: mean=120s, std=60s
        time_val = max(0, np.random.normal(120, 60))
        all_events.append({
            "experiment_id": exp2_id, "user_id": user_id,
            "variant_id": ctrl2_id, "metric_name": "time_on_site",
            "value": round(time_val, 1), "timestamp": ts.isoformat(),
        })
    for i in range(n2):
        user_id = f"user_hp_{n2 + i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))
        # Treatment: mean=122s (tiny, non-significant difference)
        time_val = max(0, np.random.normal(122, 60))
        all_events.append({
            "experiment_id": exp2_id, "user_id": user_id,
            "variant_id": treat2_id, "metric_name": "time_on_site",
            "value": round(time_val, 1), "timestamp": ts.isoformat(),
        })

    # Experiment 3: Search Algorithm (significant)
    exp3_id = str(uuid.uuid4())
    ctrl3_id = str(uuid.uuid4())
    treat3_id = str(uuid.uuid4())

    experiments.append({
        "id": exp3_id,
        "name": "Search Algorithm v2",
        "description": "Testing improved search ranking algorithm",
        "hypothesis": "New algorithm will increase click-through rate by 3%",
        "analysis_type": "frequentist",
        "variants": [
            {"id": ctrl3_id, "name": "control_v1", "is_control": True, "traffic_pct": 50.0},
            {"id": treat3_id, "name": "variant_v2", "is_control": False, "traffic_pct": 50.0},
        ],
        "metrics": [
            {"name": "click_through_rate", "metric_type": "primary", "data_type": "binomial"},
            {"name": "searches_per_session", "metric_type": "secondary", "data_type": "continuous"},
        ],
    })

    n3 = 10000
    for i in range(n3):
        user_id = f"user_search_{i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))
        clicked = 1.0 if np.random.random() < 0.30 else 0.0
        all_events.append({
            "experiment_id": exp3_id, "user_id": user_id,
            "variant_id": ctrl3_id, "metric_name": "click_through_rate",
            "value": clicked, "timestamp": ts.isoformat(),
        })
        searches = max(1, int(np.random.poisson(3.5)))
        all_events.append({
            "experiment_id": exp3_id, "user_id": user_id,
            "variant_id": ctrl3_id, "metric_name": "searches_per_session",
            "value": float(searches), "timestamp": ts.isoformat(),
        })
    for i in range(n3):
        user_id = f"user_search_{n3 + i:06d}"
        ts = base_time + timedelta(minutes=np.random.randint(0, 14 * 24 * 60))
        clicked = 1.0 if np.random.random() < 0.35 else 0.0  # 5pp lift
        all_events.append({
            "experiment_id": exp3_id, "user_id": user_id,
            "variant_id": treat3_id, "metric_name": "click_through_rate",
            "value": clicked, "timestamp": ts.isoformat(),
        })
        searches = max(1, int(np.random.poisson(3.8)))
        all_events.append({
            "experiment_id": exp3_id, "user_id": user_id,
            "variant_id": treat3_id, "metric_name": "searches_per_session",
            "value": float(searches), "timestamp": ts.isoformat(),
        })

    return experiments, all_events


if __name__ == "__main__":
    experiments, events = generate_demo_data()
    print(f"Generated {len(experiments)} experiments")
    print(f"Generated {len(events)} events")

    for exp in experiments:
        print(f"\n  Experiment: {exp['name']}")
        print(f"  Variants: {len(exp['variants'])}")
        n_events = sum(1 for e in events if e['experiment_id'] == exp['id'])
        print(f"  Events: {n_events:,}")

    print(f"\nTotal events: {len(events):,}")
    print("\nTo use: import generate_demo_data from this module")
    print("Experiments and events are returned as dicts compatible with the API schemas.")
