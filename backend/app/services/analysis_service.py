from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Experiment, Event
from app.schemas.analysis import ExperimentResults, MetricResult
from app.stats.frequentist import welch_t_test, z_test_proportions


async def analyze_experiment(db: AsyncSession, experiment_id) -> ExperimentResults:
    """Run statistical analysis on all metrics for an experiment."""
    # Load experiment with relationships
    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.variants), selectinload(Experiment.metrics))
        .where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    # Find control variant
    control = next((v for v in experiment.variants if v.is_control), None)
    if not control:
        raise ValueError("No control variant found")

    # Get treatment variants (non-control)
    treatments = [v for v in experiment.variants if not v.is_control]

    metric_results = []

    for metric in experiment.metrics:
        for treatment in treatments:
            # Fetch events for control
            control_events = await db.execute(
                select(Event.value).where(
                    Event.experiment_id == experiment_id,
                    Event.variant_id == control.id,
                    Event.metric_name == metric.name,
                )
            )
            control_values = np.array([row[0] for row in control_events.all()])

            # Fetch events for treatment
            treatment_events = await db.execute(
                select(Event.value).where(
                    Event.experiment_id == experiment_id,
                    Event.variant_id == treatment.id,
                    Event.metric_name == metric.name,
                )
            )
            treatment_values = np.array([row[0] for row in treatment_events.all()])

            if len(control_values) == 0 or len(treatment_values) == 0:
                continue

            # Run appropriate test
            if metric.data_type == "binomial":
                stat_result = z_test_proportions(
                    successes_control=int(control_values.sum()),
                    n_control=len(control_values),
                    successes_treatment=int(treatment_values.sum()),
                    n_treatment=len(treatment_values),
                )
            else:
                stat_result = welch_t_test(control_values, treatment_values)

            metric_results.append(MetricResult(
                metric_name=metric.name,
                metric_type=metric.metric_type,
                control_mean=stat_result.mean_control,
                treatment_mean=stat_result.mean_treatment,
                absolute_effect=stat_result.absolute_effect,
                relative_effect=stat_result.relative_effect,
                ci_lower=stat_result.ci_lower,
                ci_upper=stat_result.ci_upper,
                p_value=stat_result.p_value,
                is_significant=stat_result.is_significant,
                sample_size_control=stat_result.sample_size_control,
                sample_size_treatment=stat_result.sample_size_treatment,
            ))

    # Determine recommendation
    recommendation = _compute_recommendation(metric_results, experiment.status)

    return ExperimentResults(
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        status=experiment.status,
        analysis_type=experiment.analysis_type,
        metrics=metric_results,
        recommendation=recommendation,
        computed_at=datetime.now(timezone.utc),
    )


def _compute_recommendation(metrics: list[MetricResult], status: str) -> str:
    """Determine recommendation based on primary metric results."""
    if status == "draft":
        return "keep_running"

    primary_metrics = [m for m in metrics if m.metric_type == "primary"]
    if not primary_metrics:
        return "inconclusive"

    # Check if any primary metric is significant and positive
    significant_positive = any(
        m.is_significant and m.absolute_effect > 0 for m in primary_metrics
    )
    significant_negative = any(
        m.is_significant and m.absolute_effect < 0 for m in primary_metrics
    )

    # Check guardrails
    guardrail_metrics = [m for m in metrics if m.metric_type == "guardrail"]
    guardrail_violated = any(
        m.is_significant and m.absolute_effect < 0 for m in guardrail_metrics
    )

    if guardrail_violated:
        return "dont_ship"
    if significant_positive:
        return "ship"
    if significant_negative:
        return "dont_ship"
    if status == "completed":
        return "inconclusive"
    return "keep_running"
