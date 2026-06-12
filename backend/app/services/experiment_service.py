from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Experiment, Variant, Metric
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


async def create_experiment(db: AsyncSession, data: ExperimentCreate) -> Experiment:
    """Create a new experiment with variants and metrics."""
    experiment = Experiment(
        name=data.name,
        description=data.description,
        hypothesis=data.hypothesis,
        analysis_type=data.analysis_type,
        allocation_pct=data.allocation_pct,
    )
    db.add(experiment)
    await db.flush()  # get the experiment.id

    for v in data.variants:
        variant = Variant(
            experiment_id=experiment.id,
            name=v.name,
            is_control=v.is_control,
            traffic_pct=v.traffic_pct,
            description=v.description,
        )
        db.add(variant)

    for m in data.metrics:
        metric = Metric(
            experiment_id=experiment.id,
            name=m.name,
            metric_type=m.metric_type,
            data_type=m.data_type,
            minimum_detectable_effect=m.minimum_detectable_effect,
            cuped_enabled=m.cuped_enabled,
        )
        db.add(metric)

    await db.commit()
    await db.refresh(experiment)
    # Reload with relationships
    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.variants), selectinload(Experiment.metrics))
        .where(Experiment.id == experiment.id)
    )
    return result.scalar_one()


async def get_experiment(db: AsyncSession, experiment_id) -> Experiment | None:
    """Get experiment by ID with variants and metrics loaded."""
    result = await db.execute(
        select(Experiment)
        .options(selectinload(Experiment.variants), selectinload(Experiment.metrics))
        .where(Experiment.id == experiment_id)
    )
    return result.scalar_one_or_none()


async def list_experiments(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
) -> tuple[list[Experiment], int]:
    """List experiments with pagination and optional status filter."""
    query = select(Experiment).options(
        selectinload(Experiment.variants), selectinload(Experiment.metrics)
    )
    count_query = select(func.count()).select_from(Experiment)

    if status:
        query = query.where(Experiment.status == status)
        count_query = count_query.where(Experiment.status == status)

    query = query.order_by(Experiment.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    count_result = await db.execute(count_query)

    return result.scalars().all(), count_result.scalar()


async def update_experiment(
    db: AsyncSession, experiment_id, data: ExperimentUpdate
) -> Experiment | None:
    """Update experiment fields."""
    experiment = await get_experiment(db, experiment_id)
    if not experiment:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(experiment, key, value)

    await db.commit()
    await db.refresh(experiment)
    return experiment


async def start_experiment(db: AsyncSession, experiment_id) -> Experiment | None:
    """Start an experiment (set status to running)."""
    experiment = await get_experiment(db, experiment_id)
    if not experiment:
        return None
    if experiment.status != "draft":
        raise ValueError(f"Cannot start experiment in '{experiment.status}' status")

    experiment.status = "running"
    experiment.started_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def stop_experiment(db: AsyncSession, experiment_id) -> Experiment | None:
    """Stop a running experiment."""
    experiment = await get_experiment(db, experiment_id)
    if not experiment:
        return None
    if experiment.status != "running":
        raise ValueError(f"Cannot stop experiment in '{experiment.status}' status")

    experiment.status = "completed"
    experiment.ended_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(experiment)
    return experiment
