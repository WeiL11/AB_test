from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event import Assignment, Event
from app.schemas.event import EventCreate


async def ingest_event(db: AsyncSession, data: EventCreate) -> Event:
    """Ingest a single event. Looks up the user's variant assignment."""
    # Get user's assignment
    result = await db.execute(
        select(Assignment).where(
            Assignment.experiment_id == data.experiment_id,
            Assignment.user_id == data.user_id,
        )
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise ValueError(
            f"User {data.user_id} not assigned to experiment {data.experiment_id}"
        )

    event = Event(
        experiment_id=data.experiment_id,
        user_id=data.user_id,
        variant_id=assignment.variant_id,
        metric_name=data.metric_name,
        value=data.value,
        timestamp=data.timestamp,
        metadata_=data.metadata or {},
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event


async def ingest_batch(db: AsyncSession, events: list[EventCreate]) -> int:
    """Ingest a batch of events. Returns count of successfully ingested events."""
    # Pre-fetch all needed assignments
    count = 0
    for data in events:
        try:
            result = await db.execute(
                select(Assignment).where(
                    Assignment.experiment_id == data.experiment_id,
                    Assignment.user_id == data.user_id,
                )
            )
            assignment = result.scalar_one_or_none()
            if not assignment:
                continue  # skip unassigned users

            event = Event(
                experiment_id=data.experiment_id,
                user_id=data.user_id,
                variant_id=assignment.variant_id,
                metric_name=data.metric_name,
                value=data.value,
                timestamp=data.timestamp,
                metadata_=data.metadata or {},
            )
            db.add(event)
            count += 1
        except Exception:
            continue

    await db.commit()
    return count
