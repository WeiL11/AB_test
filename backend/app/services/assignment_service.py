import uuid as uuid_mod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.assignment.hasher import assign_variant
from app.models.event import Assignment
from app.models.experiment import Variant


async def get_or_create_assignment(
    db: AsyncSession,
    experiment_id,
    user_id: str,
    segments: dict | None = None,
) -> Assignment:
    """Get existing assignment or create new one using hash-based splitting."""
    # Check for existing assignment
    result = await db.execute(
        select(Assignment).where(
            Assignment.experiment_id == experiment_id,
            Assignment.user_id == user_id,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing

    # Get experiment variants
    variant_result = await db.execute(
        select(Variant).where(Variant.experiment_id == experiment_id)
    )
    variants = variant_result.scalars().all()
    if not variants:
        raise ValueError(f"No variants found for experiment {experiment_id}")

    # Determine assignment via hash
    variant_dicts = [{"id": str(v.id), "traffic_pct": v.traffic_pct} for v in variants]
    selected = assign_variant(str(experiment_id), user_id, variant_dicts)

    # Create assignment record
    assignment = Assignment(
        experiment_id=experiment_id,
        user_id=user_id,
        variant_id=uuid_mod.UUID(selected["id"]),
        segments=segments or {},
    )
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)
    return assignment
