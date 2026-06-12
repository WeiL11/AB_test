import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import assignment_service

router = APIRouter(prefix="/assign", tags=["assignment"])


@router.get("/{experiment_id}/{user_id}")
async def get_assignment(
    experiment_id: uuid.UUID,
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        assignment = await assignment_service.get_or_create_assignment(
            db, experiment_id, user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "experiment_id": str(assignment.experiment_id),
        "user_id": assignment.user_id,
        "variant_id": str(assignment.variant_id),
        "assigned_at": (
            assignment.assigned_at.isoformat()
            if assignment.assigned_at
            else None
        ),
    }
