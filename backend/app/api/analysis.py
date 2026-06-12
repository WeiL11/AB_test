import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.analysis import ExperimentResults
from app.services import analysis_service

router = APIRouter(prefix="/experiments", tags=["analysis"])


@router.get("/{experiment_id}/results", response_model=ExperimentResults)
async def get_results(
    experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    try:
        results = await analysis_service.analyze_experiment(db, experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return results
