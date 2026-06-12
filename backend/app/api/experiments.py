import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentUpdate,
)
from app.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("/", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    data: ExperimentCreate, db: AsyncSession = Depends(get_db)
):
    experiment = await experiment_service.create_experiment(db, data)
    return experiment


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    experiments, total = await experiment_service.list_experiments(
        db, page, page_size, status
    )
    return ExperimentListResponse(
        experiments=experiments, total=total, page=page, page_size=page_size
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    experiment = await experiment_service.get_experiment(db, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: uuid.UUID,
    data: ExperimentUpdate,
    db: AsyncSession = Depends(get_db),
):
    experiment = await experiment_service.update_experiment(
        db, experiment_id, data
    )
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    try:
        experiment = await experiment_service.start_experiment(db, experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/stop", response_model=ExperimentResponse)
async def stop_experiment(
    experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    try:
        experiment = await experiment_service.stop_experiment(db, experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment
