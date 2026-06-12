from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.event import EventBatchCreate, EventCreate, EventResponse
from app.services import event_service

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/", response_model=EventResponse, status_code=201)
async def create_event(data: EventCreate, db: AsyncSession = Depends(get_db)):
    try:
        event = await event_service.ingest_event(db, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return event


@router.post("/batch", status_code=201)
async def create_events_batch(
    data: EventBatchCreate, db: AsyncSession = Depends(get_db)
):
    count = await event_service.ingest_batch(db, data.events)
    return {"ingested": count, "total": len(data.events)}
