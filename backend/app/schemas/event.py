import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class EventCreate(BaseModel):
    experiment_id: uuid.UUID
    user_id: str
    metric_name: str
    value: float
    timestamp: datetime | None = None
    metadata: dict | None = None


class EventBatchCreate(BaseModel):
    events: list[EventCreate]

    @field_validator("events")
    @classmethod
    def validate_batch_size(cls, v: list[EventCreate]) -> list[EventCreate]:
        if len(v) > 1000:
            raise ValueError("Batch size must not exceed 1000 events")
        if len(v) == 0:
            raise ValueError("Batch must contain at least one event")
        return v


class EventResponse(BaseModel):
    id: int
    experiment_id: uuid.UUID
    user_id: str
    variant_id: uuid.UUID
    metric_name: str
    value: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
