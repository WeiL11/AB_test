import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator


class VariantCreate(BaseModel):
    name: str
    is_control: bool = False
    traffic_pct: float
    description: str | None = None


class MetricCreate(BaseModel):
    name: str
    metric_type: str  # primary|secondary|guardrail
    data_type: str  # binomial|continuous|count|ratio
    minimum_detectable_effect: float | None = None
    cuped_enabled: bool = False


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    hypothesis: str | None = None
    analysis_type: str = "frequentist"
    allocation_pct: float = 100.0
    variants: list[VariantCreate]
    metrics: list[MetricCreate]

    @model_validator(mode="after")
    def validate_variants(self) -> "ExperimentCreate":
        # Validate that traffic_pct sums to approximately 100
        total_traffic = sum(v.traffic_pct for v in self.variants)
        if not (99.0 <= total_traffic <= 101.0):
            raise ValueError(
                f"Variant traffic percentages must sum to ~100, got {total_traffic}"
            )

        # Validate at least one control variant
        control_count = sum(1 for v in self.variants if v.is_control)
        if control_count == 0:
            raise ValueError("At least one variant must be marked as control")
        if control_count > 1:
            raise ValueError("Only one variant can be marked as control")

        return self


class ExperimentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    hypothesis: str | None = None
    status: str | None = None


# Response schemas


class VariantResponse(BaseModel):
    id: uuid.UUID
    name: str
    is_control: bool
    traffic_pct: float
    description: str | None

    model_config = ConfigDict(from_attributes=True)


class MetricResponse(BaseModel):
    id: uuid.UUID
    name: str
    metric_type: str
    data_type: str
    minimum_detectable_effect: float | None
    cuped_enabled: bool

    model_config = ConfigDict(from_attributes=True)


class ExperimentResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    hypothesis: str | None
    status: str
    analysis_type: str
    allocation_pct: float
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    variants: list[VariantResponse]
    metrics: list[MetricResponse]

    model_config = ConfigDict(from_attributes=True)


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    total: int
    page: int
    page_size: int
