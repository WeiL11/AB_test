import uuid
from datetime import datetime

from pydantic import BaseModel


class MetricResult(BaseModel):
    metric_name: str
    metric_type: str  # primary|secondary|guardrail
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float  # percentage
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int


class ExperimentResults(BaseModel):
    experiment_id: uuid.UUID
    experiment_name: str
    status: str
    analysis_type: str
    metrics: list[MetricResult]
    recommendation: str  # "ship" | "dont_ship" | "keep_running" | "inconclusive"
    computed_at: datetime
