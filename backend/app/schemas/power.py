from pydantic import BaseModel


class SampleSizeRequest(BaseModel):
    baseline_rate: float  # e.g. 0.10
    minimum_detectable_effect: float  # e.g. 0.01
    alpha: float = 0.05
    power: float = 0.80
    metric_type: str = "binomial"
    variance: float | None = None  # required for continuous
    daily_traffic: int | None = None
    n_variants: int = 2


class PowerCurvePoint(BaseModel):
    effect_size: float
    power: float


class SampleSizeResponse(BaseModel):
    sample_size_per_variant: int
    total_sample_size: int
    estimated_days: int | None
    power: float
    alpha: float
    minimum_detectable_effect: float
    baseline_rate: float
    power_curve: list[PowerCurvePoint]
