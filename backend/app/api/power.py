from fastapi import APIRouter

from app.schemas.power import PowerCurvePoint, SampleSizeRequest, SampleSizeResponse
from app.stats.power_analysis import (
    estimate_duration,
    power_curve,
    required_sample_size,
)

router = APIRouter(prefix="/power", tags=["power"])


@router.post("/sample-size", response_model=SampleSizeResponse)
async def calculate_sample_size(data: SampleSizeRequest):
    result = required_sample_size(
        baseline_rate=data.baseline_rate,
        minimum_detectable_effect=data.minimum_detectable_effect,
        alpha=data.alpha,
        power=data.power,
        metric_type=data.metric_type,
        variance=data.variance,
    )

    # Generate power curve
    curve = power_curve(
        baseline_rate=data.baseline_rate,
        sample_size_per_variant=result.required_sample_size_per_variant,
        alpha=data.alpha,
        metric_type=data.metric_type,
        variance=data.variance,
    )

    # Estimate duration if daily traffic provided
    days = None
    if data.daily_traffic:
        days = estimate_duration(
            sample_size_per_variant=result.required_sample_size_per_variant,
            daily_traffic=data.daily_traffic,
            n_variants=data.n_variants,
        )

    return SampleSizeResponse(
        sample_size_per_variant=result.required_sample_size_per_variant,
        total_sample_size=result.total_sample_size,
        estimated_days=days,
        power=data.power,
        alpha=data.alpha,
        minimum_detectable_effect=data.minimum_detectable_effect,
        baseline_rate=data.baseline_rate,
        power_curve=[
            PowerCurvePoint(effect_size=p.effect_size, power=p.power)
            for p in curve
        ],
    )
