import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20), default="draft"
    )  # draft|running|paused|completed|killed
    allocation_pct: Mapped[float] = mapped_column(Float, default=100.0)
    analysis_type: Mapped[str] = mapped_column(
        String(20), default="frequentist"
    )  # frequentist|bayesian|sequential|bandit

    # Sequential config
    max_sample_size: Mapped[int | None] = mapped_column(Integer)
    interim_analyses: Mapped[int] = mapped_column(Integer, default=5)
    spending_function: Mapped[str] = mapped_column(
        String(30), default="obrien_fleming"
    )

    # Bayesian config
    prior_alpha: Mapped[float] = mapped_column(Float, default=1.0)
    prior_beta: Mapped[float] = mapped_column(Float, default=1.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    variants: Mapped[list["Variant"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class Variant(Base):
    __tablename__ = "variants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_control: Mapped[bool] = mapped_column(Boolean, default=False)
    traffic_pct: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    experiment: Mapped["Experiment"] = relationship(back_populates="variants")

    __table_args__ = (UniqueConstraint("experiment_id", "name"),)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # primary|secondary|guardrail
    data_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # binomial|continuous|count|ratio
    minimum_detectable_effect: Mapped[float | None] = mapped_column(Float)

    # CUPED
    cuped_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    cuped_covariate: Mapped[str | None] = mapped_column(String(100))

    # Guardrail
    guardrail_direction: Mapped[str | None] = mapped_column(String(10))
    guardrail_threshold: Mapped[float | None] = mapped_column(Float)

    experiment: Mapped["Experiment"] = relationship(back_populates="metrics")

    __table_args__ = (UniqueConstraint("experiment_id", "name"),)
