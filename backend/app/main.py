from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.api.errors import APIError, api_error_handler, validation_error_handler

from app.config import settings
from app.db import init_db
from app.api import analysis, assign, events, experiments, health, power


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.DEBUG:
        await init_db()
    yield


app = FastAPI(
    title="Experimentor",
    description="Production-grade A/B testing and experimentation platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)

app.include_router(health.router, prefix=settings.API_PREFIX, tags=["health"])
app.include_router(experiments.router, prefix=settings.API_PREFIX)
app.include_router(events.router, prefix=settings.API_PREFIX)
app.include_router(analysis.router, prefix=settings.API_PREFIX)
app.include_router(power.router, prefix=settings.API_PREFIX)
app.include_router(assign.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root() -> dict:
    return {"status": "healthy", "service": "experimentor"}
