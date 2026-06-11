"""
FinOpt Compute API

Minimal FastAPI backend for simulation and optimization compute.
CRUD operations are handled directly by Supabase via PostgREST.

This service only handles:
- POST /simulate: Queue a Monte Carlo simulation job
- POST /optimize: Queue a CVaR optimization job
- GET /health: Health check endpoint

All jobs run as background tasks and update their status in Supabase,
which the frontend monitors via Supabase Realtime.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import __version__
from api.config import Settings, get_settings
from api.services.optimization import run_optimization
from api.services.simulation import run_simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    Runs on startup and shutdown for resource initialization and cleanup.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting FinOpt API v{__version__} in {settings.environment} mode")
    logger.info(f"CORS origins: {settings.cors_origins}")

    # Reap jobs orphaned by a prior restart: background tasks die with the
    # process, so any 'running'/'pending' job at startup is dead and would
    # otherwise show a frozen progress bar forever. Never block startup on this.
    try:
        from api.supabase_client import reap_orphaned_jobs

        reaped = reap_orphaned_jobs()
        if reaped:
            logger.warning(f"Reaped {reaped} orphaned job(s) left by a prior restart")
    except Exception as exc:
        logger.error(f"Job reaper failed on startup: {exc}")

    yield

    # Shutdown
    logger.info("Shutting down FinOpt API")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinOpt Compute API",
    description=(
        "Backend service for FinOpt portfolio optimization. "
        "Handles simulation and optimization compute jobs."
    ),
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_settings_to_request(request: Request, call_next: Callable) -> Response:
    """Add settings to request state for access in routes."""
    request.state.settings = get_settings()
    response = await call_next(request)
    return response


def setup_cors(app: FastAPI, settings: Settings) -> None:
    """Configure CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Setup CORS on startup
# Note: This runs at import time
# Uses try/except to allow import even without env vars (for testing)
try:
    _settings = get_settings()
    setup_cors(app, _settings)
except Exception:
    # Default CORS for development/testing when env vars not set
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Request/Response Schemas
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """Base request for queuing a job."""

    scenario_id: str = Field(
        ...,
        description="UUID of the scenario to process",
        examples=["123e4567-e89b-12d3-a456-426614174000"],
    )
    job_id: str = Field(
        ...,
        description="UUID of the job (created by frontend in Supabase)",
        examples=["123e4567-e89b-12d3-a456-426614174001"],
    )


class SimulateRequest(JobRequest):
    """Request to queue a simulation job."""

    pass


class OptimizeRequest(JobRequest):
    """Request to queue an optimization job."""

    pass


class JobResponse(BaseModel):
    """Response after queuing a job."""

    status: str = Field(
        default="queued",
        description="Job status",
    )
    job_id: str = Field(
        ...,
        description="UUID of the queued job",
    )
    message: str = Field(
        default="Job queued successfully",
        description="Human-readable message",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = __version__
    environment: str = "development"


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version, and environment.
    Used by Render for health monitoring.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
    )


@app.post(
    "/simulate",
    response_model=JobResponse,
    tags=["jobs"],
    summary="Queue simulation job",
    responses={
        200: {"description": "Job queued successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def simulate(
    request: SimulateRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """
    Queue a Monte Carlo simulation job.

    The frontend should:
    1. Create a job record in Supabase with status='pending'
    2. Call this endpoint with scenario_id and job_id
    3. Subscribe to job updates via Supabase Realtime
    4. Fetch results from Supabase when job completes

    The simulation:
    - Fetches scenario and profile from Supabase
    - Runs Monte Carlo simulation
    - Computes summary statistics
    - Saves results to Supabase
    - Updates job status throughout
    """
    logger.info(f"Queueing simulation job {request.job_id} for scenario {request.scenario_id}")

    # Queue the job as a background task
    background_tasks.add_task(
        run_simulation,
        request.scenario_id,
        request.job_id,
    )

    return JobResponse(
        status="queued",
        job_id=request.job_id,
        message="Simulation job queued successfully",
    )


@app.post(
    "/optimize",
    response_model=JobResponse,
    tags=["jobs"],
    summary="Queue optimization job",
    responses={
        200: {"description": "Job queued successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def optimize(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """
    Queue a CVaR optimization job with goal-seeking.

    The frontend should:
    1. Create a job record in Supabase with status='pending'
    2. Call this endpoint with scenario_id and job_id
    3. Subscribe to job updates via Supabase Realtime
    4. Fetch results from Supabase when job completes

    The optimization:
    - Fetches scenario and profile from Supabase
    - Runs bilevel optimization (binary search over horizon T)
    - Finds minimum T* and optimal allocation X*
    - Saves results to Supabase
    - Updates job status throughout
    """
    logger.info(f"Queueing optimization job {request.job_id} for scenario {request.scenario_id}")

    # Queue the job as a background task
    background_tasks.add_task(
        run_optimization,
        request.scenario_id,
        request.job_id,
    )

    return JobResponse(
        status="queued",
        job_id=request.job_id,
        message="Optimization job queued successfully",
    )


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> None:
    """Handle ValueError as 400 Bad Request."""
    logger.warning(f"ValueError: {exc}")
    raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> None:
    """Handle unexpected errors."""
    logger.exception(f"Unexpected error: {exc}")
    raise HTTPException(
        status_code=500,
        detail="Internal server error. Check logs for details.",
    )


# ---------------------------------------------------------------------------
# Main (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
