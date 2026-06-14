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

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api import __version__
from api.config import Settings, get_settings
from api.supabase_client import authorize_job_for_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def run_simulation(scenario_id: str, job_id: str) -> None:
    """Lazy-import and execute the simulation worker."""
    from api.services.simulation import run_simulation as _run_simulation

    _run_simulation(scenario_id, job_id)


def run_optimization(scenario_id: str, job_id: str) -> None:
    """Lazy-import and execute the optimization worker."""
    from api.services.optimization import run_optimization as _run_optimization

    _run_optimization(scenario_id, job_id)


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


class AuthenticatedUser(BaseModel):
    """Minimal authenticated Supabase user payload."""

    id: str
    email: str | None = None


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthenticatedUser:
    """Validate a Supabase bearer token and return the authenticated user."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        response = httpx.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {credentials.credentials}",
                "apikey": settings.supabase_anon_key,
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to validate the current session with Supabase Auth",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    payload = response.json()
    user_id = payload.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid auth payload")

    return AuthenticatedUser(id=user_id, email=payload.get("email"))


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
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
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
    try:
        authorize_job_for_user(
            job_id=request.job_id,
            scenario_id=request.scenario_id,
            user_id=current_user.id,
            expected_job_type="simulation",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Queueing simulation job %s for scenario %s (user=%s)",
        request.job_id,
        request.scenario_id,
        current_user.id,
    )

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
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
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
    try:
        authorize_job_for_user(
            job_id=request.job_id,
            scenario_id=request.scenario_id,
            user_id=current_user.id,
            expected_job_type="optimization",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Queueing optimization job %s for scenario %s (user=%s)",
        request.job_id,
        request.scenario_id,
        current_user.id,
    )

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
