"""
Supabase Client

Provides a configured Supabase client for database operations.
Uses service_role key to bypass RLS for backend operations.
"""

from functools import lru_cache
from typing import Any

from api.config import get_settings
from supabase import Client, create_client


@lru_cache
def get_supabase_client() -> Client:
    """
    Get a cached Supabase client instance.

    Uses the service_role key which bypasses RLS, suitable for backend
    operations where we need to update jobs/results regardless of user.

    Returns
    -------
    Client
        Configured Supabase client.
    """
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key
    )


def get_supabase() -> Client:
    """
    Dependency for FastAPI to inject Supabase client.

    Usage
    -----
    ```python
    @app.get("/endpoint")
    async def endpoint(supabase: Client = Depends(get_supabase)):
        result = supabase.table("profiles").select("*").execute()
    ```
    """
    return get_supabase_client()


# ---------------------------------------------------------------------------
# Job Update Helpers
# ---------------------------------------------------------------------------

def update_job(
    job_id: str,
    status: str | None = None,
    progress: int | None = None,
    step: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Update a job's status and progress in Supabase.

    This function is called from background tasks to report progress.
    Updates trigger Supabase Realtime, so the frontend receives updates.

    Parameters
    ----------
    job_id : str
        UUID of the job to update.
    status : str, optional
        New status ('pending', 'running', 'completed', 'failed', 'cancelled').
    progress : int, optional
        Progress percentage (0-100).
    step : str, optional
        Current step description (e.g., "Loading model", "Optimizing...").
    error_message : str, optional
        Error message if status is 'failed'.
    """
    client = get_supabase_client()

    update_data: dict[str, Any] = {}

    if status is not None:
        update_data["status"] = status
        if status == "running":
            from datetime import datetime, timezone
            update_data["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in ("completed", "failed", "cancelled"):
            from datetime import datetime, timezone
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    if progress is not None:
        update_data["progress"] = max(0, min(100, progress))

    if step is not None:
        update_data["current_step"] = step

    if error_message is not None:
        update_data["error_message"] = error_message

    if update_data:
        client.table("jobs").update(update_data).eq("id", job_id).execute()


def save_optimization_result(
    job_id: str,
    allocation_policy: list[list[float]],
    optimal_horizon: int,
    objective_value: float,
    feasible: bool,
    solve_time: float,
    diagnostics: dict[str, Any] | None = None,
    goal_status: list[dict[str, Any]] | None = None,
    summary_stats: dict[str, Any] | None = None,
) -> None:
    """
    Save optimization result to Supabase.

    Parameters
    ----------
    job_id : str
        UUID of the job this result belongs to.
    allocation_policy : list[list[float]]
        Optimal allocation matrix X, shape (T, M).
    optimal_horizon : int
        Minimum horizon T* found by goal seeker.
    objective_value : float
        Final objective function value.
    feasible : bool
        Whether a feasible solution was found.
    solve_time : float
        Total solve time in seconds.
    diagnostics : dict, optional
        Additional solver diagnostics.
    goal_status : list[dict], optional
        Goal satisfaction status for each goal.
    summary_stats : dict, optional
        Wealth trajectory statistics (percentiles, mean, std per account).
    """
    client = get_supabase_client()

    result_data = {
        "job_id": job_id,
        "result_type": "optimization",
        "allocation_policy": allocation_policy,
        "optimal_horizon": optimal_horizon,
        "objective_value": objective_value,
        "feasible": feasible,
        "solve_time": solve_time,
        "diagnostics": diagnostics,
        "goal_status": goal_status,
        "summary_stats": summary_stats,
    }

    client.table("results").insert(result_data).execute()


def save_simulation_result(
    job_id: str,
    summary_stats: dict[str, Any],
    goal_status: list[dict[str, Any]] | None = None,
) -> None:
    """
    Save simulation result to Supabase.

    Parameters
    ----------
    job_id : str
        UUID of the job this result belongs to.
    summary_stats : dict
        Summary statistics (mean, median, std, percentiles).
    goal_status : list[dict], optional
        Goal satisfaction status for each goal.
    """
    client = get_supabase_client()

    result_data = {
        "job_id": job_id,
        "result_type": "simulation",
        "summary_stats": summary_stats,
        "goal_status": goal_status,
    }

    client.table("results").insert(result_data).execute()


def fetch_scenario_with_profile(scenario_id: str) -> dict[str, Any]:
    """
    Fetch a scenario with its associated profile from Supabase.

    Parameters
    ----------
    scenario_id : str
        UUID of the scenario to fetch.

    Returns
    -------
    dict
        Scenario data including nested profile data under 'profiles' key.

    Raises
    ------
    ValueError
        If scenario not found.
    """
    client = get_supabase_client()

    response = (
        client.table("scenarios")
        .select("*, profiles(*)")
        .eq("id", scenario_id)
        .single()
        .execute()
    )

    if not response.data:
        raise ValueError(f"Scenario not found: {scenario_id}")

    return response.data
