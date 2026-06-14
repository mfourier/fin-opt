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


def reap_orphaned_jobs(older_than_seconds: int = 0) -> int:
    """
    Mark stuck jobs as failed.

    Background tasks run in-process, so a job in 'pending'/'running' cannot
    survive a process restart. On startup, any such job is orphaned (e.g. the
    instance was restarted by the platform mid-job) and would otherwise stay
    'running' forever, leaving the UI with a frozen progress bar. This marks
    them 'failed'.

    Parameters
    ----------
    older_than_seconds : int, default 0
        Only reap jobs created more than this many seconds ago. 0 reaps every
        pending/running job — correct on startup, where a fresh process means
        nothing is genuinely running (single-instance deployment).

    Returns
    -------
    int
        Number of jobs reaped.
    """
    from datetime import datetime, timedelta, timezone

    client = get_supabase_client()
    now = datetime.now(timezone.utc)

    query = (
        client.table("jobs")
        .update(
            {
                "status": "failed",
                "error_message": "Orphaned: the server restarted while this job was running.",
                "completed_at": now.isoformat(),
            }
        )
        .in_("status", ["pending", "running"])
    )
    if older_than_seconds > 0:
        cutoff = (now - timedelta(seconds=older_than_seconds)).isoformat()
        query = query.lt("created_at", cutoff)

    response = query.execute()
    return len(response.data or [])


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


def authorize_job_for_user(
    job_id: str,
    scenario_id: str,
    user_id: str,
    expected_job_type: str,
) -> None:
    """
    Ensure a queued job belongs to the authenticated user and matches the API call.

    Parameters
    ----------
    job_id : str
        UUID of the job row created by the frontend.
    scenario_id : str
        UUID of the scenario referenced in the API request.
    user_id : str
        Authenticated Supabase user id from the bearer token.
    expected_job_type : str
        Expected job_type for the endpoint ('simulation' or 'optimization').

    Raises
    ------
    PermissionError
        If the job/scenario does not belong to the authenticated user.
    ValueError
        If the job/scenario are inconsistent or not queueable.
    """
    client = get_supabase_client()

    job_response = (
        client.table("jobs")
        .select("id, scenario_id, job_type, status")
        .eq("id", job_id)
        .single()
        .execute()
    )
    job = job_response.data
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    if job.get("scenario_id") != scenario_id:
        raise PermissionError("Job does not belong to the requested scenario")
    if job.get("job_type") != expected_job_type:
        raise ValueError(
            f"Job type mismatch: expected {expected_job_type}, got {job.get('job_type')!r}"
        )
    if job.get("status") != "pending":
        raise ValueError(
            f"Job must be pending before queueing, got {job.get('status')!r}"
        )

    scenario_response = (
        client.table("scenarios")
        .select("id, is_demo, profiles(user_id)")
        .eq("id", scenario_id)
        .single()
        .execute()
    )
    scenario = scenario_response.data
    if not scenario:
        raise ValueError(f"Scenario not found: {scenario_id}")

    if scenario.get("is_demo"):
        raise PermissionError("Demo scenarios cannot be queued for compute jobs")

    profile_data = scenario.get("profiles")
    if isinstance(profile_data, list):
        profile_data = profile_data[0] if profile_data else None

    owner_user_id = profile_data.get("user_id") if isinstance(profile_data, dict) else None
    if owner_user_id != user_id:
        raise PermissionError("You do not have access to this scenario")
