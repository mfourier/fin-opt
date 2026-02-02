"""
Optimization Service

Handles optimization jobs: fetching scenario data, running CVaR optimization
with goal-seeking, computing results, and saving to Supabase.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

from api.supabase_client import (
    fetch_scenario_with_profile,
    save_optimization_result,
    update_job,
)
from api.services.reconstruction import reconstruct_from_scenario

logger = logging.getLogger(__name__)


def compute_goal_status_from_result(
    opt_result,
    model,
    start_date: date,
) -> list[dict[str, Any]]:
    """
    Compute goal satisfaction status from optimization result.

    Parameters
    ----------
    opt_result : OptimizationResult
        Result from CVaROptimizer.
    model : FinancialModel
        The financial model used.
    start_date : date
        Simulation start date.

    Returns
    -------
    list[dict]
        Goal status for each goal.
    """
    from finopt import IntermediateGoal, TerminalGoal

    status_list = []

    for goal in opt_result.goals:
        # Get account name
        if isinstance(goal.account, str):
            account_name = goal.account
        else:
            account_name = model.accounts[goal.account].name

        if isinstance(goal, IntermediateGoal):
            goal_type = "intermediate"
            goal_desc = f"{account_name} by {goal.date.isoformat()}"
        else:
            goal_type = "terminal"
            goal_desc = f"{account_name} at horizon T={opt_result.T}"

        status_list.append({
            "goal": goal_desc,
            "type": goal_type,
            "account": account_name,
            "threshold": goal.threshold,
            "required_confidence": goal.confidence,
            # CVaR optimization guarantees satisfaction if feasible
            "satisfied": opt_result.feasible,
        })

    return status_list


async def run_optimization(scenario_id: str, job_id: str) -> None:
    """
    Run CVaR optimization with goal-seeking for a scenario.

    This is the main entry point called from FastAPI background tasks.
    It fetches the scenario, runs bilevel optimization (outer: horizon search,
    inner: convex allocation), and saves results.

    Parameters
    ----------
    scenario_id : str
        UUID of the scenario to optimize.
    job_id : str
        UUID of the job for progress tracking.
    """
    try:
        # Update job status
        update_job(job_id, status="running", progress=5, step="Loading scenario")
        logger.info(f"Starting optimization job {job_id} for scenario {scenario_id}")

        # Fetch scenario with profile
        scenario_data = fetch_scenario_with_profile(scenario_id)

        update_job(job_id, progress=10, step="Reconstructing model")

        # Reconstruct model and goals
        model, goals, withdrawals = reconstruct_from_scenario(scenario_data)

        if not goals:
            raise ValueError("No goals defined for optimization")

        # Get optimization parameters
        n_sims = scenario_data.get("n_sims", 500)
        seed = scenario_data.get("seed")
        t_max = scenario_data.get("t_max", 240)
        t_min = scenario_data.get("t_min", 12)
        solver = scenario_data.get("solver", "ECOS")
        objective = scenario_data.get("objective", "balanced")
        start_date_str = scenario_data.get("start_date")
        start_date = date.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str

        update_job(job_id, progress=15, step="Initializing optimizer")

        # Import CVaROptimizer (lazy loaded to avoid cvxpy import at module level)
        from finopt import CVaROptimizer

        # Create optimizer
        optimizer = CVaROptimizer(
            n_accounts=len(model.accounts),
            objective=objective,
            account_names=[acc.name for acc in model.accounts],
        )

        update_job(job_id, progress=20, step="Starting goal-seeking optimization")

        # Run optimization with goal-seeking
        # This uses bilevel optimization: binary search over T, convex solve for X
        opt_result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=t_max,
            n_sims=n_sims,
            seed=seed,
            start=start_date,
            verbose=False,  # Don't print to console
            search_method="binary",
            withdrawals=withdrawals,
            withdrawal_epsilon=0.05,
            solver=solver,
        )

        update_job(job_id, progress=85, step="Processing results")

        # Prepare allocation policy for storage
        # X has shape (T, M), convert to list of lists
        allocation_policy = opt_result.X.tolist()

        # Compute goal status
        goal_status = compute_goal_status_from_result(
            opt_result,
            model,
            start_date,
        )

        # Prepare diagnostics
        diagnostics = {
            "solver": solver,
            "objective": objective,
            "n_iterations": opt_result.n_iterations,
            "search_method": "binary",
        }
        if opt_result.diagnostics:
            diagnostics.update(opt_result.diagnostics)

        update_job(job_id, progress=95, step="Saving results")

        # Save results to Supabase
        save_optimization_result(
            job_id=job_id,
            allocation_policy=allocation_policy,
            optimal_horizon=opt_result.T,
            objective_value=float(opt_result.objective_value) if opt_result.objective_value else 0.0,
            feasible=opt_result.feasible,
            solve_time=float(opt_result.solve_time) if opt_result.solve_time else 0.0,
            diagnostics=diagnostics,
            goal_status=goal_status,
        )

        # Mark job complete
        update_job(job_id, status="completed", progress=100, step="Done")
        logger.info(
            f"Optimization job {job_id} completed: "
            f"T*={opt_result.T}, feasible={opt_result.feasible}"
        )

    except Exception as e:
        logger.exception(f"Optimization job {job_id} failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))
        raise
