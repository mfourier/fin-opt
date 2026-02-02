"""
Simulation Service

Handles simulation jobs: fetching scenario data, running Monte Carlo
simulation, computing summary statistics, and saving results to Supabase.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

from api.supabase_client import (
    fetch_scenario_with_profile,
    save_simulation_result,
    update_job,
)
from api.services.reconstruction import reconstruct_from_scenario

logger = logging.getLogger(__name__)


def compute_summary_stats(
    wealth: np.ndarray,
    account_names: list[str],
) -> dict[str, Any]:
    """
    Compute summary statistics from wealth trajectories.

    Parameters
    ----------
    wealth : np.ndarray
        Wealth array of shape (n_sims, T+1, M).
    account_names : list[str]
        Names of accounts.

    Returns
    -------
    dict
        Summary statistics including mean, median, std, percentiles
        for total wealth and per-account wealth.
    """
    n_sims, T_plus_1, M = wealth.shape
    T = T_plus_1 - 1

    # Total wealth at terminal time
    total_terminal = wealth[:, -1, :].sum(axis=1)  # (n_sims,)

    # Per-account terminal wealth
    per_account_terminal = {
        name: wealth[:, -1, m].tolist()
        for m, name in enumerate(account_names)
    }

    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    stats = {
        "n_sims": n_sims,
        "T": T,
        "total_wealth": {
            "mean": float(np.mean(total_terminal)),
            "median": float(np.median(total_terminal)),
            "std": float(np.std(total_terminal)),
            "min": float(np.min(total_terminal)),
            "max": float(np.max(total_terminal)),
            "p10": float(np.percentile(total_terminal, 10)),
            "p90": float(np.percentile(total_terminal, 90)),
            "percentiles": {
                f"p{p}": float(np.percentile(total_terminal, p))
                for p in percentiles
            },
        },
        # Final total wealth (duplicate for compatibility)
        "final_total_wealth": {
            "mean": float(np.mean(total_terminal)),
            "median": float(np.median(total_terminal)),
            "std": float(np.std(total_terminal)),
        },
        # Per-account stats as list (for compatibility with tests)
        "accounts": [
            {
                "name": name,
                "mean": float(np.mean(wealth[:, -1, m])),
                "median": float(np.median(wealth[:, -1, m])),
                "std": float(np.std(wealth[:, -1, m])),
            }
            for m, name in enumerate(account_names)
        ],
        "per_account": {
            name: {
                "mean": float(np.mean(wealth[:, -1, m])),
                "median": float(np.median(wealth[:, -1, m])),
                "std": float(np.std(wealth[:, -1, m])),
            }
            for m, name in enumerate(account_names)
        },
        # Time series of median total wealth (for plotting)
        "median_trajectory": [
            float(np.median(wealth[:, t, :].sum(axis=1)))
            for t in range(T_plus_1)
        ],
        # Percentile bands for plotting
        "trajectory_bands": {
            "p10": [
                float(np.percentile(wealth[:, t, :].sum(axis=1), 10))
                for t in range(T_plus_1)
            ],
            "p90": [
                float(np.percentile(wealth[:, t, :].sum(axis=1), 90))
                for t in range(T_plus_1)
            ],
        },
    }

    return stats


def compute_goal_status(
    wealth: np.ndarray,
    goals: list,
    accounts: list,
    start_date: date,
) -> list[dict[str, Any]]:
    """
    Compute goal satisfaction status from simulation results.

    Parameters
    ----------
    wealth : np.ndarray
        Wealth array of shape (n_sims, T+1, M).
    goals : list
        List of IntermediateGoal and TerminalGoal objects.
    accounts : list[Account]
        List of account objects.
    start_date : date
        Simulation start date.

    Returns
    -------
    list[dict]
        Goal status for each goal with satisfaction probability.
    """
    from finopt import IntermediateGoal, TerminalGoal

    n_sims, T_plus_1, M = wealth.shape
    T = T_plus_1 - 1

    status_list = []

    for goal in goals:
        # Resolve account index
        if isinstance(goal.account, str):
            account_idx = next(
                (i for i, acc in enumerate(accounts) if acc.name == goal.account),
                None
            )
        else:
            account_idx = goal.account

        if account_idx is None:
            status_list.append({
                "goal": str(goal),
                "error": f"Account not found: {goal.account}",
            })
            continue

        # Determine time index and wealth to check
        if isinstance(goal, IntermediateGoal):
            # Calculate month offset from start_date
            month_diff = (goal.date.year - start_date.year) * 12 + (goal.date.month - start_date.month)
            t_idx = min(month_diff, T)
            goal_type = "intermediate"
            goal_desc = f"{goal.account} by {goal.date.isoformat()}"
        else:  # TerminalGoal
            t_idx = T
            goal_type = "terminal"
            goal_desc = f"{goal.account} at horizon"

        # Get wealth at goal time for target account
        wealth_at_t = wealth[:, t_idx, account_idx]

        # Calculate satisfaction probability
        satisfied_count = np.sum(wealth_at_t >= goal.threshold)
        actual_prob = satisfied_count / n_sims

        status_list.append({
            "goal": goal_desc,
            "type": goal_type,
            "account": goal.account if isinstance(goal.account, str) else accounts[goal.account].name,
            "threshold": goal.threshold,
            "required_confidence": goal.confidence,
            "probability": float(actual_prob),
            "actual_probability": float(actual_prob),  # Keep for compatibility
            "satisfied": bool(actual_prob >= goal.confidence),  # Convert numpy bool to Python bool
            "t_idx": t_idx,
        })

    return status_list


async def run_simulation(scenario_id: str, job_id: str) -> None:
    """
    Run a Monte Carlo simulation for a scenario.

    This is the main entry point called from FastAPI background tasks.
    It fetches the scenario, runs simulation, computes stats, and saves results.

    Parameters
    ----------
    scenario_id : str
        UUID of the scenario to simulate.
    job_id : str
        UUID of the job for progress tracking.
    """
    try:
        # Update job status
        update_job(job_id, status="running", progress=5, step="Loading scenario")
        logger.info(f"Starting simulation job {job_id} for scenario {scenario_id}")

        # Fetch scenario with profile
        scenario_data = fetch_scenario_with_profile(scenario_id)

        update_job(job_id, progress=10, step="Reconstructing model")

        # Reconstruct model and goals
        model, goals, withdrawals = reconstruct_from_scenario(scenario_data)

        # Get simulation parameters
        n_sims = scenario_data.get("n_sims", 500)
        seed = scenario_data.get("seed")
        t_max = scenario_data.get("t_max", 120)
        start_date_str = scenario_data.get("start_date")
        start_date = date.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str

        update_job(job_id, progress=20, step="Running Monte Carlo simulation")

        # Create default allocation (equal weights)
        M = len(model.accounts)
        X = np.ones((t_max, M)) / M

        # Run simulation
        result = model.simulate(
            T=t_max,
            X=X,
            n_sims=n_sims,
            start=start_date,
            seed=seed,
            withdrawals=withdrawals,
        )

        update_job(job_id, progress=80, step="Computing statistics")

        # Compute summary stats
        account_names = [acc.name for acc in model.accounts]
        summary_stats = compute_summary_stats(result.wealth, account_names)

        # Compute goal status
        goal_status = None
        if goals:
            goal_status = compute_goal_status(
                result.wealth,
                goals,
                model.accounts,
                start_date,
            )

        update_job(job_id, progress=95, step="Saving results")

        # Save results to Supabase
        save_simulation_result(
            job_id=job_id,
            summary_stats=summary_stats,
            goal_status=goal_status,
        )

        # Mark job complete
        update_job(job_id, status="completed", progress=100, step="Done")
        logger.info(f"Simulation job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Simulation job {job_id} failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))
        raise
