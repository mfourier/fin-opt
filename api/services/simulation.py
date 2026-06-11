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

from api.services._goal_metrics import compute_goal_probability
from api.services.reconstruction import reconstruct_from_scenario
from api.supabase_client import (
    fetch_scenario_with_profile,
    save_simulation_result,
    update_job,
)

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
    from finopt import IntermediateGoal

    T = wealth.shape[1] - 1
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

        # Calculate satisfaction probability and CVaR dual metrics
        actual_prob, dual = compute_goal_probability(
            wealth[:, t_idx, account_idx], goal.threshold, goal.confidence
        )

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
            # CVaR transparency metrics
            "empirical_probability": dual["empirical_probability"],
            "confidence_gap": dual["confidence_gap"],
            "note": dual["note"],
        })

    return status_list


def _parse_sim_params(scenario_data: dict) -> tuple:
    """Extract and validate simulation parameters from scenario data."""
    raw_n_sims = scenario_data.get("n_sims", 500)
    raw_t_max = scenario_data.get("t_max", 120)
    seed = scenario_data.get("seed")
    start_date_str = scenario_data.get("start_date")

    try:
        n_sims = int(raw_n_sims)
    except (TypeError, ValueError):
        raise ValueError(f"n_sims must be an integer, got {raw_n_sims!r}")
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive, got {n_sims}")

    try:
        t_max = int(raw_t_max)
    except (TypeError, ValueError):
        raise ValueError(f"t_max must be an integer, got {raw_t_max!r}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    try:
        start_date = date.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
    except ValueError:
        raise ValueError(f"start_date is not a valid ISO date: {start_date_str!r}")

    return n_sims, t_max, seed, start_date


def run_simulation(scenario_id: str, job_id: str) -> None:
    """
    Run a Monte Carlo simulation for a scenario.

    This is the main entry point called from FastAPI background tasks.
    It fetches the scenario, runs simulation, computes stats, and saves results.

    NOTE: intentionally a SYNC ``def`` (CPU-bound body, no ``await``). Starlette
    threadpools sync background tasks, keeping the event loop free for ``/health``.
    Declaring it ``async`` blocks the loop and gets the instance health-checked to
    death mid-job. Do not make it async. See run_optimization for details.

    Parameters
    ----------
    scenario_id : str
        UUID of the scenario to simulate.
    job_id : str
        UUID of the job for progress tracking.
    """
    current_step = "initializing"
    try:
        update_job(job_id, status="running", progress=5, step="Loading scenario")
        logger.info(f"Starting simulation job {job_id} for scenario {scenario_id}")

        current_step = "loading scenario"
        scenario_data = fetch_scenario_with_profile(scenario_id)

        current_step = "reconstructing model"
        update_job(job_id, progress=10, step="Reconstructing model")
        model, goals, withdrawals = reconstruct_from_scenario(scenario_data)

        current_step = "parsing parameters"
        n_sims, t_max, seed, start_date = _parse_sim_params(scenario_data)

        current_step = "running Monte Carlo simulation"
        update_job(job_id, progress=20, step="Running Monte Carlo simulation")
        M = len(model.accounts)
        X = np.ones((t_max, M)) / M
        result = model.simulate(
            T=t_max,
            X=X,
            n_sims=n_sims,
            start=start_date,
            seed=seed,
            withdrawals=withdrawals,
        )

        current_step = "computing statistics"
        update_job(job_id, progress=80, step="Computing statistics")
        account_names = [acc.name for acc in model.accounts]
        summary_stats = compute_summary_stats(result.wealth, account_names)

        goal_status = None
        if goals:
            goal_status = compute_goal_status(
                result.wealth,
                goals,
                model.accounts,
                start_date,
            )

        current_step = "saving results"
        update_job(job_id, progress=95, step="Saving results")
        save_simulation_result(
            job_id=job_id,
            summary_stats=summary_stats,
            goal_status=goal_status,
        )

        update_job(job_id, status="completed", progress=100, step="Done")
        logger.info(f"Simulation job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Simulation job {job_id} failed at '{current_step}': {e}")
        update_job(job_id, status="failed", error_message=f"[{current_step}] {e}")
        raise
