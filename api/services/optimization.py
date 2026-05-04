"""
Optimization Service

Handles optimization jobs: fetching scenario data, running CVaR optimization
with goal-seeking, computing results, and saving to Supabase.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finopt.model import FinancialModel, SimulationResult
    from finopt.optimization import OptimizationResult

import numpy as np

from api.services._goal_metrics import compute_goal_probability
from api.services.reconstruction import reconstruct_from_scenario
from api.supabase_client import (
    fetch_scenario_with_profile,
    save_optimization_result,
    update_job,
)

logger = logging.getLogger(__name__)


def compute_goal_status_from_result(
    opt_result: OptimizationResult,
    model: FinancialModel,
    sim_result: SimulationResult,
    start_date: date,
) -> list[dict[str, Any]]:
    """
    Compute goal satisfaction status from optimization and simulation results.

    Uses the re-simulated wealth trajectories to compute actual achievement
    probabilities for each goal (empirical probability from Monte Carlo).

    Parameters
    ----------
    opt_result : OptimizationResult
        Result from CVaROptimizer.
    model : FinancialModel
        The financial model used.
    sim_result : SimulationResult
        Re-simulation result with wealth trajectories.
    start_date : date
        Simulation start date.

    Returns
    -------
    list[dict]
        Goal status for each goal, including actual probabilities.
    """
    from finopt import IntermediateGoal

    status_list = []

    # Map account names to indices
    account_name_to_idx = {acc.name: idx for idx, acc in enumerate(model.accounts)}

    for goal in opt_result.goals:
        # Get account name and index
        if isinstance(goal.account, str):
            account_name = goal.account
            account_idx = account_name_to_idx.get(goal.account, 0)
        else:
            account_name = model.accounts[goal.account].name
            account_idx = goal.account

        if isinstance(goal, IntermediateGoal):
            goal_type = "intermediate"
            goal_desc = f"{account_name} by {goal.date.isoformat()}"
            resolved_month = goal.resolve_month(start_date)
            if resolved_month < sim_result.wealth.shape[1]:
                actual_prob, dual = compute_goal_probability(
                    sim_result.wealth[:, resolved_month, account_idx],
                    goal.threshold,
                    goal.confidence,
                )
            else:
                actual_prob = None
                dual = {"empirical_probability": None, "confidence_gap": None, "note": None}
        else:
            goal_type = "terminal"
            goal_desc = f"{account_name} at horizon T={opt_result.T}"
            actual_prob, dual = compute_goal_probability(
                sim_result.wealth[:, -1, account_idx], goal.threshold, goal.confidence
            )

        status_list.append({
            "goal": goal_desc,
            "type": goal_type,
            "account": account_name,
            "threshold": goal.threshold,
            "required_confidence": goal.confidence,
            "satisfied": actual_prob >= goal.confidence if actual_prob is not None else opt_result.feasible,
            "actual_probability": actual_prob,
            # CVaR transparency metrics
            "empirical_probability": dual["empirical_probability"],
            "confidence_gap": dual["confidence_gap"],
            "note": dual["note"],
        })

    return status_list


def compute_cash_flow_stats(
    sim_result: SimulationResult,
    model: FinancialModel,
) -> dict[str, Any]:
    """
    Compute cash flow statistics (contributions and withdrawals) for visualization.

    Parameters
    ----------
    sim_result : SimulationResult
        Simulation result containing contributions and withdrawal arrays.
    model : FinancialModel
        The financial model (for account names).

    Returns
    -------
    dict
        Cash flow statistics with keys:
        - contributions_mean: list of T mean contribution values
        - contributions_by_account: list of dicts per account with mean allocations
        - withdrawals_mean: list of T mean total withdrawal values (or None)
        - withdrawals_by_account: list of dicts per account (or None)
    """
    T = sim_result.T
    M = sim_result.M
    X = sim_result.allocation  # (T, M)

    # Contributions: sim_result.contributions is (n_sims, T) or (T,)
    contributions = sim_result.contributions
    if contributions.ndim == 1:
        # Deterministic: same for all sims
        mean_contributions = contributions  # (T,)
    else:
        mean_contributions = np.mean(contributions, axis=0)  # (T,)

    # Per-account contributions = total_contribution * allocation_fraction
    contributions_by_account = []
    for m, acc in enumerate(model.accounts):
        acc_contributions = (mean_contributions * X[:, m]).tolist()
        contributions_by_account.append({
            "account": acc.name,
            "display_name": acc.display_name or acc.name,
            "mean": acc_contributions,
        })

    result: dict[str, Any] = {
        "contributions_mean": mean_contributions.tolist(),
        "contributions_by_account": contributions_by_account,
    }

    # Withdrawals
    if sim_result.withdrawals is not None:
        D = sim_result.withdrawals
        if D.ndim == 2:
            # Deterministic: (T, M)
            mean_D = D
        else:
            # Stochastic: (n_sims, T, M)
            mean_D = np.mean(D, axis=0)  # (T, M)

        # Total withdrawals per month
        result["withdrawals_mean"] = np.sum(mean_D, axis=1).tolist()  # (T,)

        # Per-account withdrawals
        withdrawals_by_account = []
        for m, acc in enumerate(model.accounts):
            acc_D = mean_D[:T, m] if mean_D.shape[0] >= T else mean_D[:, m]
            withdrawals_by_account.append({
                "account": acc.name,
                "display_name": acc.display_name or acc.name,
                "mean": acc_D.tolist(),
            })
        result["withdrawals_by_account"] = withdrawals_by_account

    return result


def compute_wealth_percentiles(
    sim_result: SimulationResult,
    model: FinancialModel,
) -> dict[str, Any]:
    """
    Compute wealth trajectory percentiles for visualization.

    Computes P10, P25, P50, P75, P90 percentiles for total wealth
    and per-account wealth across all Monte Carlo scenarios.

    Parameters
    ----------
    sim_result : SimulationResult
        Simulation result containing wealth trajectories.
    model : FinancialModel
        The financial model (for account names).

    Returns
    -------
    dict
        Summary statistics with keys:
        - total_wealth: dict with percentile arrays (T+1 values each)
        - per_account: list of dicts, one per account, with percentile arrays
    """
    wealth = sim_result.wealth          # (n_sims, T+1, M)
    total_wealth = sim_result.total_wealth  # (n_sims, T+1)

    percentile_levels = [10, 25, 50, 75, 90]

    # Total wealth percentiles
    total_stats = {
        "mean": np.mean(total_wealth, axis=0).tolist(),
    }
    for p in percentile_levels:
        total_stats[f"p{p}"] = np.percentile(total_wealth, p, axis=0).tolist()

    # Per-account percentiles
    per_account = []
    for m, acc in enumerate(model.accounts):
        account_wealth = wealth[:, :, m]  # (n_sims, T+1)
        acc_stats = {
            "account": acc.name,
            "display_name": acc.display_name or acc.name,
            "mean": np.mean(account_wealth, axis=0).tolist(),
        }
        for p in percentile_levels:
            acc_stats[f"p{p}"] = np.percentile(account_wealth, p, axis=0).tolist()
        per_account.append(acc_stats)

    # Sample individual trajectories for Monte Carlo visualization
    n_sims = total_wealth.shape[0]
    max_trajectories = 50
    if n_sims <= max_trajectories:
        sample_idx = np.arange(n_sims)
    else:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_sims, size=max_trajectories, replace=False)
        sample_idx.sort()

    sampled_total = total_wealth[sample_idx, :].tolist()  # list of lists

    sampled_per_account = []
    for m, acc in enumerate(model.accounts):
        sampled_per_account.append({
            "account": acc.name,
            "display_name": acc.display_name or acc.name,
            "trajectories": wealth[sample_idx, :, m].tolist(),
        })

    return {
        "total_wealth": total_stats,
        "per_account": per_account,
        "trajectories": {
            "total": sampled_total,
            "per_account": sampled_per_account,
            "n_sampled": len(sample_idx),
            "n_total": int(n_sims),
        },
    }


_VALID_SOLVERS = {"ECOS", "SCS", "CLARABEL", "MOSEK"}
_VALID_OBJECTIVES = {"risky", "balanced", "risky_turnover", "conservative"}


def _parse_opt_params(scenario_data: dict) -> tuple:
    """Extract and validate optimization parameters from scenario data."""
    raw_n_sims = scenario_data.get("n_sims", 500)
    raw_t_max = scenario_data.get("t_max", 240)
    raw_t_min = scenario_data.get("t_min", 12)
    solver = scenario_data.get("solver", "ECOS")
    objective = scenario_data.get("objective", "balanced")
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
        t_min = int(raw_t_min)
    except (TypeError, ValueError):
        raise ValueError(f"t_min must be an integer, got {raw_t_min!r}")
    if t_min <= 0 or t_min >= t_max:
        raise ValueError(f"t_min must be in (0, t_max={t_max}), got {t_min}")

    if not isinstance(solver, str) or solver.upper() not in _VALID_SOLVERS:
        raise ValueError(f"solver must be one of {sorted(_VALID_SOLVERS)}, got {solver!r}")

    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_VALID_OBJECTIVES)}, got {objective!r}")

    try:
        start_date = date.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
    except ValueError:
        raise ValueError(f"start_date is not a valid ISO date: {start_date_str!r}")

    return n_sims, t_max, t_min, solver, objective, seed, start_date


async def run_optimization(scenario_id: str, job_id: str) -> None:
    """
    Run CVaR optimization with goal-seeking for a scenario.

    This is the main entry point called from FastAPI background tasks.
    It fetches the scenario, runs bilevel optimization (outer: horizon search,
    inner: convex allocation), re-simulates with optimal policy to get
    wealth trajectories, and saves results.

    Parameters
    ----------
    scenario_id : str
        UUID of the scenario to optimize.
    job_id : str
        UUID of the job for progress tracking.
    """
    current_step = "initializing"
    try:
        update_job(job_id, status="running", progress=5, step="Loading scenario")
        logger.info(f"Starting optimization job {job_id} for scenario {scenario_id}")

        current_step = "loading scenario"
        scenario_data = fetch_scenario_with_profile(scenario_id)

        current_step = "reconstructing model"
        update_job(job_id, progress=10, step="Reconstructing model")
        model, goals, withdrawals = reconstruct_from_scenario(scenario_data)

        if not goals:
            raise ValueError("No goals defined for optimization")

        current_step = "parsing parameters"
        n_sims, t_max, t_min, solver, objective, seed, start_date = _parse_opt_params(scenario_data)

        current_step = "initializing optimizer"
        update_job(job_id, progress=15, step="Initializing optimizer")
        from finopt import CVaROptimizer
        optimizer = CVaROptimizer(
            n_accounts=len(model.accounts),
            objective=objective,
            account_names=[acc.name for acc in model.accounts],
        )

        current_step = "running optimization"
        update_job(job_id, progress=20, step="Starting goal-seeking optimization")

        def _optimization_progress(info: Any) -> None:
            try:
                if info.phase != "solving":
                    return
                fraction = min(info.iteration / max(info.total_estimated, 1), 1.0)
                progress = 20 + int(fraction * 50)
                progress = min(progress, 69)
                update_job(
                    job_id,
                    progress=progress,
                    step=f"Testing horizon T={info.current_T} months [{info.iteration}/{info.total_estimated}]",
                )
            except Exception as exc:
                logger.warning(f"Progress update failed (non-fatal): {exc}")

        opt_result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=t_max,
            n_sims=n_sims,
            seed=seed,
            start=start_date,
            verbose=False,
            search_method="binary",
            withdrawals=withdrawals,
            withdrawal_epsilon=0.05,
            solver=solver,
            progress_callback=_optimization_progress,
        )

        current_step = "simulating wealth trajectories"
        update_job(job_id, progress=70, step="Simulating wealth trajectories")
        sim_result = model.simulate_from_optimization(
            opt_result,
            n_sims=n_sims,
            seed=seed,
            start=start_date,
            withdrawals=withdrawals,
        )

        current_step = "processing results"
        update_job(job_id, progress=85, step="Processing results")
        allocation_policy = opt_result.X.tolist()

        goal_status = compute_goal_status_from_result(
            opt_result, model, sim_result, start_date,
        )
        summary_stats = compute_wealth_percentiles(sim_result, model)
        summary_stats["cash_flow"] = compute_cash_flow_stats(sim_result, model)

        diagnostics = {
            "solver": solver,
            "objective": objective,
            "n_iterations": opt_result.n_iterations,
            "search_method": "binary",
        }
        if opt_result.diagnostics:
            diagnostics.update(opt_result.diagnostics)

        current_step = "saving results"
        update_job(job_id, progress=95, step="Saving results")
        save_optimization_result(
            job_id=job_id,
            allocation_policy=allocation_policy,
            optimal_horizon=opt_result.T,
            objective_value=float(opt_result.objective_value) if opt_result.objective_value else 0.0,
            feasible=opt_result.feasible,
            solve_time=float(opt_result.solve_time) if opt_result.solve_time else 0.0,
            diagnostics=diagnostics,
            goal_status=goal_status,
            summary_stats=summary_stats,
        )

        update_job(job_id, status="completed", progress=100, step="Done")
        logger.info(
            f"Optimization job {job_id} completed: "
            f"T*={opt_result.T}, feasible={opt_result.feasible}"
        )

    except Exception as e:
        logger.exception(f"Optimization job {job_id} failed at '{current_step}': {e}")
        update_job(job_id, status="failed", error_message=f"[{current_step}] {e}")
        raise
