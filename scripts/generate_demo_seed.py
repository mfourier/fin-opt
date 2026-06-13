"""
Generate the shared demo seed migration.

Runs the *real* FinOpt pipeline (income -> returns -> portfolio -> CVaR
optimization with goal-seeking + withdrawals) for a canonical demo scenario,
then emits supabase/migrations/003_demo_shared_scenario.sql containing:

  - schema changes: ``is_demo`` flag on profiles/scenarios, nullable
    ``profiles.user_id`` (so the demo is owned by nobody)
  - RLS policies letting any user read the demo rows (read-only)
  - the demo profile + scenario + a completed job + the precomputed result

The result JSON mirrors exactly what ``api/services/optimization.run_optimization``
saves, so the existing ResultsPage renders it unchanged.

Usage
-----
    python scripts/generate_demo_seed.py            # validate only (prints summary)
    python scripts/generate_demo_seed.py --write     # also write the migration
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# Allow running as a plain script (`python scripts/generate_demo_seed.py`) by
# putting the repo root on the path so `api` / `finopt` import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the API's result-formatting helpers so the JSON shape matches prod 1:1.
from api.services.optimization import (
    compute_cash_flow_stats,
    compute_goal_status_from_result,
    compute_wealth_percentiles,
)
from api.services.reconstruction import reconstruct_from_scenario

# ---------------------------------------------------------------------------
# Fixed identifiers (stable so the migration is idempotent and the frontend
# can discover the demo by flag, not by hard-coded id).
# ---------------------------------------------------------------------------
PROFILE_ID = "00000000-0000-4000-8000-000000000001"
SCENARIO_ID = "00000000-0000-4000-8000-000000000002"
JOB_ID = "00000000-0000-4000-8000-000000000003"
RESULT_ID = "00000000-0000-4000-8000-000000000004"

# ---------------------------------------------------------------------------
# Demo content. Generic English names, USD-scale nominal figures, three
# accounts spanning the recalibrated risk ladder
# (money-market / fixed-income / equity).
# ---------------------------------------------------------------------------
PROFILE = {
    "name": "Demo — First home & nest egg",
    "description": "A worked example showing the full machinery: three accounts, "
    "scheduled and surprise withdrawals, dated milestones and a long-term goal.",
    "income_config": {
        "fixed": {"base": 2_000, "annual_growth": 0.03},
        "variable": {"base": 400, "sigma": 0.15, "annual_growth": 0.02, "floor": 0},
        "contribution_rate_fixed": 0.3,
        "contribution_rate_variable": 1.0,
    },
    "accounts_config": [
        {
            "name": "safe_savings",
            "display_name": "Safe savings",
            "annual_return": 0.045,
            "annual_volatility": 0.01,
            "initial_wealth": 2_500,
        },
        {
            "name": "house_savings",
            "display_name": "House savings",
            "annual_return": 0.065,
            "annual_volatility": 0.05,
            "initial_wealth": 3_500,
        },
        {
            "name": "global_etf",
            "display_name": "Global ETF",
            "annual_return": 0.14,
            "annual_volatility": 0.17,
            "initial_wealth": 2_000,
        },
    ],
    "correlation_matrix": [
        [1.0, 0.10, 0.0],
        [0.10, 1.0, 0.30],
        [0.0, 0.30, 1.0],
    ],
}

SCENARIO = {
    "name": "Demo plan",
    "description": "Build an emergency fund, save a house down payment, and grow "
    "a long-term nest egg — all at once.",
    "start_date": "2026-07-01",
    "intermediate_goals": [
        {
            "account": "safe_savings",
            "threshold": 4_000,
            "confidence": 0.90,
            "date": "2027-07-01",
        },
        {
            "account": "house_savings",
            "threshold": 10_000,
            "confidence": 0.85,
            "date": "2028-07-01",
        },
    ],
    "terminal_goals": [
        {"account": "global_etf", "threshold": 20_000, "confidence": 0.80},
    ],
    "withdrawals": {
        "scheduled": [
            {
                "account": "safe_savings",
                "amount": 2_000,
                "date": "2027-01-01",
                "description": "Planned trip",
            }
        ],
        "stochastic": [
            {
                "account": "safe_savings",
                "base_amount": 500,
                "sigma": 0.2,
                "date": "2027-10-01",
                "floor": 0,
                "description": "Unexpected expense",
            }
        ],
    },
    # Optimization parameters (mirror the frontend hidden defaults).
    "n_sims": 500,
    "seed": 42,
    "t_max": 240,
    "t_min": 12,
    "solver": "ECOS",
    "objective": "proportional",
}


def run_pipeline() -> dict:
    """Run the real optimization + simulation and return the result dict."""
    scenario_data = {**SCENARIO, "profiles": PROFILE}
    model, goals, withdrawals = reconstruct_from_scenario(scenario_data)
    start = date.fromisoformat(SCENARIO["start_date"])

    from finopt import CVaROptimizer

    optimizer = CVaROptimizer(
        n_accounts=len(model.accounts),
        objective=SCENARIO["objective"],
        account_names=[acc.name for acc in model.accounts],
    )

    opt_result = model.optimize(
        goals=goals,
        optimizer=optimizer,
        T_max=SCENARIO["t_max"],
        n_sims=SCENARIO["n_sims"],
        seed=SCENARIO["seed"],
        start=start,
        verbose=False,
        search_method="bracketed",
        withdrawals=withdrawals,
        withdrawal_epsilon=0.05,
        solver=SCENARIO["solver"],
    )

    sim_result = model.simulate_from_optimization(
        opt_result,
        n_sims=SCENARIO["n_sims"],
        seed=SCENARIO["seed"],
        start=start,
        withdrawals=withdrawals,
    )

    goal_status = compute_goal_status_from_result(opt_result, model, sim_result, start)
    summary_stats = compute_wealth_percentiles(sim_result, model)
    summary_stats["cash_flow"] = compute_cash_flow_stats(sim_result, model)

    diagnostics = {
        "solver": SCENARIO["solver"],
        "objective": SCENARIO["objective"],
        "n_iterations": opt_result.n_iterations,
    }
    if opt_result.diagnostics:
        diagnostics.update(opt_result.diagnostics)

    return {
        "allocation_policy": opt_result.X.tolist(),
        "optimal_horizon": opt_result.T,
        "objective_value": float(opt_result.objective_value)
        if opt_result.objective_value
        else 0.0,
        "feasible": bool(opt_result.feasible),
        "solve_time": float(opt_result.solve_time) if opt_result.solve_time else 0.0,
        "diagnostics": diagnostics,
        "goal_status": goal_status,
        "summary_stats": summary_stats,
    }


def _jsonb(obj) -> str:
    """Render a Python object as a single-quoted SQL JSONB literal."""
    return "'" + json.dumps(obj).replace("'", "''") + "'::jsonb"


def _text(value: str) -> str:
    """Render a plain SQL string literal (single-quoted, escaped)."""
    return "'" + value.replace("'", "''") + "'"


def build_sql(result: dict) -> str:
    p, s = PROFILE, SCENARIO
    return f"""\
-- ============================================================================
-- FinOpt Migration 003: Shared read-only DEMO scenario
-- Generated by scripts/generate_demo_seed.py — DO NOT edit by hand.
-- ============================================================================
-- Adds an ``is_demo`` flag and RLS policies so a single canonical demo
-- (profile + scenario + completed job + precomputed result) is readable by
-- every user while remaining immutable (owned by no auth user).
-- Idempotent: safe to run multiple times.
-- ============================================================================

-- ---- Schema: demo flag + ownerless demo rows -------------------------------
ALTER TABLE profiles  ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE profiles  ALTER COLUMN user_id DROP NOT NULL;

-- ---- RLS: anyone may READ demo rows (writes still blocked by ownership) -----
DROP POLICY IF EXISTS "Anyone can view demo profiles" ON profiles;
CREATE POLICY "Anyone can view demo profiles" ON profiles
    FOR SELECT USING (is_demo = true);

DROP POLICY IF EXISTS "Anyone can view demo scenarios" ON scenarios;
CREATE POLICY "Anyone can view demo scenarios" ON scenarios
    FOR SELECT USING (is_demo = true);

DROP POLICY IF EXISTS "Anyone can view demo jobs" ON jobs;
CREATE POLICY "Anyone can view demo jobs" ON jobs
    FOR SELECT USING (
        scenario_id IN (SELECT id FROM scenarios WHERE is_demo = true)
    );

DROP POLICY IF EXISTS "Anyone can view demo results" ON results;
CREATE POLICY "Anyone can view demo results" ON results
    FOR SELECT USING (
        job_id IN (
            SELECT j.id FROM jobs j
            JOIN scenarios s ON j.scenario_id = s.id
            WHERE s.is_demo = true
        )
    );

-- ---- Seed: rebuild the demo from scratch (cascade clears children) ---------
DELETE FROM profiles WHERE id = '{PROFILE_ID}';

INSERT INTO profiles (id, user_id, name, description, income_config, accounts_config, correlation_matrix, is_demo)
VALUES (
    '{PROFILE_ID}', NULL,
    {_text(p["name"])},
    {_text(p["description"])},
    {_jsonb(p["income_config"])},
    {_jsonb(p["accounts_config"])},
    {_jsonb(p["correlation_matrix"])},
    true
);

INSERT INTO scenarios (id, profile_id, name, description, intermediate_goals, terminal_goals, withdrawals,
                       start_date, n_sims, seed, t_max, t_min, solver, objective, is_demo)
VALUES (
    '{SCENARIO_ID}', '{PROFILE_ID}',
    {_text(s["name"])},
    {_text(s["description"])},
    {_jsonb(s["intermediate_goals"])},
    {_jsonb(s["terminal_goals"])},
    {_jsonb(s["withdrawals"])},
    '{s["start_date"]}', {s["n_sims"]}, {s["seed"]}, {s["t_max"]}, {s["t_min"]},
    '{s["solver"]}', '{s["objective"]}', true
);

INSERT INTO jobs (id, scenario_id, job_type, status, progress, current_step, started_at, completed_at, created_at)
VALUES (
    '{JOB_ID}', '{SCENARIO_ID}', 'optimization', 'completed', 100, 'Done', NOW(), NOW(), NOW()
);

INSERT INTO results (id, job_id, result_type, allocation_policy, optimal_horizon, objective_value,
                     feasible, solve_time, diagnostics, summary_stats, goal_status)
VALUES (
    '{RESULT_ID}', '{JOB_ID}', 'optimization',
    {_jsonb(result["allocation_policy"])},
    {result["optimal_horizon"]},
    {result["objective_value"]},
    {str(result["feasible"]).lower()},
    {result["solve_time"]},
    {_jsonb(result["diagnostics"])},
    {_jsonb(result["summary_stats"])},
    {_jsonb(result["goal_status"])}
);
"""


def main() -> None:
    write = "--write" in sys.argv
    result = run_pipeline()

    # Summary to stderr for human inspection.
    print("=== DEMO PIPELINE SUMMARY ===", file=sys.stderr)
    print(f"feasible      : {result['feasible']}", file=sys.stderr)
    print(f"T* (months)   : {result['optimal_horizon']}", file=sys.stderr)
    print(f"solve_time (s): {result['solve_time']:.2f}", file=sys.stderr)
    print("goals:", file=sys.stderr)
    for g in result["goal_status"]:
        print(
            f"  - {g['goal']:<40} satisfied={g['satisfied']!s:<5} "
            f"p={g['actual_probability']} (req {g['required_confidence']})",
            file=sys.stderr,
        )

    if write:
        out = Path("supabase/migrations/003_demo_shared_scenario.sql")
        out.write_text(build_sql(result))
        print(f"\nWrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)
    else:
        print("\n(dry run — pass --write to emit the migration)", file=sys.stderr)


if __name__ == "__main__":
    main()
