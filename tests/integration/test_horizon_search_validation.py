"""
CI guard for the bracketed horizon search (slow).

Enforces, on a reduced scenario grid × small seed sweep, the hard correctness
invariants validated in depth by ``benchmarks/bench_horizon_search.py``:

  - ``bracketed.T == oracle.T`` for every seed (linear brute-force = ground truth).
  - certificate ``false_prunes == 0`` (never skips a feasible horizon).
  - ``monotonicity_violations == 0`` (FEAS(T) is 0…0 1…1 around T*).
  - infeasible problems raise ``InfeasibleError`` via upward galloping.

Marked ``slow`` (runs many real convex solves); run with ``-m slow``.
The reusable harness lives in the benchmark module to avoid duplication.
"""

import sys
from pathlib import Path

import pytest

# Make the benchmark harness importable
BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bench_horizon_search as bench  # noqa: E402

from finopt.exceptions import InfeasibleError  # noqa: E402

pytestmark = pytest.mark.slow

N_SIMS = 200
SEEDS = [1, 2, 3]

# Scenarios with small/moderate T* keep the suite tractable while covering the
# key regimes: easy, high-confidence, withdrawals, non-zero W0, multi-goal coupling,
# already-satisfied edge, and intermediate+terminal.
_ALL = {s.name: s for s in bench.build_scenarios()}
CORRECTNESS_SCENARIOS = [
    "terminal_easy", "high_confidence", "with_withdrawals",
    "nonzero_initial_wealth", "multi_terminal_diff_accounts",
    "already_satisfied", "intermediate_plus_terminal",
]
# Audit solves EVERY horizon up to T*+buffer, so restrict to the smaller-T cases.
AUDIT_SCENARIOS = [
    "terminal_easy", "already_satisfied", "with_withdrawals",
    "nonzero_initial_wealth", "multi_terminal_diff_accounts",
]


@pytest.mark.parametrize("scenario_name", CORRECTNESS_SCENARIOS)
@pytest.mark.parametrize("seed", SEEDS)
def test_bracketed_matches_oracle(scenario_name, seed):
    """Bracketed returns the exact minimum horizon found by the brute-force oracle."""
    sc = _ALL[scenario_name]
    oracle = bench.run_method(sc, "linear", N_SIMS, seed)
    bracketed = bench.run_method(sc, "bracketed", N_SIMS, seed)
    assert oracle.error is None and bracketed.error is None
    assert bracketed.T_star == oracle.T_star
    assert bracketed.feasible
    # Bracketed must never do more real solves than the brute-force oracle.
    assert bracketed.n_evals <= oracle.n_evals


@pytest.mark.parametrize("scenario_name", AUDIT_SCENARIOS)
@pytest.mark.parametrize("seed", [1, 2])
def test_certificate_and_monotonicity(scenario_name, seed):
    """Certificate never false-prunes a feasible horizon; FEAS(T) is monotone."""
    sc = _ALL[scenario_name]
    audit = bench.audit_scenario(sc, N_SIMS, seed)
    assert audit.false_prunes == 0, (
        f"{scenario_name} seed={seed}: certificate pruned a feasible horizon"
    )
    assert audit.monotonicity_violations == 0, (
        f"{scenario_name} seed={seed}: FEAS(T) is non-monotone"
    )


def test_infeasible_within_tmax_raises():
    """An unreachable goal within T_max raises InfeasibleError (upward galloping)."""
    sc = bench.infeasible_scenario()
    result = bench.run_method(sc, "bracketed", N_SIMS, seed=1)
    # run_method catches InfeasibleError and records it as an error string.
    assert result.error is not None and "feasible" in result.error.lower()


def test_infeasible_raises_directly():
    """Direct seek() call raises InfeasibleError (not swallowed)."""

    from finopt.optimization import CVaROptimizer, GoalSeeker

    sc = bench.infeasible_scenario()
    A_gen, R_gen, D_gen = bench.make_generators(sc)
    seeker = GoalSeeker(
        CVaROptimizer(n_accounts=len(sc.accounts), objective=bench.OBJECTIVE),
        T_max=sc.T_max, verbose=False,
    )
    with pytest.raises(InfeasibleError):
        seeker.seek(
            goals=sc.goals, A_generator=A_gen, R_generator=R_gen,
            initial_wealth=sc.initial_wealth, accounts=sc.accounts,
            start_date=bench.START_DATE, n_sims=N_SIMS, seed=1,
            search_method="bracketed", D_generator=D_gen,
        )
