"""
Benchmark & deep validation harness for horizon search in GoalSeeker.

Purpose
-------
Two jobs in one module:

1. **Efficiency benchmark** — how many convex solves each search method
   (`linear`, `binary`, `bracketed`) needs to locate the minimum feasible horizon
   T*, plus a T-weighted cost proxy (Σ T over solves, since solve size grows with
   the horizon) and wall-time.

2. **Deep correctness validation** — turns "it looked right on one seed" into
   statistical + hard-invariant guarantees:
     - Seed sweep: `bracketed.T == oracle.T` must hold for EVERY seed (the linear
       brute-force from the floor is the ground-truth oracle).
     - Certificate audit: for every horizon the necessary-feasibility certificate
       rejects, the real solve must also be infeasible -> **false-prunes == 0**.
     - Monotonicity diagnostic: FEAS(T) must be 0…0 1…1 over a window around T*
       (the assumption both binary and bracketed rely on) -> **violations == 0**.

The reusable functions (`build_scenarios`, `run_method`, `audit_scenario`) are
imported by `tests/integration/test_horizon_search_validation.py` so the same
invariants are enforced in CI.

Usage
-----
    python benchmarks/bench_horizon_search.py                     # headline table
    python benchmarks/bench_horizon_search.py --seeds 30          # + seed sweep
    python benchmarks/bench_horizon_search.py --seeds 30 --audit  # + cert/monotonicity
    python benchmarks/bench_horizon_search.py --nsims-sweep       # + n_sims sweep
    python benchmarks/bench_horizon_search.py --all               # everything
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finopt.goals import GoalSet, IntermediateGoal, TerminalGoal  # noqa: E402
from finopt.optimization import CVaROptimizer, GoalSeeker  # noqa: E402
from finopt.portfolio import Account  # noqa: E402
from finopt.income import FixedIncome, IncomeModel, VariableIncome  # noqa: E402
from finopt.model import FinancialModel  # noqa: E402
from finopt.exceptions import InfeasibleError  # noqa: E402

START_DATE = date(2025, 6, 1)
OBJECTIVE = "balanced"

GenTriple = Tuple[Callable, Callable, Optional[Callable]]


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single benchmark / validation case.

    backend "synthetic": flat contributions + i.i.d. lognormal returns derived
    from the account parameters (fast, deterministic, isolates the search).

    backend "model": real FinancialModel (seasonal VariableIncome + correlated
    ReturnModel) — exercises the all-in curve fidelity under seasonality.
    """
    name: str
    description: str
    accounts: List[Account]
    goals: List[Union[IntermediateGoal, TerminalGoal]]
    initial_wealth: np.ndarray
    T_max: int
    backend: str = "synthetic"
    monthly_contribution: float = 500_000.0
    withdrawal: Optional[Tuple[int, int, float]] = None  # (month, account_idx, amount)
    model: Optional[FinancialModel] = None
    # Scenarios where the true optimum is unreachable within T_max.
    expect_infeasible: bool = False


def _two_accounts() -> List[Account]:
    return [
        Account.from_annual("Conservative", 0.04, 0.05),
        Account.from_annual("Aggressive", 0.14, 0.15),
    ]


def _real_income_model(accounts: List[Account]) -> FinancialModel:
    fixed = FixedIncome(base=1_500_000, annual_growth=0.03)
    # Seasonality: bonus months heavier (12-element factors via VariableIncome)
    variable = VariableIncome(base=250_000, sigma=0.12, seed=7)
    income = IncomeModel(fixed=fixed, variable=variable)
    return FinancialModel(income=income, accounts=accounts)


def build_scenarios() -> List[Scenario]:
    """Representative grid spanning the heuristic's regimes and edge cases."""
    A = _two_accounts
    iw0 = np.array([0.0, 0.0])

    scenarios = [
        Scenario("terminal_easy",
                 "Single terminal goal, low threshold (T* small, near floor)",
                 A(), [TerminalGoal(account="Aggressive", threshold=2_000_000,
                                    confidence=0.70)], iw0, 120),
        Scenario("terminal_midrange",
                 "Single terminal goal, moderate threshold",
                 A(), [TerminalGoal(account="Aggressive", threshold=8_000_000,
                                    confidence=0.80)], iw0, 180),
        Scenario("terminal_hard",
                 "Single terminal goal, high threshold + high confidence (T* large)",
                 A(), [TerminalGoal(account="Aggressive", threshold=20_000_000,
                                    confidence=0.90)], iw0, 240),
        Scenario("high_confidence",
                 "Single terminal goal at eps=0.01 (hard left-tail quantile)",
                 A(), [TerminalGoal(account="Aggressive", threshold=6_000_000,
                                    confidence=0.99)], iw0, 240),
        Scenario("intermediate_plus_terminal",
                 "Emergency fund (intermediate) + retirement (terminal)",
                 A(), [
                     IntermediateGoal(account="Conservative", threshold=3_000_000,
                                      confidence=0.85, date=date(2026, 1, 1)),
                     TerminalGoal(account="Aggressive", threshold=10_000_000,
                                  confidence=0.80),
                 ], iw0, 240, monthly_contribution=600_000.0),
        Scenario("multi_terminal_diff_accounts",
                 "Two terminal goals in different accounts (coupling)",
                 A(), [
                     TerminalGoal(account="Conservative", threshold=4_000_000,
                                  confidence=0.80),
                     TerminalGoal(account="Aggressive", threshold=8_000_000,
                                  confidence=0.80),
                 ], iw0, 240, monthly_contribution=600_000.0),
        Scenario("nonzero_initial_wealth",
                 "Single terminal goal with non-zero starting wealth",
                 A(), [TerminalGoal(account="Aggressive", threshold=10_000_000,
                                    confidence=0.80)],
                 np.array([1_000_000.0, 2_000_000.0]), 180),
        Scenario("already_satisfied",
                 "Initial wealth already exceeds threshold (T* = floor)",
                 A(), [TerminalGoal(account="Aggressive", threshold=1_000_000,
                                    confidence=0.80)],
                 np.array([0.0, 3_000_000.0]), 60),
        Scenario("with_withdrawals",
                 "Terminal goal with a mid-horizon withdrawal from the account",
                 A(), [TerminalGoal(account="Aggressive", threshold=8_000_000,
                                    confidence=0.80)], iw0, 180,
                 monthly_contribution=600_000.0, withdrawal=(6, 1, 200_000.0)),
    ]

    # Real-income (seasonal) scenario
    real_accounts = _two_accounts()
    scenarios.append(Scenario(
        "real_income_seasonal",
        "Real IncomeModel (seasonal variable income) + correlated returns",
        real_accounts,
        [TerminalGoal(account="Aggressive", threshold=12_000_000, confidence=0.80)],
        np.array([0.0, 0.0]), 240, backend="model",
        model=_real_income_model(real_accounts),
    ))
    return scenarios


def infeasible_scenario() -> Scenario:
    """A scenario whose goal is unreachable within T_max (for error-path tests)."""
    return Scenario(
        "infeasible_within_tmax",
        "Unreachable goal within a short T_max",
        _two_accounts(),
        [TerminalGoal(account="Aggressive", threshold=500_000_000, confidence=0.95)],
        np.array([0.0, 0.0]), 24, expect_infeasible=True,
    )


# ---------------------------------------------------------------------------
# Generators (must replicate seek's seed offsets exactly)
# ---------------------------------------------------------------------------

def make_generators(scenario: Scenario) -> GenTriple:
    """Build (A_gen, R_gen, D_gen) for a scenario.

    Determinism note: seek() calls R_generator with seed+1 and A/D_generator with
    seed. These generators are pure functions of their seed argument, so the audit
    can reproduce the exact per-T samples by calling them with the same offsets.
    """
    if scenario.backend == "model":
        model = scenario.model
        assert model is not None

        def A_gen(T, n, s):
            return model.income.contributions(
                months=T, n_sims=n, seed=s, output="array", start=START_DATE)

        def R_gen(T, n, s):
            return model.returns.generate(T, n_sims=n, seed=s)

        return A_gen, R_gen, None

    # synthetic backend
    accounts = scenario.accounts
    M = len(accounts)
    mu = np.array([a.monthly_params["mu"] for a in accounts])
    sigma = np.array([a.monthly_params["sigma"] for a in accounts])

    def A_gen(T, n, s):
        return np.full((n, T), scenario.monthly_contribution)

    def R_gen(T, n, s):
        rng = np.random.default_rng(s)
        z = rng.standard_normal((n, T, M))
        return np.exp(mu[None, None, :] + sigma[None, None, :] * z) - 1.0

    D_gen = None
    if scenario.withdrawal is not None:
        w_month, w_acc, w_amt = scenario.withdrawal

        def D_gen(T, n, s):
            D = np.zeros((n, T, M))
            if T > w_month:
                D[:, w_month, w_acc] = w_amt
            return D

    return A_gen, R_gen, D_gen


def _make_seeker(scenario: Scenario) -> Tuple[GoalSeeker, CVaROptimizer]:
    optimizer = CVaROptimizer(n_accounts=len(scenario.accounts), objective=OBJECTIVE)
    seeker = GoalSeeker(optimizer, T_max=scenario.T_max, verbose=False)
    return seeker, optimizer


# ---------------------------------------------------------------------------
# Running a single method
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    scenario: str
    method: str
    feasible: bool
    T_star: Optional[int]
    n_evals: Optional[int]
    weighted_cost: Optional[int]  # Σ T over real solves (solve size proxy)
    horizons: Optional[List[int]]
    wall_time: float
    error: Optional[str] = None


def run_method(scenario: Scenario, method: str, n_sims: int,
               seed: Optional[int]) -> RunResult:
    seeker, _ = _make_seeker(scenario)
    A_gen, R_gen, D_gen = make_generators(scenario)

    t0 = time.time()
    try:
        result = seeker.seek(
            goals=scenario.goals, A_generator=A_gen, R_generator=R_gen,
            initial_wealth=scenario.initial_wealth, accounts=scenario.accounts,
            start_date=START_DATE, n_sims=n_sims, seed=seed,
            search_method=method, D_generator=D_gen,
        )
        wall = time.time() - t0
        diag = result.diagnostics or {}
        horizons = diag.get("horizons_evaluated") or []
        return RunResult(
            scenario=scenario.name, method=method, feasible=result.feasible,
            T_star=result.T, n_evals=diag.get("n_horizon_evals"),
            weighted_cost=int(sum(horizons)), horizons=horizons, wall_time=wall,
        )
    except InfeasibleError as e:
        return RunResult(
            scenario=scenario.name, method=method, feasible=False, T_star=None,
            n_evals=None, weighted_cost=None, horizons=None,
            wall_time=time.time() - t0, error=str(e)[:50],
        )
    except Exception as e:  # defensive: keep a sweep alive on unexpected failures
        return RunResult(
            scenario=scenario.name, method=method, feasible=False, T_star=None,
            n_evals=None, weighted_cost=None, horizons=None,
            wall_time=time.time() - t0, error=f"{type(e).__name__}: {str(e)[:40]}",
        )


# ---------------------------------------------------------------------------
# Deep validation: certificate audit + monotonicity
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    scenario: str
    seed: int
    floor: int
    T_star: Optional[int]
    n_cert_rejects: int          # horizons the certificate declared infeasible
    false_prunes: int            # cert rejected but real solve was feasible (MUST be 0)
    monotonicity_violations: int # feas[T] and not feas[T+1] (MUST be 0)
    feas_vector: List[Tuple[int, bool, bool]] = field(default_factory=list)


def audit_scenario(scenario: Scenario, n_sims: int, seed: int,
                   buffer: int = 3) -> AuditResult:
    """Solve every horizon in [floor, T*+buffer] and check hard invariants.

    For each T we compute, on the EXACT sample the search would use:
      - cert = GoalSeeker._necessary_feasible(...)  (no convex solve)
      - feas = optimizer.solve(...).feasible        (real solve)
    Invariants checked:
      - false_prunes      : feas == True but cert == False  -> certificate unsound
      - monotonicity      : feas drops back to False after being True
    """
    seeker, optimizer = _make_seeker(scenario)
    A_gen, R_gen, D_gen = make_generators(scenario)
    goal_set = GoalSet(scenario.goals, scenario.accounts, START_DATE)
    iw = scenario.initial_wealth
    floor = max(goal_set.T_min, 1)

    # First locate T* (true minimum) via the brute-force oracle to size the window.
    oracle = run_method(scenario, "linear", n_sims, seed)
    if oracle.T_star is None:
        return AuditResult(scenario.name, seed, floor, None, 0, 0, 0, [])
    hi = min(oracle.T_star + buffer, scenario.T_max)

    records: List[Tuple[int, bool, bool]] = []
    n_cert_rejects = 0
    false_prunes = 0
    for T in range(floor, hi + 1):
        A = A_gen(T, n_sims, seed)
        R = R_gen(T, n_sims, None if seed is None else seed + 1)
        D = D_gen(T, n_sims, seed) if D_gen is not None else None

        cert = seeker._necessary_feasible(goal_set, A, R, iw, D, T)
        result = optimizer.solve(
            T=T, A=A, R=R, initial_wealth=iw, goal_set=goal_set, D=D,
        )
        feas = result.feasible
        if not cert:
            n_cert_rejects += 1
        if feas and not cert:
            false_prunes += 1
        records.append((T, cert, feas))

    # Monotonicity over the scanned window
    feas_seq = [f for (_, _, f) in records]
    monotonicity_violations = sum(
        1 for i in range(len(feas_seq) - 1) if feas_seq[i] and not feas_seq[i + 1]
    )
    T_star = next((T for (T, _, f) in records if f), None)
    return AuditResult(
        scenario.name, seed, floor, T_star, n_cert_rejects, false_prunes,
        monotonicity_violations, records,
    )


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_headline(scenarios: List[Scenario], n_sims: int, seed: int,
                     methods: List[str]) -> None:
    print(f"\n[1] Headline (single seed={seed}, n_sims={n_sims})")
    print("=" * 100)
    print(f"{'scenario':<28} {'method':<10} {'T*':>4} {'evals':>6} "
          f"{'ΣT cost':>8} {'time(s)':>8}  match")
    print("-" * 100)
    totals = {m: [0, 0] for m in methods}  # [evals, cost]
    for sc in scenarios:
        oracle_T = None
        for m in methods:
            r = run_method(sc, m, n_sims, seed)
            if m == "linear":
                oracle_T = r.T_star
            match = ""
            if m != "linear" and oracle_T is not None and r.T_star is not None:
                match = "OK" if r.T_star == oracle_T else f"MISMATCH(orc={oracle_T})"
            if r.error:
                print(f"{sc.name:<28} {m:<10} {'--':>4} {'--':>6} {'--':>8} "
                      f"{r.wall_time:>8.2f}  ERR:{r.error}")
            else:
                totals[m][0] += r.n_evals
                totals[m][1] += r.weighted_cost
                print(f"{sc.name:<28} {m:<10} {r.T_star:>4} {r.n_evals:>6} "
                      f"{r.weighted_cost:>8} {r.wall_time:>8.2f}  {match}")
        print("-" * 100)
    print("Totals (lower is better):")
    for m in methods:
        print(f"  {m:<10} solves={totals[m][0]:>4}   ΣT-cost={totals[m][1]:>6}")


def section_seed_sweep(scenarios: List[Scenario], n_sims: int,
                       seeds: List[int]) -> bool:
    print(f"\n[2] Seed sweep ({len(seeds)} seeds, n_sims={n_sims}) "
          f"— correctness: bracketed.T == oracle.T for every seed")
    print("=" * 100)
    print(f"{'scenario':<28} {'mismatches':>10} "
          f"{'bracketed evals (min/med/max)':>32} {'binary med':>11}")
    print("-" * 100)
    all_ok = True
    for sc in scenarios:
        mismatches = 0
        br_evals, bin_evals = [], []
        for s in seeds:
            orc = run_method(sc, "linear", n_sims, s)
            br = run_method(sc, "bracketed", n_sims, s)
            bn = run_method(sc, "binary", n_sims, s)
            if br.T_star != orc.T_star:
                mismatches += 1
            if br.n_evals is not None:
                br_evals.append(br.n_evals)
            if bn.n_evals is not None:
                bin_evals.append(bn.n_evals)
        all_ok = all_ok and (mismatches == 0)
        flag = "" if mismatches == 0 else "  <-- FAIL"
        med_bin = int(statistics.median(bin_evals)) if bin_evals else 0
        if br_evals:
            stat = f"{min(br_evals)}/{int(statistics.median(br_evals))}/{max(br_evals)}"
        else:
            stat = "n/a"
        print(f"{sc.name:<28} {mismatches:>10} {stat:>32} {med_bin:>11}{flag}")
    print("-" * 100)
    print(f"Seed-sweep correctness: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return all_ok


def section_audit(scenarios: List[Scenario], n_sims: int,
                  seeds: List[int]) -> bool:
    print(f"\n[3] Certificate & monotonicity audit ({len(seeds)} seeds, "
          f"n_sims={n_sims}) — false_prunes and monotonicity_violations MUST be 0")
    print("=" * 100)
    print(f"{'scenario':<28} {'seeds':>6} {'cert_rejects':>13} "
          f"{'false_prunes':>13} {'monot_viol':>11}")
    print("-" * 100)
    all_ok = True
    for sc in scenarios:
        tot_rej = tot_fp = tot_mv = 0
        for s in seeds:
            a = audit_scenario(sc, n_sims, s)
            tot_rej += a.n_cert_rejects
            tot_fp += a.false_prunes
            tot_mv += a.monotonicity_violations
        ok = (tot_fp == 0 and tot_mv == 0)
        all_ok = all_ok and ok
        flag = "" if ok else "  <-- FAIL"
        print(f"{sc.name:<28} {len(seeds):>6} {tot_rej:>13} "
              f"{tot_fp:>13} {tot_mv:>11}{flag}")
    print("-" * 100)
    print(f"Audit: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return all_ok


def section_nsims_sweep(scenario: Scenario, seed: int,
                        n_sims_grid: List[int]) -> None:
    print(f"\n[4] n_sims sweep on '{scenario.name}' (seed={seed}) "
          f"— correctness & bracket quality vs sample size")
    print("=" * 100)
    print(f"{'n_sims':>8} {'oracle T*':>10} {'bracketed T*':>13} "
          f"{'evals':>6} {'match':>7}")
    print("-" * 100)
    for n in n_sims_grid:
        orc = run_method(scenario, "linear", n, seed)
        br = run_method(scenario, "bracketed", n, seed)
        match = "OK" if br.T_star == orc.T_star else "MISMATCH"
        print(f"{n:>8} {str(orc.T_star):>10} {str(br.T_star):>13} "
              f"{str(br.n_evals):>6} {match:>7}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon search benchmark + validation")
    parser.add_argument("--n-sims", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for the sweep (>1 enables section 2)")
    parser.add_argument("--audit", action="store_true",
                        help="Run certificate & monotonicity audit (section 3)")
    parser.add_argument("--nsims-sweep", action="store_true",
                        help="Run n_sims sweep (section 4)")
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--methods", nargs="+",
                        default=["linear", "binary", "bracketed"])
    args = parser.parse_args()

    scenarios = build_scenarios()
    print(f"\nHorizon search benchmark + validation "
          f"({len(scenarios)} scenarios)")

    section_headline(scenarios, args.n_sims, args.seed, args.methods)

    seeds = list(range(args.seed, args.seed + max(args.seeds, 1)))
    if args.all or args.seeds > 1:
        section_seed_sweep(scenarios, args.n_sims, seeds)

    if args.all or args.audit:
        # Audit is expensive (solves every T up to T*+buffer); use smaller-T
        # scenarios and the first few seeds.
        audit_scs = [s for s in scenarios
                     if s.name in {
                         "terminal_easy", "already_satisfied", "with_withdrawals",
                         "nonzero_initial_wealth", "multi_terminal_diff_accounts",
                         "intermediate_plus_terminal",
                     }]
        section_audit(audit_scs, args.n_sims, seeds[:min(len(seeds), 5)])

    if args.all or args.nsims_sweep:
        rep = next(s for s in scenarios if s.name == "terminal_midrange")
        section_nsims_sweep(rep, args.seed, [100, 200, 300, 500, 1000])

    print()


if __name__ == "__main__":
    main()
