"""
optimization.py — Optimization solvers for FinOpt

This module defines *problem-oriented* solvers that integrate with:
- income.py (IncomeModel)
- investment.py (simulate_capital, fixed_rate_path, lognormal_iid, compute_metrics)
- simulation.py (ScenarioConfig, SimulationEngine, ScenarioResult)
- goals.py (Goal, evaluate_goals)
- utils.py (ensure_1d, check_non_negative)

Design principles
-----------------
- Problem-first API: clear dataclasses for inputs/outputs.
- Deterministic by default (stochasticity only via provided seeds/paths).
- No external solver dependencies in the MVP (closed-form + search + MC wrappers).
- Extensible via a registry of solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .income import IncomeModel
from .investment import (
    simulate_capital,
    fixed_rate_path,
    lognormal_iid,
    compute_metrics,
)
from .simulation import ScenarioConfig, SimulationEngine, ScenarioResult
from .goals import Goal, evaluate_goals
from .utils import ensure_1d, check_non_negative


__all__ = [
    # Inputs/Outputs
    "MinContributionInput",
    "MinContributionResult",
    "MinTimeInput",
    "MinTimeResult",
    "ChanceConstraintsInput",
    "ChanceConstraintsResult",
    # Facades
    "min_constant_contribution",
    "min_time_given_contribution",
    "chance_constraints",
    # Registry utilities
    "register_solver",
    "get_solver",
]


# ===========================================================================
# Dataclasses (Inputs / Outputs)
# ===========================================================================

@dataclass(frozen=True)
class MinContributionInput:
    """Input for the minimum constant contribution problem.
    Given target B, start wealth W0 and a returns_path r[0..T-1],
    find the smallest constant a (optionally clamped at 0).
    """
    target_amount: float              # B
    start_wealth: float               # W0
    returns_path: Sequence[float] | np.ndarray | pd.Series  # r[0..T-1]
    non_negative: bool = True         # clamp a >= 0


@dataclass(frozen=True)
class MinContributionResult:
    """Output for the minimum constant contribution problem."""
    a_star: float
    annuity_factor: float             # AF = sum_{t=0}^{T-1} G_{t+1}
    growth_W0: float                  # W0 * G0
    T: int


@dataclass(frozen=True)
class MinTimeInput:
    """Input for the minimum time problem (given a constant contribution)."""
    contribution: float                              # a
    start_wealth: float                              # W0
    returns_path: Sequence[float] | np.ndarray | pd.Series  # r[0..T-1]
    success_threshold: float                         # B
    search_lo_hi: Optional[Tuple[int, int]] = None   # (lo, hi) in months


@dataclass(frozen=True)
class MinTimeResult:
    """Output for the minimum time problem."""
    T_hat: int
    wealth_path: pd.Series


@dataclass(frozen=True)
class ChanceConstraintsInput:
    """Input for chance-constraints (probability of meeting goals).

    Notes
    -----
    - Provide an IncomeModel and a ScenarioConfig with mc_paths > 0.
    - Goals are evaluated on each Monte Carlo wealth path using goals.evaluate_goals.
    """
    goals: Iterable[Goal]
    income_model: IncomeModel
    scen_cfg: ScenarioConfig
    mc_paths: int = 1000             # number of MC paths (overrides scen_cfg.mc_paths if >0)


@dataclass(frozen=True)
class ChanceConstraintsResult:
    """Output for chance-constraints."""
    success_prob_by_goal: pd.Series  # index = goal names
    summary: Dict[str, float]        # e.g., {"joint_success": ... , "paths": N}


# ===========================================================================
# Scenario Provider (adapter over SimulationEngine)
# ===========================================================================

class EngineScenarioProvider:
    """Adapter around SimulationEngine to expose scenario runs uniformly."""

    def __init__(self, income: IncomeModel, cfg: ScenarioConfig):
        self.eng = SimulationEngine(income, cfg)

    def three_cases(self) -> Dict[str, ScenarioResult]:
        return self.eng.run_three_cases()

    def monte_carlo(self) -> Optional[Dict[str, ScenarioResult]]:
        return self.eng.run_monte_carlo()


# ===========================================================================
# Solver Registry (extensibility point)
# ===========================================================================

_SOLVERS: Dict[str, Callable[..., object]] = {}

def register_solver(name: str, fn: Callable[..., object]) -> None:
    """Register a solver function under a name."""
    _SOLVERS[name] = fn

def get_solver(name: str) -> Callable[..., object]:
    """Retrieve a solver function by name."""
    if name not in _SOLVERS:
        raise KeyError(f"Solver '{name}' not registered.")
    return _SOLVERS[name]


# ===========================================================================
# Core Solvers (MVP)
# ===========================================================================

def _solve_min_constant_contribution_closed_form(inp: MinContributionInput) -> MinContributionResult:
    """Closed-form minimum constant contribution under arithmetic returns.

    Dynamics:
        W_{t+1} = (W_t + a) * (1 + r_t),  t = 0..T-1

    Let G_t = Π_{u=t}^{T-1}(1+r_u), with G_T=1.
    Terminal wealth:
        W_T = W_0 * G_0 + a * AF,  where AF = Σ_{t=0}^{T-1} G_{t+1}.
    Hence:
        a* = max(0, (B - W0*G0) / AF) if AF > 0 else +inf (when clamped).
    """
    check_non_negative("target_amount", float(inp.target_amount))
    r = ensure_1d(inp.returns_path, name="returns_path")
    T = int(r.shape[0])
    if T <= 0:
        raise ValueError("returns_path must have positive length.")
    one_plus_r = 1.0 + r

    # Backward cumulative products G_t
    G = np.empty(T + 1, dtype=float)
    G[T] = 1.0
    acc = 1.0
    for t in range(T - 1, -1, -1):
        acc *= one_plus_r[t]
        G[t] = acc

    AF = float(np.sum(G[1:]))                 # annuity factor
    growth_W0 = float(inp.start_wealth) * float(G[0])
    if AF <= 0:
        return MinContributionResult(a_star=float("inf"), annuity_factor=AF, growth_W0=growth_W0, T=T)

    a_star = (float(inp.target_amount) - growth_W0) / AF
    if inp.non_negative:
        a_star = max(0.0, a_star)
    return MinContributionResult(a_star=a_star, annuity_factor=AF, growth_W0=growth_W0, T=T)


def _solve_min_time_given_contribution(inp: MinTimeInput) -> MinTimeResult:
    """Binary search over T for the smallest horizon achieving the target B."""
    r = ensure_1d(inp.returns_path, name="returns_path")
    T_max = len(r)
    lo, hi = inp.search_lo_hi or (1, T_max)
    if lo < 1 or hi > T_max or lo > hi:
        raise ValueError("Invalid search range.")

    def feasible(T: int) -> Tuple[bool, pd.Series]:
        contrib = np.full(T, float(inp.contribution), dtype=float)
        wealth = simulate_capital(contrib, r[:T], start_value=float(inp.start_wealth))
        return (float(wealth.iloc[-1]) >= float(inp.success_threshold)), wealth

    best_T, best_path = None, None
    L, R = lo, hi
    while L <= R:
        mid = (L + R) // 2
        ok, w = feasible(mid)
        if ok:
            best_T, best_path = mid, w
            R = mid - 1
        else:
            L = mid + 1

    if best_T is None:
        # Not feasible within given horizon → return horizon hi and the last path
        _, w = feasible(hi)
        return MinTimeResult(T_hat=hi, wealth_path=w)

    return MinTimeResult(T_hat=best_T, wealth_path=best_path)


def _solve_chance_constraints(inp: ChanceConstraintsInput) -> ChanceConstraintsResult:
    """Monte Carlo wrapper to estimate per-goal and joint success probabilities."""
    # Ensure MC paths
    cfg = inp.scen_cfg
    if inp.mc_paths and inp.mc_paths > 0 and inp.mc_paths != cfg.mc_paths:
        cfg = ScenarioConfig(
            months=cfg.months,
            start=cfg.start,
            alpha_fixed=cfg.alpha_fixed,
            beta_variable=cfg.beta_variable,
            base_r=cfg.base_r,
            optimistic_r=cfg.optimistic_r,
            pessimistic_r=cfg.pessimistic_r,
            mc_mu=cfg.mc_mu,
            mc_sigma=cfg.mc_sigma,
            mc_paths=inp.mc_paths,
            seed=cfg.seed,
        )

    provider = EngineScenarioProvider(inp.income_model, cfg)
    mc = provider.monte_carlo()
    if not mc:
        return ChanceConstraintsResult(
            success_prob_by_goal=pd.Series(dtype=float),
            summary={"joint_success": np.nan, "paths": 0},
        )

    goals_list = list(inp.goals)
    counts = np.zeros(len(goals_list), dtype=int)
    joint_success = 0
    total = len(mc)

    for _, res in mc.items():
        df = evaluate_goals(res.wealth, goals_list, start=cfg.start)
        flags = df["success"].values.astype(bool)
        counts += flags
        if flags.all():
            joint_success += 1

    probs = counts / total
    return ChanceConstraintsResult(
        success_prob_by_goal=pd.Series(probs, index=[g.name for g in goals_list]),
        summary={"joint_success": joint_success / total, "paths": total},
    )


# Register default solvers
register_solver("min_contribution.closed_form", _solve_min_constant_contribution_closed_form)
register_solver("min_time.binary_search", _solve_min_time_given_contribution)
register_solver("chance_constraints.monte_carlo", _solve_chance_constraints)


# ===========================================================================
# Facade (public API)
# ===========================================================================

def min_constant_contribution(inp: MinContributionInput, *, solver: str = "min_contribution.closed_form") -> MinContributionResult:
    """Public API: minimum constant contribution for a fixed horizon.
    You may specify a different solver via `solver` if registered.
    """
    return get_solver(solver)(inp)


def min_time_given_contribution(inp: MinTimeInput, *, solver: str = "min_time.binary_search") -> MinTimeResult:
    """Public API: minimum time to reach a target given a constant contribution."""
    return get_solver(solver)(inp)


def chance_constraints(inp: ChanceConstraintsInput, *, solver: str = "chance_constraints.monte_carlo") -> ChanceConstraintsResult:
    """Public API: chance-constraint estimation via Monte Carlo."""
    return get_solver(solver)(inp)


# ===========================================================================
# Manual quick test (integration smoke tests)
# ===========================================================================
if __name__ == "__main__":
    from datetime import date
    import numpy as np
    import pandas as pd

    from .investment import fixed_rate_path, simulate_capital
    from .income import FixedIncome, VariableIncome, IncomeModel
    from .simulation import ScenarioConfig
    from .goals import Goal

    # ---------------- 1) Closed-form sanity check + verification ----------------
    T = 24
    r = fixed_rate_path(T, 0.004)  # 0.4% monthly
    B = 20_000_000.0
    W0 = 0.0

    mc_inp = MinContributionInput(target_amount=B, start_wealth=W0, returns_path=r)
    mc_res = min_constant_contribution(mc_inp)
    print("[min_constant_contribution] a* (CLP):", round(mc_res.a_star))

    # Verificar por simulación directa que W_T >= B (tolerancia numérica)
    contrib_series = np.full(T, mc_res.a_star, dtype=float)
    wealth_series = simulate_capital(contrib_series, r, start_value=W0)
    WT = float(wealth_series.iloc[-1])
    print("[min_constant_contribution] W_T:", round(WT), ">= B?", WT + 1e-6 >= B)

    # ---------------- 2) Min time given contribution (binary search) -----------
    a_fixed = 700_000.0
    B2 = 6_000_000.0
    mt_inp = MinTimeInput(
        contribution=a_fixed,
        start_wealth=0.0,
        returns_path=r,
        success_threshold=B2,
    )
    mt_res = min_time_given_contribution(mt_inp)
    print("[min_time_given_contribution] T_hat (months):", mt_res.T_hat)

    # Chequeos de borde: en T_hat-1 no debería alcanzar; en T_hat sí debería
    if mt_res.T_hat > 1:
        wealth_Tm1 = simulate_capital(np.full(mt_res.T_hat - 1, a_fixed), r[: mt_res.T_hat - 1], start_value=0.0)
        WTm1 = float(wealth_Tm1.iloc[-1])
        print("[min_time_given_contribution] W_{T_hat-1}:", round(WTm1), "< B?", WTm1 < B2)
    WT_hat = float(mt_res.wealth_path.iloc[-1])
    print("[min_time_given_contribution] W_{T_hat}:", round(WT_hat), ">= B?", WT_hat >= B2)

    # ---------------- 3) Chance constraints with SimulationEngine --------------
    # Income model (matches your project defaults)
    income = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
        variable=VariableIncome(base=200_000.0, sigma=0.00),
    )
    cfg = ScenarioConfig(
        months=24,
        start=date(2025, 9, 1),
        alpha_fixed=0.35,
        beta_variable=1.0,
        base_r=0.004,
        optimistic_r=0.007,
        pessimistic_r=0.001,
        mc_mu=0.004,
        mc_sigma=0.02,
        mc_paths=200,       # pequeño para prueba rápida
        seed=42,
    )

    goals = [
        Goal(name="housing", target_amount=20_000_000.0, target_month_index=23),
        Goal(name="emergency", target_amount=6_000_000.0, target_month_index=11),
    ]

    cc_inp = ChanceConstraintsInput(goals=goals, income_model=income, scen_cfg=cfg, mc_paths=200)
    cc_res = chance_constraints(cc_inp)
    print("[chance_constraints] success_prob_by_goal:")
    print(cc_res.success_prob_by_goal)
    print("[chance_constraints] summary:", cc_res.summary)
