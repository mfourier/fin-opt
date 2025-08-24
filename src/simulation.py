"""Simulation orchestrator for FinOpt

Connects `income.py` and `investment.py` to generate contribution paths,
return scenarios (base/optimistic/pessimistic or simple Monte Carlo), and
simulate the wealth trajectory with core metrics.

Design goals
------------
- Deterministic by default (explicit RNG seed).

Typical usage
-------------
>>> from datetime import date
>>> from .income import FixedIncome, VariableIncome, IncomeModel
>>> from .investment import simulate_capital
>>> from .utils import fixed_rate_path, lognormal_iid, compute_metrics
>>> cfg = ScenarioConfig(
...     months=36,
...     start=date(2025, 9, 1),
...     alpha_fixed=0.35,
...     beta_variable=1.0,
...     base_r=0.004,  # 0.4% monthly (~4.9% annual compounded)
...     optimistic_r=0.007,
...     pessimistic_r=0.001,
... )
>>> income = IncomeModel(
...     fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
...     variable=VariableIncome(base=200_000.0, sigma=0.00),
... )
>>> sim = SimulationEngine(income, cfg)
>>> results = sim.run_three_cases()
>>> results["base"].wealth.tail()
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .income import IncomeModel
from .investment import simulate_capital
from .utils import (
    # arrays / indexing
    ensure_1d,
    to_series,
    month_index,
    align_index_like,
    # randomness / scenarios
    set_random_seed,
    fixed_rate_path,
    lognormal_iid,
    # metrics
    compute_metrics,
    PortfolioMetrics,
)

__all__ = [
    "ScenarioConfig",
    "ScenarioResult",
    "SimulationEngine",
]


# ---------------------------------------------------------------------------
# Config and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    months: int
    start: Optional[date] = None
    # Investment proportions over fixed/variable income
    alpha_fixed: float = 0.35
    beta_variable: float = 1.00
    # Three-case deterministic rates (monthly arithmetic)
    base_r: float = 0.004
    optimistic_r: float = 0.007
    pessimistic_r: float = 0.001
    # Monte Carlo (optional)
    mc_mu: float = 0.004
    mc_sigma: float = 0.02
    mc_paths: int = 0            # 0 disables Monte Carlo in MVP
    seed: Optional[int] = 42


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    contributions: pd.Series
    returns: pd.Series
    wealth: pd.Series
    metrics: PortfolioMetrics


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """High-level orchestrator that builds contributions and return paths
    and runs portfolio simulations (single aggregate asset in the MVP).
    """

    def __init__(self, income: IncomeModel, config: ScenarioConfig):
        if config.months <= 0:
            raise ValueError("months must be positive.")
        self.income = income
        self.cfg = config

    # -------------------- Contributions --------------------
    def build_contributions(self) -> pd.Series:
        """Compute monthly contributions from income proportions."""
        return self.income.contributions(
            months=self.cfg.months,
            alpha_fixed=self.cfg.alpha_fixed,
            beta_variable=self.cfg.beta_variable,
            start=self.cfg.start,
        )

    # -------------------- Deterministic three cases --------------------
    def _returns_path(self, r: float) -> pd.Series:
        """Build a fixed-rate return path indexed to the simulation calendar."""
        path = fixed_rate_path(self.cfg.months, r)
        # Reuse contributions index when available; otherwise fall back to month_index
        contrib_idx = self._index()
        return to_series(path, contrib_idx, name="returns")

    def run_case(self, name: str, r: float) -> ScenarioResult:
        contrib = self.build_contributions()
        r_path = self._returns_path(r)
        # Ensure alignment and pass arrays to the simulator
        wealth = simulate_capital(
            ensure_1d(contrib.values, name="contributions"),
            ensure_1d(r_path.values, name="returns"),
            index_like=contrib.index,
        )
        metrics = compute_metrics(wealth, contributions=contrib.values)
        return ScenarioResult(
            name=name,
            contributions=contrib,
            returns=r_path,
            wealth=wealth,
            metrics=metrics,
        )

    def run_three_cases(self) -> Dict[str, ScenarioResult]:
        """Run base/optimistic/pessimistic scenarios using fixed monthly rates."""
        return {
            "base": self.run_case("base", self.cfg.base_r),
            "optimistic": self.run_case("optimistic", self.cfg.optimistic_r),
            "pessimistic": self.run_case("pessimistic", self.cfg.pessimistic_r),
        }

    # -------------------- Monte Carlo (optional) --------------------
    def run_monte_carlo(self) -> Optional[Dict[str, ScenarioResult]]:
        """Run simple IID lognormal Monte Carlo paths if `mc_paths > 0`."""
        if self.cfg.mc_paths <= 0:
            return None
        set_random_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        contrib = self.build_contributions()
        results: Dict[str, ScenarioResult] = {}
        for k in range(self.cfg.mc_paths):
            seed_k = int(rng.integers(0, 1_000_000_000))
            r_path = lognormal_iid(
                self.cfg.months,
                mu=self.cfg.mc_mu,
                sigma=self.cfg.mc_sigma,
                seed=seed_k,
            )
            r_series = to_series(r_path, contrib.index, name="returns")
            wealth = simulate_capital(contrib.values, r_path, index_like=contrib.index)
            metrics = compute_metrics(wealth, contributions=contrib.values)
            results[f"mc_{k:03d}"] = ScenarioResult(
                name=f"mc_{k:03d}",
                contributions=contrib,
                returns=r_series,
                wealth=wealth,
                metrics=metrics,
            )
        return results

    # -------------------- Helpers --------------------
    def _index(self) -> pd.DatetimeIndex:
        """Return the calendar index to be used across series."""
        # Prefer the income model's contribution index for consistency.
        try:
            contrib = self.income.contributions(
                months=self.cfg.months,
                alpha_fixed=self.cfg.alpha_fixed,
                beta_variable=self.cfg.beta_variable,
                start=self.cfg.start,
            )
            return contrib.index
        except Exception:
            # Fallback to a simple first-of-month index
            return month_index(self.cfg.start, self.cfg.months)


# ---------------------------------------------------------------------------
# Manual quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .income import FixedIncome, VariableIncome

    model = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
        variable=VariableIncome(base=200_000.0, sigma=0.00),
    )
    cfg = ScenarioConfig(months=24, alpha_fixed=0.35, beta_variable=1.0, base_r=0.005)
    eng = SimulationEngine(model, cfg)
    res = eng.run_three_cases()
    for k, v in res.items():
        print(k, v.metrics)
