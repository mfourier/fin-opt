"""
Scenario engine for FinOpt

Purpose
-------
This module builds investment scenarios by connecting:
- IncomeModel → monthly contributions (aggregate).
- Allocation rules → per-account contributions (constant weights or time-varying matrix C[t,k]).
- Return paths → per-account returns (scalar, (T,), or (T,K)).
Then simulates (single-asset or multi-account) wealth and reports metrics.

Backwards compatibility
-----------------------
- Keeps the legacy single-asset API used by optimization.py:
  * ScenarioConfig, ScenarioResult (single series)
  * SimulationEngine.run_three_cases() and .run_monte_carlo()
- Adds a multi-account API:
  * MultiScenarioResult
  * SimulationEngine.run_case_named(...)  (general runner)
  * Helpers to build returns/allocations for (T,K)

Design goals
------------
- Deterministic by default; explicit RNG seed for any stochasticity.
- Calendar-first: preserves a monthly DatetimeIndex across outputs.
- Flexible inputs: proportional weights or full C[t,k]; scalar/paths/matrices of returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .income import IncomeModel
from .investment import simulate_capital, simulate_portfolio, allocate_contributions
from .utils import (
    # arrays / indexing
    ensure_1d,
    to_series,
    month_index,
    # randomness / scenarios
    set_random_seed,
    fixed_rate_path,
    lognormal_iid,
    # metrics
    compute_metrics,
    PortfolioMetrics,
)

__all__ = [
    # Back-compat (single-asset)
    "ScenarioConfig",
    "ScenarioResult",
    "SimulationEngine",
    # Multi-account result
    "MultiScenarioResult",
]

# ---------------------------------------------------------------------------
# Config and results (legacy single-asset)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    months: int
    start: Optional[date] = None
    # Contribution rule over fixed/variable income
    alpha_fixed: float = 0.35
    beta_variable: float = 1.00
    # Three-case deterministic rates (monthly arithmetic)
    base_r: float = 0.004
    optimistic_r: float = 0.007
    pessimistic_r: float = 0.001
    # Monte Carlo (optional)
    mc_mu: float = 0.004
    mc_sigma: float = 0.02
    mc_paths: int = 0
    seed: Optional[int] = 42


@dataclass(frozen=True)
class ScenarioResult:
    """Legacy single-asset result (used by optimization.py)."""
    name: str
    contributions: pd.Series
    returns: pd.Series
    wealth: pd.Series
    metrics: PortfolioMetrics


# ---------------------------------------------------------------------------
# Multi-account result (new)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiScenarioResult:
    name: str
    contributions_total: pd.Series           # (T,)
    contributions_by_account: pd.DataFrame   # (T,K)
    returns_by_account: pd.DataFrame         # (T,K)
    wealth_by_account: pd.DataFrame          # (T,K [+ 'total'])
    metrics: PortfolioMetrics                # computed on total wealth


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """Scenario builder supporting:
       (1) legacy single-asset runs (back-compat), and
       (2) general multi-account runs with time-varying C[t,k].
    """

    def __init__(
        self,
        income: IncomeModel,
        config: ScenarioConfig,
        accounts: Optional[Sequence[str]] = None,
    ):
        if config.months <= 0:
            raise ValueError("months must be positive.")
        self.income = income
        self.cfg = config
        # Default single aggregate if not provided (for multi-account API)
        self.accounts: list[str] = list(accounts) if accounts is not None else ["acct_1"]

    # ===================== Common helpers =====================

    def _index(self) -> pd.DatetimeIndex:
        """Return the calendar index to be used across series."""
        try:
            contrib = self.income.contributions(
                months=self.cfg.months,
                alpha_fixed=self.cfg.alpha_fixed,
                beta_variable=self.cfg.beta_variable,
                start=self.cfg.start,
            )
            return contrib.index
        except Exception:
            return month_index(self.cfg.start, self.cfg.months)

    def build_contributions_total(
        self,
        *,
        alpha_fixed: Optional[float] = None,
        beta_variable: Optional[float] = None,
    ) -> pd.Series:
        """Aggregate monthly contributions via the α,β rule on income."""
        return self.income.contributions(
            months=self.cfg.months,
            alpha_fixed=self.cfg.alpha_fixed if alpha_fixed is None else float(alpha_fixed),
            beta_variable=self.cfg.beta_variable if beta_variable is None else float(beta_variable),
            start=self.cfg.start,
        )

    # ===================== Legacy single-asset API =====================

    def _returns_path_single(self, r: float) -> pd.Series:
        """Fixed-rate return path aligned to the calendar (single-asset)."""
        path = fixed_rate_path(self.cfg.months, r)
        return to_series(path, self._index(), name="returns")

    def run_case(self, name: str, r: float) -> ScenarioResult:
        """Single-asset scenario (legacy)."""
        contrib = self.build_contributions_total()
        r_path = self._returns_path_single(r)
        wealth = simulate_capital(
            ensure_1d(contrib.values, name="contributions"),
            ensure_1d(r_path.values, name="returns"),
            index_like=contrib.index,
        )
        metrics = compute_metrics(wealth, contributions=contrib.values)
        return ScenarioResult(name=name, contributions=contrib, returns=r_path, wealth=wealth, metrics=metrics)

    def run_three_cases(self) -> Dict[str, ScenarioResult]:
        """Run base/optimistic/pessimistic with fixed monthly rates (single-asset)."""
        return {
            "base": self.run_case("base", self.cfg.base_r),
            "optimistic": self.run_case("optimistic", self.cfg.optimistic_r),
            "pessimistic": self.run_case("pessimistic", self.cfg.pessimistic_r),
        }

    def run_monte_carlo(self) -> Optional[Dict[str, ScenarioResult]]:
        """Run IID lognormal Monte Carlo (single-asset) if mc_paths > 0."""
        if self.cfg.mc_paths <= 0:
            return None
        set_random_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        contrib = self.build_contributions_total()
        results: Dict[str, ScenarioResult] = {}
        for k in range(self.cfg.mc_paths):
            seed_k = int(rng.integers(0, 1_000_000_000))
            r_path = lognormal_iid(self.cfg.months, mu=self.cfg.mc_mu, sigma=self.cfg.mc_sigma, seed=seed_k)
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

    # ===================== Multi-account API (new) =====================

    @staticmethod
    def _normalize_rows(C: np.ndarray) -> np.ndarray:
        rs = C.sum(axis=1, keepdims=True)
        Cn = C.copy()
        mask = (rs.squeeze() > 0.0)
        Cn[mask] = C[mask] / rs[mask]
        return Cn

    def allocate_by_weights(
        self,
        contributions_total: pd.Series,
        *,
        weights_by_account: Optional[Mapping[str, float]] = None,
        C_matrix: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Build (T,K) contributions by account from either:
        - weights_by_account: constant proportions; or
        - C_matrix: time-varying weight matrix (T,K) aligned to the calendar.
        """
        idx = contributions_total.index
        K = len(self.accounts)

        if (weights_by_account is None) and (C_matrix is None):
            # default: equal weights
            w = np.full(K, 1.0 / K, dtype=float)
            C_df = pd.DataFrame(np.repeat(w.reshape(1, -1), len(idx), axis=0), index=idx, columns=self.accounts)
        elif C_matrix is not None:
            if isinstance(C_matrix, pd.DataFrame):
                C_df = C_matrix.copy()
                if not C_df.index.equals(idx):
                    C_df = C_df.reindex(idx)
                # ensure column order
                if list(C_df.columns) != self.accounts:
                    if set(self.accounts).issubset(C_df.columns):
                        C_df = C_df[self.accounts]
                    else:
                        C_df = C_df.set_axis(self.accounts, axis=1)
            else:
                C_arr = np.asarray(C_matrix, dtype=float)
                if C_arr.shape != (len(idx), K):
                    raise ValueError(f"C_matrix must have shape (T={len(idx)}, K={K}).")
                C_arr = self._normalize_rows(C_arr)
                C_df = pd.DataFrame(C_arr, index=idx, columns=self.accounts)
        else:
            # constant weights
            items = [(str(k), float(v)) for k, v in weights_by_account.items() if float(v) > 0.0]
            if not items:
                raise ValueError("weights_by_account must contain at least one positive weight.")
            name2w = dict(items)
            w = np.array([name2w.get(name, 0.0) for name in self.accounts], dtype=float)
            if w.sum() <= 0:
                raise ValueError("Sum of provided weights must be > 0.")
            w = w / w.sum()
            C_df = pd.DataFrame(np.repeat(w.reshape(1, -1), len(idx), axis=0), index=idx, columns=self.accounts)

        # Delegate to robust allocator
        A_df = allocate_contributions(
            contributions=contributions_total,
            C=C_df,
            normalize_rows=True,
            rowsum_tol=1e-8,
        )
        return A_df

    def build_returns_by_account(
        self,
        *,
        returns_by_account: Union[
            float,
            Sequence[float],
            np.ndarray,
            pd.DataFrame,
            Mapping[str, object],
        ] = 0.0,
    ) -> pd.DataFrame:
        """
        Normalize returns input to a (T,K) DataFrame aligned to the calendar.

        Accepted forms:
        - scalar r: same scalar for all accounts/months.
        - (T,) array/Series: same path for all accounts.
        - (T,K) array/DataFrame: per-account paths.
        - mapping {account: r_spec} where each r_spec is scalar | (T,) | Series.
        """
        T = int(self.cfg.months)
        idx = month_index(self.cfg.start, T)
        K = len(self.accounts)

        # Case A: mapping per account
        if isinstance(returns_by_account, Mapping):
            cols: dict[str, np.ndarray] = {}
            for name in self.accounts:
                spec = returns_by_account.get(name, 0.0)
                if np.isscalar(spec):
                    cols[name] = np.full(T, float(spec), dtype=float)
                else:
                    arr = np.asarray(spec, dtype=float)
                    if arr.ndim != 1 or arr.shape[0] != T:
                        raise ValueError(f"Return path for '{name}' must be (T,) with T={T}.")
                    cols[name] = arr
            return pd.DataFrame(cols, index=idx, columns=self.accounts)

        # Case B: DataFrame (T,K)
        if isinstance(returns_by_account, pd.DataFrame):
            Rdf = returns_by_account.copy()
            if not Rdf.index.equals(idx):
                Rdf = Rdf.reindex(idx)
            if list(Rdf.columns) != self.accounts:
                if set(self.accounts).issubset(Rdf.columns):
                    Rdf = Rdf[self.accounts]
                else:
                    Rdf = Rdf.set_axis(self.accounts, axis=1)
            return Rdf.astype(float)

        # Case C: ndarray/list/tuple
        if isinstance(returns_by_account, (np.ndarray, list, tuple)):
            arr = np.asarray(returns_by_account, dtype=float)
            if arr.ndim == 2:
                if arr.shape != (T, K):
                    raise ValueError(f"(T,K) returns must match (T={T}, K={K}); got {arr.shape}.")
                return pd.DataFrame(arr, index=idx, columns=self.accounts)
            elif arr.ndim == 1:
                if arr.shape[0] != T:
                    raise ValueError(f"(T,) returns must have T={T}; got {arr.shape[0]}.")
                R = np.repeat(arr.reshape(T, 1), K, axis=1)
                return pd.DataFrame(R, index=idx, columns=self.accounts)
            else:
                raise ValueError("returns_by_account array must be 1-D or 2-D.")

        # Case D: scalar broadcast
        if np.isscalar(returns_by_account):
            r = float(returns_by_account)
            R = np.full((T, K), r, dtype=float)
            return pd.DataFrame(R, index=idx, columns=self.accounts)

        raise TypeError("Unsupported type for returns_by_account.")


    def run_case_named(
        self,
        name: str,
        *,
        weights_by_account: Optional[Mapping[str, float]] = None,
        C_matrix: Optional[pd.DataFrame | np.ndarray] = None,
        returns_by_account: Union[
            float, Sequence[float], np.ndarray, pd.DataFrame, Mapping[str, object]
        ] = 0.0,
        start_values: Union[float, Sequence[float]] = 0.0,
        include_total_col: bool = True,
        alpha_fixed: Optional[float] = None,
        beta_variable: Optional[float] = None,
    ) -> MultiScenarioResult:
        """
        General multi-account runner:
        - Builds contributions via (alpha,beta),
        - Allocates per-account contributions from constant weights or C[t,k],
        - Normalizes returns to (T,K),
        - Simulates multi-account wealth and computes metrics on total.
        """
        contrib_total = self.build_contributions_total(alpha_fixed=alpha_fixed, beta_variable=beta_variable)
        A_df = self.allocate_by_weights(contrib_total, weights_by_account=weights_by_account, C_matrix=C_matrix)
        R_df = self.build_returns_by_account(returns_by_account=returns_by_account)

        W_df = simulate_portfolio(
            contributions_matrix=A_df,
            returns_matrix=R_df.values,
            start_values=start_values,
            index_like=A_df.index,
            column_names=list(self.accounts),
            include_total_col=include_total_col,
        )
        total_series = W_df["total"] if include_total_col else W_df.sum(axis=1)
        metrics = compute_metrics(total_series, contributions=contrib_total.values)

        return MultiScenarioResult(
            name=name,
            contributions_total=contrib_total,
            contributions_by_account=A_df,
            returns_by_account=R_df,
            wealth_by_account=W_df,
            metrics=metrics,
        )

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

from typing import Tuple, Dict as _Dict

def _fmt_millions():
    import matplotlib.ticker as mticker
    return mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M")


def plot_scenario(
    result: "MultiScenarioResult",
    *,
    show: bool = True,
) -> Tuple["plt.Figure", Tuple["plt.Axes", "plt.Axes"]]:
    """
    Quick plots for a multi-account scenario:
      (1) Wealth by account (+ total dashed)
      (2) Monthly contributions by account (+ total dashed)

    Parameters
    ----------
    result : MultiScenarioResult
        Output from SimulationEngine.run_case_named(...).
    show : bool
        If True, calls plt.show() for each figure.

    Returns
    -------
    (fig, (ax1, ax2)) : tuple
        Figure and axes for later customization.
    """
    import matplotlib.pyplot as plt

    A_df = result.contributions_by_account
    W_df = result.wealth_by_account
    contrib_total = result.contributions_total

    millions_fmt = _fmt_millions()

    # (1) Wealth
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for col in A_df.columns:
        W_df[col].plot(ax=ax1, lw=1.8, label=f"Wealth: {col}")
    if "total" in W_df.columns:
        W_df["total"].plot(ax=ax1, lw=2.5, linestyle="--", label="Wealth: total")

    ax1.set_title("Multi-account wealth trajectories")
    ax1.set_ylabel("Wealth (Million CLP)")
    ax1.yaxis.set_major_formatter(millions_fmt)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    if show:
        plt.show()

    # (2) Contributions
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for col in A_df.columns:
        A_df[col].plot(ax=ax2, lw=1.5, label=f"Contribution: {col}")
    contrib_total.plot(ax=ax2, lw=2.0, linestyle="--", color="black", label="Contribution: total")

    ax2.set_title("Monthly contributions by account")
    ax2.set_ylabel("Contribution (Million CLP)")
    ax2.yaxis.set_major_formatter(millions_fmt)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    if show:
        plt.show()

    return fig1, (ax1, ax2)


def plot_scenarios(
    results: _Dict[str, "MultiScenarioResult"],
    *,
    show: bool = True,
) -> "plt.Figure":
    """
    Compare total wealth across multiple scenarios on a single chart.

    Parameters
    ----------
    results : dict[str, MultiScenarioResult]
        Mapping scenario_name -> result.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    millions_fmt = _fmt_millions()

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        W = res.wealth_by_account
        total_series = W["total"] if "total" in W.columns else W.sum(axis=1)
        total_series.plot(ax=ax, lw=2.0, label=f"{name}: total")

    ax.set_title("Scenario comparison: total wealth")
    ax.set_ylabel("Wealth (Million CLP)")
    ax.yaxis.set_major_formatter(millions_fmt)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if show:
        plt.show()
    return fig
