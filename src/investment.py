"""Investment modeling module for FinOpt

Key components
--------------
- simulate_capital: core capital accumulation with scalar or path returns.
- simulate_portfolio: multi-asset version using weights and asset returns.
- scenario generators: IID lognormal (arithmetic returns), fixed rate.
- metrics: final value, total contributions, CAGR, volatility, max drawdown.

Design goals
------------
- Deterministic by default; stochastic only via explicit seed.

Example
-------
>>> import numpy as np
>>> contrib = np.full(12, 50000.0)
>>> path = np.full(12, 0.005)  # 0.5% monthly
>>> wealth = simulate_capital(contrib, path)
>>> round(float(wealth.iloc[-1]), 2) > 0
True
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .utils import (
    ensure_1d,
    align_index_like,
    compute_cagr,
    drawdown,
)

__all__ = [
    "simulate_capital",
    "simulate_portfolio",
    "lognormal_iid",
    "fixed_rate_path",
    "PortfolioMetrics",
    "compute_metrics",
]


# ---------------------------------------------------------------------------
# Utilities & Types
# ---------------------------------------------------------------------------

ArrayLike = Sequence[float] | np.ndarray | pd.Series


# ---------------------------------------------------------------------------
# Core Simulators
# ---------------------------------------------------------------------------

def simulate_capital(
    contributions: ArrayLike,
    returns: float | ArrayLike,
    *,
    start_value: float = 0.0,
    index_like: Optional[pd.Index | pd.Series | pd.DataFrame] = None,
    clip_negative_wealth: bool = True,
) -> pd.Series:
    """Simulate wealth evolution with contributions and (possibly varying) returns.

    Dynamics (monthly):
        W[t+1] = (W[t] + a[t]) * (1 + r[t])

    Parameters
    ----------
    contributions : array-like shape (T,)
        Monthly contributions a[t]. Must be finite; negatives allowed to
        represent withdrawals.
    returns : float or array-like
        Either a scalar monthly rate r or a path of shape (T,).
        Arithmetic returns (e.g., 0.01 => +1%).
    start_value : float, default 0.0
        Initial wealth W[0].
    index_like : pandas Index/Series/DataFrame, optional
        If provided, attempts to reuse a DatetimeIndex for the output.
    clip_negative_wealth : bool, default True
        If True, floors wealth at zero to avoid negative trajectories.

    Returns
    -------
    pd.Series of length T with DatetimeIndex when available.
    """
    a = ensure_1d(contributions, name="contributions")
    T = a.shape[0]

    if isinstance(returns, (float, int)):
        r = np.full(T, float(returns), dtype=float)
    else:
        r = ensure_1d(returns, name="returns")
        if r.shape[0] != T:
            raise ValueError(
                f"returns length {r.shape[0]} must match contributions length {T}."
            )

    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("contributions and returns must be finite.")

    W = np.empty(T, dtype=float)
    w_prev = float(start_value)
    for t in range(T):
        w_t = (w_prev + a[t]) * (1.0 + r[t])
        if clip_negative_wealth and w_t < 0.0:
            w_t = 0.0
        W[t] = w_t
        w_prev = w_t

    idx = align_index_like(T, index_like)
    return pd.Series(W, index=idx, name="wealth")


def simulate_portfolio(
    contributions: ArrayLike,
    asset_returns: np.ndarray | pd.DataFrame,
    weights: ArrayLike | np.ndarray,
    *,
    start_value: float = 0.0,
    rebalance: bool = True,
    index_like: Optional[pd.Index | pd.Series | pd.DataFrame] = None,
    clip_negative_wealth: bool = True,
) -> pd.Series:
    """Simulate wealth with a multi-asset portfolio.

    Parameters
    ----------
    contributions : array-like shape (T,)
        Monthly contributions a[t].
    asset_returns : array shape (T, N)
        Arithmetic returns per asset and month. DataFrame allowed.
    weights : array-like shape (N,) or (T, N)
        Portfolio weights. If shape (N,), constant weights with monthly
        *rebalancing* when `rebalance=True`. If shape (T, N), treated as a
        pre-specified schedule of weights at the *start* of each month t.
    start_value : float, default 0.0
        Initial wealth W[0].
    rebalance : bool, default True
        When weights are (N,), if True rebalance to target each month; if
        False, weights drift with returns (buy-and-hold on existing capital;
        contributions are added pro-rata to target weights).
    index_like : optional
        Datetime index source.
    clip_negative_wealth : bool, default True
        Floor wealth at zero.

    Returns
    -------
    pd.Series: wealth path (length T).
    """
    a = ensure_1d(contributions, name="contributions")
    R = np.asarray(asset_returns, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"asset_returns must be 2-D, got shape {R.shape}.")
    T, N = R.shape

    if a.shape[0] != T:
        raise ValueError(
            f"contributions length {a.shape[0]} must match asset_returns T={T}."
        )

    W = np.empty(T, dtype=float)
    W_prev = float(start_value)

    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        if w.shape[0] != N:
            raise ValueError(f"weights length {w.shape[0]} must equal N={N}.")
        if (w < -1e-12).any():
            raise ValueError("weights must be >= 0 for MVP (no shorting).")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1.0.")
        # Constant target weights
        current_weights = w.copy()
        for t in range(T):
            W_alloc = W_prev + a[t]
            if rebalance and W_alloc > 0:
                # Rebalance to target
                alloc = W_alloc * current_weights
            else:
                # Buy-and-hold drift on previous alloc; add contrib pro-rata
                # to target weights to avoid overweight cash bias.
                alloc = W_prev * current_weights * (1.0 + R[t])
                alloc += a[t] * current_weights
                total = alloc.sum()
                current_weights = alloc / total if total > 0 else current_weights
                W[t] = total
                W_prev = total
                continue

            # Apply returns for month t
            alloc *= (1.0 + R[t])
            total = alloc.sum()
            if clip_negative_wealth and total < 0.0:
                total = 0.0
            W[t] = total
            W_prev = total
            if not rebalance:
                current_weights = alloc / total if total > 0 else current_weights
    elif w.ndim == 2:
        if w.shape != (T, N):
            raise ValueError(
                f"weights shape {w.shape} must match (T={T}, N={N})."
            )
        if (w < -1e-12).any():
            raise ValueError("weights must be >= 0 for MVP (no shorting).")
        if not np.allclose(w.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each row of weights must sum to 1.0.")

        for t in range(T):
            W_alloc = W_prev + a[t]
            alloc = W_alloc * w[t]
            alloc *= (1.0 + R[t])
            total = alloc.sum()
            if clip_negative_wealth and total < 0.0:
                total = 0.0
            W[t] = total
            W_prev = total
    else:
        raise ValueError(f"weights must be 1-D or 2-D, got shape {w.shape}.")

    idx = align_index_like(T, index_like)
    return pd.Series(W, index=idx, name="wealth")


# ---------------------------------------------------------------------------
# Scenario Generators
# ---------------------------------------------------------------------------

def fixed_rate_path(months: int, r: float) -> np.ndarray:
    """Constant arithmetic return path of length `months`."""
    return np.full(months, float(r), dtype=float)


def lognormal_iid(
    months: int,
    *,
    mu: float,
    sigma: float,
    seed: Optional[int] = None,
    drift_in_logs: bool = False,
) -> np.ndarray:
    """Generate IID arithmetic returns from a lognormal model.

    If `drift_in_logs` is False (default), interpret (mu, sigma) as the
    mean/vol of **arithmetic** monthly returns approximately via:
        r_t = exp(m - 0.5*s^2 + s*Z) - 1  with m ≈ ln(1+mu)
    For small returns, mu ≈ expected arithmetic return.

    If `drift_in_logs` is True, interpret mu, sigma directly in log space
    as parameters of log(1+r).
    """
    if months <= 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    s = float(sigma)
    if s < 0:
        raise ValueError("sigma must be non-negative.")
    m = float(mu) if drift_in_logs else float(np.log1p(mu))
    z = rng.normal(size=months)
    y = np.exp(m - 0.5 * s * s + s * z) - 1.0
    return y.astype(float, copy=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioMetrics:
    final_wealth: float
    total_contributions: float
    cagr: float
    vol: float
    max_drawdown: float


def compute_metrics(
    wealth: pd.Series,
    contributions: Optional[ArrayLike] = None,
    *,
    periods_per_year: int = 12,
) -> PortfolioMetrics:
    """Compute basic performance metrics from a wealth path.

    Parameters
    ----------
    wealth : pd.Series
        Wealth path with length T and increasing time index.
    contributions : array-like, optional
        Used to compute total contributions. If None, assumes zero.
    periods_per_year : int, default 12
        Calendar frequency used for CAGR.
    """
    if not isinstance(wealth, pd.Series) or wealth.empty:
        raise ValueError("wealth must be a non-empty pandas Series.")

    W = wealth.astype(float).values
    T = len(W)

    # Total contributions
    tot_contrib = float(np.sum(ensure_1d(contributions, name="contributions"))) if contributions is not None else 0.0

    # CAGR (robust when W0=0) and drawdown from utils
    cagr = float(compute_cagr(wealth, periods_per_year=periods_per_year))
    dd_series = drawdown(wealth)
    max_dd = float(dd_series.min()) if not dd_series.empty else 0.0

    # Volatility approximation from simple returns on wealth
    w_prev = np.roll(W, 1)
    w_prev[0] = np.nan
    simple_ret = (W - w_prev) / w_prev
    vol = float(np.nanstd(simple_ret, ddof=1)) if np.isfinite(simple_ret[1:]).any() else 0.0

    return PortfolioMetrics(
        final_wealth=float(W[-1]),
        total_contributions=tot_contrib,
        cagr=cagr,
        vol=vol,
        max_drawdown=max_dd,
    )


# ---------------------------------------------------------------------------
# Self-test (manual)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    T = 24
    contrib = np.full(T, 700000.0)
    r_path = fixed_rate_path(T, 0.005)  # 0.5% monthly
    wealth = simulate_capital(contrib, r_path)
    print(wealth.tail())
    print(compute_metrics(wealth, contrib))

    # Multi-asset example
    rng = np.random.default_rng(42)
    R = np.column_stack([
        rng.normal(0.006, 0.02, size=T),  # asset 1
        rng.normal(0.003, 0.01, size=T),  # asset 2
    ])
    w = np.array([0.6, 0.4])
    wealth2 = simulate_portfolio(contrib, R, w, rebalance=True)
    print(wealth2.tail())
