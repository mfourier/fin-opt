"""
Investment modeling module for FinOpt.

Purpose
-------
This module models *how capital grows* given monthly contributions and
(arithmetic) return paths. It provides:
- A single-asset accumulation engine (`simulate_capital`).
- A multi-account simulator (`simulate_portfolio`) to track independent
  accounts (e.g., housing, emergency, brokerage) and their aggregate wealth.

Conventions & Assumptions
-------------------------
- Time unit: monthly, horizon length T.
- Returns are *arithmetic monthly rates* r_t (e.g., 0.01 == +1%).
- Wealth recursion (order of operations each month):
      W[t+1] = (W[t] + a[t]) * (1 + r[t])
  i.e., contributions are added *before* applying returns.
- Contributions may be negative to represent withdrawals.
- Indexing: outputs are pandas Series/DataFrames indexed monthly when a
  DatetimeIndex is available (via `align_index_like`), otherwise a sensible
  default is built.

Main API
--------
- simulate_capital: wealth path for a single stream of contributions and returns.
- simulate_portfolio: parallel simulation of multiple accounts; adds a "total"
  column with the row-wise sum by default.

Example
-------
>>> import numpy as np
>>> contrib = np.full(12, 50_000.0)
>>> path = np.full(12, 0.005)   # 0.5% monthly (arithmetic)
>>> wealth = simulate_capital(contrib, path)
>>> float(wealth.iloc[-1]) > 0
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

)

__all__ = ["simulate_capital", "simulate_portfolio"]



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
    
    """Simulate the monthly wealth path under contributions and arithmetic returns.

    Wealth dynamics (monthly)
    -------------------------
    W[t+1] = (W[t] + a[t]) * (1 + r[t])

    Parameters
    ----------
    contributions : array-like of shape (T,)
        Monthly contributions a[t]. Finite values required.
        Negative values are allowed (withdrawals).
    returns : float or array-like
        Either a scalar monthly rate r (broadcast to length T), or a path r[t]
        of shape (T,). Values are **arithmetic** returns (0.01 == +1%).
    start_value : float, default 0.0
        Initial wealth W[0].
    index_like : pandas Index/Series/DataFrame, optional
        If provided, attempts to reuse a DatetimeIndex (first-of-month)
        for the output via `align_index_like(T, index_like)`.
    clip_negative_wealth : bool, default True
        If True, floors wealth at 0 to avoid negative trajectories after
        applying contributions and returns.

    Returns
    -------
    pd.Series
        Wealth path of length T. If a suitable DatetimeIndex can be inferred
        from `index_like`, it is reused; otherwise a default monthly index
        is created.

    Raises
    ------
    ValueError
        If lengths are inconsistent, arrays are not 1-D, or contain non-finite
        values.

    Notes
    -----
    - Complexity: O(T).
    - The contribution is applied *before* returns each month.
    - Use `compute_metrics` for final wealth, CAGR, drawdown, etc.

    Examples
    --------
    Constant rate:
    >>> import numpy as np
    >>> a = np.full(6, 1000.0)
    >>> r = 0.01
    >>> simulate_capital(a, r).round(2).iloc[-1] > 0
    True

    Time-varying path and a withdrawal:
    >>> a = np.array([1000, 1000, -500, 1000, 1000, 1000], dtype=float)
    >>> r = np.array([0.0, 0.01, 0.005, -0.02, 0.01, 0.0], dtype=float)
    >>> simulate_capital(a, r).shape[0] == 6
    True
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
    contributions_matrix,
    returns_matrix,
    *,
    start_values=0.0,
    index_like=None,
    column_names=None,
    clip_negative_wealth=True,
    include_total_col=True,
):
    """
    Simulate multiple independent accounts/portfolios in parallel.

    Each column represents one account k=1..K with its own contribution
    stream a^{(k)}_t and return path r^{(k)}_t. Wealth evolves as:

        W^{(k)}_{t+1} = (W^{(k)}_t + a^{(k)}_t) * (1 + r^{(k)}_t)

    The aggregate wealth is the sum across accounts:
        W_t = sum_k W^{(k)}_t

    Parameters
    ----------
    contributions_matrix : array-like or DataFrame of shape (T, K)
        Per-account monthly contributions. Negative values allowed.
    returns_matrix : scalar, (T,), or (T,K)
        - Scalar: applied to all accounts.
        - (T,): same path applied to all accounts.
        - (T,K): per-account return paths.
    start_values : float or array-like of shape (K,), default 0.0
        Initial wealths for each account.
    index_like : optional
        Reference for building the output index (DatetimeIndex if available).
    column_names : list of str, optional
        Names for the account columns. Defaults to ["acct_1", ..., "acct_K"].
    clip_negative_wealth : bool, default True
        If True, floors each wealth path at zero.
    include_total_col : bool, default True
        If True, adds a "total" column with the row-wise sum.

    Returns
    -------
    pd.DataFrame
        Wealth paths of shape (T, K [+1]) with accounts as columns (plus "total").
    """
    import numpy as np
    import pandas as pd

    # Normalize contributions
    if isinstance(contributions_matrix, pd.DataFrame):
        A = contributions_matrix.to_numpy(dtype=float, copy=False)
        if index_like is None:
            index_like = contributions_matrix.index
        if column_names is None:
            column_names = list(contributions_matrix.columns)
    else:
        A = np.asarray(contributions_matrix, dtype=float)

    if A.ndim != 2:
        raise ValueError(f"contributions_matrix must be 2-D, got {A.shape}.")
    if not np.isfinite(A).all():
        raise ValueError("contributions_matrix must contain only finite values.")

    T, K = A.shape

    # Broadcast returns to (T,K)
    def _broadcast_returns(Raw):
        if np.isscalar(Raw):
            return np.full((T, K), float(Raw), dtype=float)
        R = np.asarray(Raw, dtype=float)
        if R.ndim == 1:
            if R.shape[0] != T:
                raise ValueError("returns (T,) must match horizon length T.")
            return np.repeat(R.reshape(T, 1), K, axis=1)
        if R.ndim == 2:
            if R.shape != (T, K):
                raise ValueError(f"returns must be (T,K), got {R.shape}.")
            return R
        raise ValueError("returns must be scalar, (T,), or (T,K).")

    R = _broadcast_returns(returns_matrix)
    if not np.isfinite(R).all():
        raise ValueError("returns_matrix must contain only finite values.")

    # Start values
    if np.isscalar(start_values):
        starts = np.full(K, float(start_values), dtype=float)
    else:
        starts = np.asarray(start_values, dtype=float)
        if starts.shape != (K,):
            raise ValueError(f"start_values must be scalar or (K,), got {starts.shape}.")
        if not np.isfinite(starts).all():
            raise ValueError("start_values must contain only finite values.")

    # Run simulate_capital per account
    cols = []
    for k in range(K):
        w_k = simulate_capital(
            contributions=A[:, k],
            returns=R[:, k],
            start_value=starts[k],
            index_like=index_like,
            clip_negative_wealth=clip_negative_wealth,
        )
        cols.append(w_k)

    df = pd.concat(cols, axis=1)
    if column_names is None:
        column_names = [f"acct_{i+1}" for i in range(K)]
    df.columns = column_names
    if include_total_col:
        df["total"] = df.sum(axis=1)
    return df

