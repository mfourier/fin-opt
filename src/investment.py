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
from typing import Mapping, Sequence, Optional, Union
import warnings
import numpy as np
import pandas as pd

from .utils import (
    ensure_1d,
    align_index_like,

)

__all__ = ["simulate_capital", "simulate_portfolio", "allocate_contributions", "allocate_contributions_proportional"]



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

# ---------------------------------------------------------------------------
# Contribution allocation (matrix form)
# ---------------------------------------------------------------------------

from .utils import ensure_1d, align_index_like


def allocate_contributions(
    contributions: Union[pd.Series, np.ndarray, Sequence[float]],
    C: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
    *,
    column_names: Optional[Sequence[str]] = None,
    normalize_rows: bool = True,
    rowsum_tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Allocate a monthly contribution series `a[t]` across K accounts using a
    (time‑varying) weight matrix `C` of shape (T, K), where each row represents
    the fraction assigned to each account at month t.

    Parameters
    ----------
    contributions : Union[pd.Series, np.ndarray, Sequence[float]]
        Monthly contributions `a[t]`, length T. Negative values are allowed
        (withdrawals) and will be split according to the same row weights.
        If a pandas Series is provided, its index is propagated to the output.
    C : Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]]
        Weight matrix of shape (T, K) with non‑negative entries. If a
        DataFrame is provided and `contributions` is a Series, an attempt is
        made to align by index; otherwise lengths must match.
    column_names : Optional[Sequence[str]], default None
        Names for the K account columns. If None and `C` is a DataFrame,
        uses `C.columns`; otherwise falls back to `["acct_1", ..., "acct_K"]`.
    normalize_rows : bool, default True
        If True, each row of `C` with positive sum is normalized to sum to 1.
        Rows whose sum is exactly 0 are left as all‑zeros (i.e., no allocation).
        If False, rows must already sum to 1 within `rowsum_tol` (rows with
        sum==0 are still permitted and left as zeros).
    rowsum_tol : float, default 1e-6
        Tolerance to validate that row sums are 1 when `normalize_rows=False`.

    Returns
    -------
    pd.DataFrame
        Allocation matrix `A` of shape (T, K) with entries
        `A[t, k] = a[t] * C[t, k]` and index aligned to `contributions` when
        available.

    Raises
    ------
    ValueError
        If shapes are inconsistent, arrays contain non‑finite values,
        any entry of `C` is negative, or row‑sum validation fails.

    Notes
    -----
    - If a row sum is 0 and `a[t] != 0`, the entire row of allocations is 0.
      This is intentional to represent "no allocation rule for this month".
    - When `normalize_rows=True`, numeric stability is handled by normalizing
      only rows with strictly positive sums.
    - This function does **not** cap allocations when `a[t] < 0`; negative
      contributions (withdrawals) are split proportionally by the same weights.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> a = pd.Series(np.full(3, 1000.0), index=pd.date_range("2025-01-01", periods=3, freq="MS"))
    >>> C = pd.DataFrame([[0.2, 0.8], [0.5, 0.5], [0.0, 0.0]], index=a.index, columns=["goal_A", "goal_B"])
    >>> allocate_contributions(a, C)
                goal_A  goal_B
    2025-01-01   200.0   800.0
    2025-02-01   500.0   500.0
    2025-03-01     0.0     0.0
    """
    # ---- contributions → 1-D ndarray (+ keep index if Series) ----
    if isinstance(contributions, pd.Series):
        a_ser = contributions.astype(float)
        idx = a_ser.index
        a = ensure_1d(a_ser.values, name="contributions")
    else:
        a = ensure_1d(contributions, name="contributions")
        idx = None

    T = a.shape[0]

    # ---- C → (T, K) ndarray (+ try index alignment if DataFrame) ----
    colnames_from_df: Optional[Sequence[str]] = None
    if isinstance(C, pd.DataFrame):
        C_df = C.copy()
        # Try to align by index if contributions provided an index
        if idx is not None and not C_df.index.equals(idx):
            try:
                C_df = C_df.reindex(idx)
            except Exception:
                # Fall back silently; shape validation will catch mismatches
                pass
        C_arr = np.asarray(C_df.values, dtype=float)
        colnames_from_df = list(C_df.columns)
    else:
        C_arr = np.asarray(C, dtype=float)

    if C_arr.ndim != 2:
        raise ValueError(f"C must be 2-D (T, K); got shape {C_arr.shape}.")
    if C_arr.shape[0] != T:
        raise ValueError(f"C rows ({C_arr.shape[0]}) must match contributions length T={T}.")
    if not np.isfinite(C_arr).all():
        raise ValueError("C must contain only finite values.")
    if (C_arr < 0).any():
        raise ValueError("C must be non-negative (percentages/fractions).")

    # ---- Row normalization / validation ----
    row_sums = C_arr.sum(axis=1)
    if normalize_rows:
        C_norm = C_arr.copy()
        mask = row_sums > 0.0
        # Normalize only rows with positive sum
        C_norm[mask] = C_arr[mask] / row_sums[mask][:, None]
        # Rows with sum==0 remain exactly as provided (zeros recommended)
    else:
        # Allow rows that sum to 0 (treated as zero-allocation rows)
        bad = (row_sums > 0.0) & (np.abs(row_sums - 1.0) > rowsum_tol)
        if np.any(bad):
            t_bad = np.where(bad)[0][:5]  # show only first few in error message
            raise ValueError(
                f"Each positive-sum row of C must sum to 1 within tol={rowsum_tol}. "
                f"First bad rows (0-based): {t_bad.tolist()}."
            )
        C_norm = C_arr

    # ---- Compute A[t, k] = a[t] * C[t, k] ----
    A = (a.reshape(T, 1) * C_norm).astype(float)

    # ---- Build output index and column names ----
    out_idx = idx if idx is not None else align_index_like(T, None)

    if column_names is not None:
        if len(column_names) != A.shape[1]:
            raise ValueError("`column_names` length must match number of accounts K.")
        cols = list(column_names)
    elif colnames_from_df is not None:
        cols = list(colnames_from_df)
    else:
        cols = [f"acct_{k+1}" for k in range(A.shape[1])]

    df = pd.DataFrame(A, index=out_idx, columns=cols)
    if isinstance(contributions, pd.Series):
        df.index.name = contributions.index.name
    return df


def allocate_contributions_proportional(
    contributions: pd.Series,
    weights_by_account: Mapping[str, float],
) -> pd.DataFrame:
    """
    Convenience wrapper for constant allocation proportions over time.

    Builds a constant row-stochastic matrix C of shape (T, K) from
    `weights_by_account` and delegates to `allocate_contributions(...)`.

    Parameters
    ----------
    contributions : pd.Series
        Monthly contributions a[t], length T. The index (ideally a
        monthly DatetimeIndex) is preserved in the output.
    weights_by_account : Mapping[str, float]
        Non-negative weights per account (key → weight). Only strictly
        positive weights are kept; the remaining vector is normalized to
        sum exactly to 1. Column order follows the mapping's insertion
        order.

    Returns
    -------
    pd.DataFrame
        Allocation matrix A of shape (T, K) with columns equal to the
        account names and rows aligned to `contributions.index`.
        By construction, for each t: sum_k A[t, k] == contributions[t].

    Raises
    ------
    ValueError
        If `contributions` is not a non-empty Series, if the mapping is
        empty or if all weights are non-positive.

    Notes
    -----
    - Negative contributions (withdrawals) are split using the same
      proportions.
    - Rows with a[t] == 0 yield all-zeros for that month (as expected).

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> a = pd.Series(np.full(3, 1000.0),
    ...               index=pd.date_range("2025-01-01", periods=3, freq="MS"))
    >>> w = {"housing": 0.1, "emergency": 0.9}
    >>> allocate_contributions_proportional(a, w)
                housing  emergency
    2025-01-01    100.0      900.0
    2025-02-01    100.0      900.0
    2025-03-01    100.0      900.0
    """
    # --- Validate inputs ---
    if not isinstance(contributions, pd.Series) or contributions.empty:
        raise ValueError("contributions must be a non-empty pandas Series.")
    if not weights_by_account:
        raise ValueError("weights_by_account cannot be empty.")

    # Keep strictly positive weights, preserve insertion order
    filtered = [(str(k), float(v)) for k, v in weights_by_account.items() if float(v) > 0.0]
    if not filtered:
        raise ValueError("At least one strictly positive weight is required.")

    names, vals = zip(*filtered)
    w = np.asarray(vals, dtype=float)
    w = w / w.sum()  # make it exactly row-stochastic

    # Build constant C (T, K) and column labels
    T = len(contributions)
    C = np.repeat(w.reshape(1, -1), T, axis=0)
    C_df = pd.DataFrame(C, index=contributions.index, columns=list(names))

    # Delegate to the general allocator.
    # We already normalized rows exactly ⇒ no need to renormalize.
    return allocate_contributions(
        contributions=contributions,
        C=C_df,
        normalize_rows=False,
        rowsum_tol=1e-12,
    )
