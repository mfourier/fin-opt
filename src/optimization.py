"""
optimization.py — Min-Time Feasibility with Allocation Matrix C[t,k]

Goal
----
Find the smallest horizon T and a contribution allocation matrix C (T×K)
such that each account/goal k attains its target B_k by its deadline.

Core idea (deterministic returns)
---------------------------------
Wealth dynamics per account k:
    W_{t+1}^{(k)} = (W_t^{(k)} + a_t C_{t,k}) (1 + r_t^{(k)})

With arithmetic returns, terminal wealth is affine in contributions:
    W_T^{(k)} = W_0^{(k)} G_0^{(k)} + sum_{t=0}^{T-1} (a_t C_{t,k}) * H_{t+1}^{(k)},
where:
    G_t^{(k)} = Π_{u=t}^{T-1} (1 + r_u^{(k)}),    G_T^{(k)} = 1
    H_{t+1}^{(k)} = Π_{u=t+1}^{T-1} (1 + r_u^{(k)})    (future value factor)

Hence, for fixed T, feasibility reduces to a Linear Program in the variables
x_{t,k} := a_t C_{t,k} (the peso amount allocated to account k in month t):

Variables: x_{t,k} ≥ 0
Row sums:  sum_k x_{t,k} = a_t,   ∀t
Targets:   sum_t x_{t,k} * H_{t+1}^{(k)} ≥ B_k - W_0^{(k)} G_0^{(k)},   ∀k

We search T = 1..T_max and return the smallest feasible T and a C*.

Backends
--------
- "scipy": LP feasibility via scipy.optimize.linprog (objective 0).
- "greedy": deterministic greedy on marginal FV weights H_{t+1}^{(k)} while
            tracking remaining deficits; no external dependencies.

Outputs
-------
- T_star: smallest feasible horizon
- C_star: DataFrame (T×K) row-stochastic allocation matrix
- A_df:   DataFrame (T×K) allocated contributions (x_{t,k})
- wealth_by_account: DataFrame (T×K [+ "total"]) simulated with simulate_portfolio
- diagnostics (per-account deficit, t_hit, margins)

Notes
-----
- Supports either explicit contributions Series, or an IncomeModel + (alpha, beta).
- Returns may be scalar, length-T vector, (T×K) matrix, or dict per account.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd

# Project imports
from .income import IncomeModel
from .utils import ensure_1d, align_index_like, month_index


ArrayLike = Sequence[float] | np.ndarray | pd.Series
MatrixLike = np.ndarray | pd.DataFrame

__all__ = [
    "MinTimeAllocationInput",
    "MinTimeAllocationResult",
    "min_time_with_allocation",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinTimeAllocationInput:
    # Accounts & targets
    accounts: Sequence[str]                                 # length K
    targets_by_account: Mapping[str, float]                 # B_k per account
    start_wealth_by_account: Union[float, Mapping[str, float]] = 0.0

    # Contributions source (choose one)
    contributions: Optional[pd.Series] = None               # a_t (indexed monthly)
    income_model: Optional[IncomeModel] = None              # alternatively...
    start: Optional[pd.Timestamp] = None                    # ...calendar start
    alpha_fixed: float = 0.35
    beta_variable: float = 1.0

    # Returns per account (deterministic)
    returns_by_account: Union[
        float, ArrayLike, MatrixLike, Mapping[str, Union[float, ArrayLike]]
    ] = 0.0

    # Search settings
    T_max: int = 120                                        # months
    t_min: Union[int, Literal["auto"]] = "auto"             # <-- NUEVO
    backend: str = "scipy"                                  # "scipy" | "greedy"
    seed: Optional[int] = 42                                # for greedy tie-breaks
    equality_rows: bool = True                              # enforce sum_k x_{t,k} == a_t

@dataclass(frozen=True)
class MinTimeAllocationResult:
    T_star: int
    C_star: pd.DataFrame
    A_df: pd.DataFrame
    contributions_total: pd.Series
    returns_by_account: pd.DataFrame
    wealth_by_account: pd.DataFrame
    t_hit_by_account: pd.Series
    margin_at_T: pd.Series   # W_T^{(k)} - B_k
    feasible: bool           # True if feasibility was achieved within T_max


# ---------------------------------------------------------------------------
# Helpers — broadcasting, factors, building specs
# ---------------------------------------------------------------------------

def _as_index(T: int, start: Optional[pd.Timestamp]) -> pd.DatetimeIndex:
    return month_index(start, T)  # first-of-month index

def _broadcast_returns(
    R_spec: Union[float, ArrayLike, MatrixLike, Mapping[str, Union[float, ArrayLike]]],
    T: int,
    accounts: Sequence[str],
) -> pd.DataFrame:
    K = len(accounts)
    # dict{name -> scalar/array}
    if isinstance(R_spec, Mapping):
        cols = []
        for name in accounts:
            v = R_spec[name]
            if np.isscalar(v):
                cols.append(np.full(T, float(v), dtype=float))
            else:
                v1 = ensure_1d(v, name=f"returns[{name}]")
                if v1.shape[0] != T:
                    raise ValueError(f"returns[{name}] length {v1.shape[0]} must equal T={T}.")
                cols.append(v1)
        R = np.column_stack(cols)
        return pd.DataFrame(R, columns=list(accounts), index=_as_index(T, None))
    # DataFrame
    if isinstance(R_spec, pd.DataFrame):
        Rdf = R_spec.copy()
        if Rdf.shape[0] != T or Rdf.shape[1] != K:
            raise ValueError(f"returns DataFrame must be (T,K) = ({T},{K}).")
        Rdf.columns = list(accounts)
        return Rdf
    # ndarray
    if isinstance(R_spec, np.ndarray):
        if R_spec.ndim == 1:
            if R_spec.shape[0] != T:
                raise ValueError(f"returns length {R_spec.shape[0]} must equal T={T}.")
            R = np.repeat(R_spec.reshape(T, 1), K, axis=1)
            return pd.DataFrame(R, columns=list(accounts), index=_as_index(T, None))
        if R_spec.ndim == 2:
            if R_spec.shape != (T, K):
                raise ValueError(f"returns must be shape (T,K)=({T},{K}).")
            return pd.DataFrame(R_spec, columns=list(accounts), index=_as_index(T, None))
        raise ValueError("returns ndarray must be 1-D or 2-D.")
    # scalar or 1-D listlike
    if np.isscalar(R_spec):
        R = np.full((T, K), float(R_spec), dtype=float)
        return pd.DataFrame(R, columns=list(accounts), index=_as_index(T, None))
    r1 = ensure_1d(R_spec, name="returns")
    if r1.shape[0] != T:
        raise ValueError(f"returns length {r1.shape[0]} must equal T={T}.")
    R = np.repeat(r1.reshape(T, 1), K, axis=1)
    return pd.DataFrame(R, columns=list(accounts), index=_as_index(T, None))


def _growth_and_annuity_factors(R_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each account k, compute:
      G0[k] = Π_{u=0}^{T-1} (1+r_{u,k})
      H[t+1, k] = Π_{u=t+1}^{T-1} (1+r_{u,k})   (with H[T, k] = 1)
    Returns:
      G0: shape (K,)
      H:  shape (T+1, K)  (we will use H[1:, :] for the LP)
    """
    R = np.asarray(R_df.values, dtype=float)
    T, K = R.shape
    one_plus = 1.0 + R
    # G0
    G0 = np.prod(one_plus, axis=0).astype(float)  # (K,)
    # H (future value factors)
    H = np.empty((T + 1, K), dtype=float)
    H[T, :] = 1.0
    acc = np.ones(K, dtype=float)
    for t in range(T - 1, -1, -1):
        acc *= one_plus[t, :]
        H[t, :] = acc
    # H[t+1,:] is FV factor from month t to T
    return G0, H


def _get_contributions_series(inp: MinTimeAllocationInput, T: int) -> pd.Series:
    if inp.contributions is not None:
        a = inp.contributions
        if len(a) < T:
            raise ValueError(f"Provided contributions length {len(a)} < T={T}.")
        a = a.iloc[:T].astype(float)
        return a
    if inp.income_model is None or inp.start is None:
        raise ValueError("Provide either `contributions` Series or an IncomeModel + (months, start).")
    # Build from income model for the requested horizon T
    a = inp.income_model.contributions(
        months=T,
        alpha_fixed=inp.alpha_fixed,
        beta_variable=inp.beta_variable,
        start=inp.start,
    )
    return a.astype(float)


def _init_starts(inp: MinTimeAllocationInput) -> np.ndarray:
    K = len(inp.accounts)
    if np.isscalar(inp.start_wealth_by_account):
        return np.full(K, float(inp.start_wealth_by_account), dtype=float)
    arr = np.zeros(K, dtype=float)
    for i, name in enumerate(inp.accounts):
        arr[i] = float(inp.start_wealth_by_account.get(name, 0.0))  # type: ignore
    return arr


def _targets_vector(inp: MinTimeAllocationInput) -> np.ndarray:
    K = len(inp.accounts)
    return np.array([float(inp.targets_by_account[name]) for name in inp.accounts], dtype=float)

def _build_full_paths(inp: MinTimeAllocationInput) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build contributions and returns up to horizon T_max.
    Useful to compute a lower bound t_min 'auto'.
    """
    T = int(inp.T_max)
    # a_full
    if inp.contributions is not None:
        if len(inp.contributions) < T:
            raise ValueError(f"Provided contributions length {len(inp.contributions)} < T_max={T}.")
        a_full = inp.contributions.iloc[:T].astype(float).copy()
    else:
        if inp.income_model is None or inp.start is None:
            raise ValueError("For t_min='auto', need contributions Series or (income_model + start).")
        a_full = inp.income_model.contributions(
            months=T,
            alpha_fixed=inp.alpha_fixed,
            beta_variable=inp.beta_variable,
            start=inp.start,
        ).astype(float)

    if not isinstance(a_full.index, pd.DatetimeIndex):
        a_full.index = month_index(inp.start, T)

    # R_full (T×K)
    R_full = _broadcast_returns(inp.returns_by_account, T, inp.accounts)
    R_full.index = a_full.index
    return a_full, R_full


def _auto_t_min(
    a_full: pd.Series,
    R_full: pd.DataFrame,
    W0: np.ndarray,
    B: np.ndarray,
) -> int:
    """
    Compute t_min (lower bound) such that, for each account k,
    the accumulated 'capacity' up to T in future value (if you dedicated ALL to k)
    is enough to cover the deficit in future value at T.

    Definitions:
      cp_k[T] = Prod_{u=0..T-1} (1 + r_{u,k})       (growth prefix)
      deficit_k(T) = max(0, B_k - W0_k * cp_k[T])   (in future value)
      H_{t+1}^{(k)}(T) = Prod_{u=t+1..T-1} (1 + r_{u,k})
      Cap_k(T) = sum_{t=0..T-1} a_t * H_{t+1}^{(k)}(T)

    Key observation (O(1) update):
      Cap_k(T) = Cap_k(T-1) * (1 + r_{T-1,k}) + a_{T-1}
    """
    T, K = R_full.shape
    one_plus = 1.0 + R_full.to_numpy(dtype=float)  # (T,K)

    # Prefixes cp_k[T] with cumulative product per column
    cp = np.ones((T + 1, K), dtype=float)
    for t in range(1, T + 1):
        cp[t, :] = cp[t - 1, :] * one_plus[t - 1, :]

    # Incremental capacity Cap_k(T)
    cap = np.zeros((T + 1, K), dtype=float)  # cap[0,:] = 0
    a_vals = a_full.to_numpy(dtype=float)
    for t in range(1, T + 1):
        # Cap(T) = Cap(T-1) * (1 + r_{t-1}) + a_{t-1}
        cap[t, :] = cap[t - 1, :] * one_plus[t - 1, :] + a_vals[t - 1]

    # Find the minimum T such that Cap_k(T) >= deficit_k(T) for all k
    for t in range(1, T + 1):
        deficit = np.maximum(B - (W0 * cp[t, :]), 0.0)
        # Note: cap[t,:] is already in "t=0 weight units" transformed to the end (FV at T),
        # because each step multiplies by (1+r) and adds a_{t-1} (equivalent to H=1 in that month).
        if np.all(cap[t, :] + 1e-9 >= deficit):
            return t
    return 1  # if never satisfied, start at 1 (outer loop will find actual feasibility)


# ---------------------------------------------------------------------------
# Feasibility at fixed T — backends
# ---------------------------------------------------------------------------

def _feasible_LP(a: pd.Series, R_df: pd.DataFrame, W0: np.ndarray, B: np.ndarray, equality_rows: bool = True):
    """
    LP feasibility:
        variables x[t,k] >= 0
        Row sums: sum_k x[t,k] = a[t]  (or <= a[t] if equality_rows=False)
        Targets:  sum_t x[t,k] * H[t+1,k] >= B_k - W0_k * G0_k
    Returns (feasible: bool, X: ndarray T×K)
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return (False, None)

    T, K = R_df.shape
    idx = a.index
    G0, H = _growth_and_annuity_factors(R_df)
    rhs = B - (W0 * G0)
    rhs = np.maximum(rhs, 0.0)  # if already above target, require 0 additional FV

    # Flatten variables x[t,k] row-major
    nvar = T * K

    # Objective: 0 (pure feasibility)
    c = np.zeros(nvar, dtype=float)

    # Row-sum equalities/inequalities
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for t in range(T):
        row = np.zeros(nvar, dtype=float)
        row[t * K : (t + 1) * K] = 1.0
        if equality_rows:
            A_eq.append(row)
            b_eq.append(float(a.iloc[t]))
        else:
            A_ub.append(row)
            b_ub.append(float(a.iloc[t]))

    # Target inequalities per account k:
    # sum_t x[t,k] * H[t+1,k] >= rhs[k]  →  -sum_t (...) <= -rhs[k]
    for k in range(K):
        row = np.zeros(nvar, dtype=float)
        for t in range(T):
            row[t * K + k] = -float(H[t + 1, k])
        A_ub.append(row)
        b_ub.append(-float(rhs[k]))

    bounds = [(0.0, None)] * nvar

    res = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if A_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if A_eq else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return (False, None)

    x = res.x.reshape(T, K)
    # If we allowed <= rows, normalize tiny slack into the last positive column for exactness
    if not equality_rows:
        row_sums = x.sum(axis=1)
        target = a.to_numpy(dtype=float)
        diff = target - row_sums
        for t in range(T):
            if abs(diff[t]) > 1e-9 and target[t] > 0:
                k = int(np.argmax(x[t, :]))
                x[t, k] += diff[t]
    return (True, x)


def _feasible_greedy(a: pd.Series, R_df: pd.DataFrame, W0: np.ndarray, B: np.ndarray, seed: Optional[int] = 42):
    """
    Greedy feasibility:
      - Compute FV weights H[t+1,k].
      - Track remaining FV deficits d_k = max(0, B_k - W0_k*G0_k).
      - For each t, allocate a[t] to argmax (H[t+1,k]) among k with d_k > 0,
        but cap by what is still needed in FV space (a[t]*H contributes to d_k).
      - Break ties deterministically (np.argmax) or with RNG if provided.

    Returns (feasible: bool, X: ndarray T×K)
    """
    T, K = R_df.shape
    rng = np.random.default_rng(seed)
    G0, H = _growth_and_annuity_factors(R_df)
    deficits = np.maximum(B - (W0 * G0), 0.0)  # FV units at T
    X = np.zeros((T, K), dtype=float)

    for t in range(T):
        at = float(a.iloc[t])
        if at <= 0.0:
            continue
        # While we have contribution to place and unmet deficits:
        remaining = at
        # Build preference order by H[t+1,k] (higher FV per peso first)
        order = np.argsort(-H[t + 1, :])  # descending
        for k in order:
            if remaining <= 1e-12:
                break
            if deficits[k] <= 1e-12:
                continue
            # Max FV we can contribute to k this month if we give all remaining pesos:
            fv_from_all = remaining * H[t + 1, k]
            if fv_from_all <= deficits[k] + 1e-12:
                # Allocate all remaining to k
                X[t, k] += remaining
                deficits[k] -= fv_from_all
                remaining = 0.0
                break
            else:
                # Allocate just enough to cover k's deficit
                need_pesos = deficits[k] / max(H[t + 1, k], 1e-18)
                give = min(remaining, need_pesos)
                X[t, k] += give
                deficits[k] -= give * H[t + 1, k]
                remaining -= give

        # If still remaining (all deficits ~0), dump to best H (doesn't harm feasibility)
        if remaining > 1e-12:
            k_best = int(np.argmax(H[t + 1, :]))
            X[t, k_best] += remaining

        # Early exit if all deficits are cleared
        if np.all(deficits <= 1e-9):
            # Fill any subsequent rows (t+1..T-1) with zeros (already zeros)
            break

    feasible = bool(np.all(deficits <= 1e-6))
    return (feasible, X)


# ---------------------------------------------------------------------------
# Public solver — outer loop over T
# ---------------------------------------------------------------------------

def min_time_with_allocation(inp: MinTimeAllocationInput) -> MinTimeAllocationResult:
    """Min-Time Feasibility with Allocation Matrix C[t,k]
    Goal
    ----
    Find the smallest horizon T and a contribution allocation matrix C (T×K)
    such that each account/goal k attains its target B_k by its deadline.

    Core idea (deterministic returns)
    ---------------------------------
    Wealth dynamics per account k:
        W_{t+1}^{(k)} = (W_t^{(k)} + a_t C_{t,k}) (1 + r_t^{(k)})

    With arithmetic returns, terminal wealth is affine in contributions:
        W_T^{(k)} = W_0^{(k)} G_0^{(k)} + sum_{t=0}^{T-1} (a_t C_{t,k}) * H_{t+1}^{(k)},
    where:
        G_t^{(k)} = Π_{u=t}^{T-1} (1 + r_u^{(k)}),    G_T^{(k)} = 1
        H_{t+1}^{(k)} = Π_{u=t+1}^{T-1} (1 + r_u^{(k)})    (future value factor)

    Hence, for fixed T, feasibility reduces to a Linear Program in the variables
    x_{t,k} := a_t C_{t,k} (the peso amount allocated to account k in month t):

    Variables: x_{t,k} ≥ 0
    Row sums:  sum_k x_{t,k} = a_t,   ∀t
    Targets:   sum_t x_{t,k} * H_{t+1}^{(k)} ≥ B_k - W_0^{(k)} G_0^{(k)},   ∀k

    We search T = 1..T_max and return the smallest feasible T and a C*.

    Backends
    --------
    - "scipy": LP feasibility via scipy.optimize.linprog (objective 0).
    - "greedy": deterministic greedy on marginal FV weights H_{t+1}^{(k)} while
                tracking remaining deficits; no external dependencies.

    Outputs
    -------
    - T_star: smallest feasible horizon
    - C_star: DataFrame (T×K) row-stochastic allocation matrix
    - A_df:   DataFrame (T×K) allocated contributions (x_{t,k})
    - wealth_by_account: DataFrame (T×K [+ "total"]) simulated with simulate_portfolio
    - diagnostics (per-account deficit, t_hit, margins)

    Notes
    -----
    - Supports either explicit contributions Series, or an IncomeModel + (alpha, beta).
    - Returns may be scalar, length-T vector, (T×K) matrix, or dict per account.
    """
    accounts = list(inp.accounts)
    K = len(accounts)

    # We will reuse the calendar from contributions if provided; otherwise build per T.
    W0 = _init_starts(inp)
    B = _targets_vector(inp)
    if inp.t_min == "auto":
            a_full, R_full = _build_full_paths(inp)
            T_start = max(1, _auto_t_min(a_full, R_full, W0, B))
    else:
        T_start = max(1, int(inp.t_min))

    best = None  # (T, A (T×K), a_series, R_df_T)
    for T in range(T_start, int(inp.T_max) + 1):
        # Build contributions and returns for this T
        a = _get_contributions_series(inp, T)  # Series length T
        # Ensure monthly DatetimeIndex
        if not isinstance(a.index, pd.DatetimeIndex):
            a.index = _as_index(T, inp.start)

        R_df = _broadcast_returns(inp.returns_by_account, T, accounts)
        R_df.index = a.index  # align calendar

        # Backend feasibility at fixed T
        if inp.backend == "scipy":
            ok, X = _feasible_LP(a, R_df, W0, B, equality_rows=inp.equality_rows)
            if not ok:
                ok, X = _feasible_greedy(a, R_df, W0, B, seed=inp.seed)
        elif inp.backend == "greedy":
            ok, X = _feasible_greedy(a, R_df, W0, B, seed=inp.seed)
        else:
            raise ValueError("backend must be 'scipy' or 'greedy'.")

        if ok:
            best = (T, X, a, R_df)
            break

    if best is None:
        # Not feasible within T_max — return informative artifact at T_max using greedy allocation
        T = int(inp.T_max)
        a = _get_contributions_series(inp, T)
        if not isinstance(a.index, pd.DatetimeIndex):
            a.index = _as_index(T, inp.start)
        R_df = _broadcast_returns(inp.returns_by_account, T, accounts)
        R_df.index = a.index
        _, X = _feasible_greedy(a, R_df, W0, B, seed=inp.seed)
        feas = False
    else:
        T, X, a, R_df = best
        feas = True

    # Build DataFrames
    A_df = pd.DataFrame(X, index=a.index, columns=accounts)                # pesos allocated
    # Guard: if a[t] == 0, leave C[t,*] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        C_arr = np.where(a.values.reshape(-1, 1) > 0, X / a.values.reshape(-1, 1), 0.0)
    C_df = pd.DataFrame(C_arr, index=a.index, columns=accounts)

    # Simulate wealth-by-account for diagnostics and plots
    wealth_df = simulate_portfolio(
        contributions_matrix=A_df,
        returns_matrix=R_df,
        start_values=_init_starts(inp),
        index_like=a.index,
        column_names=accounts,
        include_total_col=True,
    )

    # Compute t_hit per account and margins
    t_hit = []
    margins = []
    for k, name in enumerate(accounts):
        target = float(inp.targets_by_account[name])
        serie = wealth_df[name]
        # first month where wealth >= target
        where = np.where(serie.values >= target)[0]
        hit = int(where[0]) if where.size > 0 else None
        t_hit.append(hit if hit is not None else -1)
        margins.append(float(serie.iloc[-1]) - target)
    t_hit_ser = pd.Series(t_hit, index=accounts, name="t_hit_index")
    margin_ser = pd.Series(margins, index=accounts, name="margin_at_T")

    return MinTimeAllocationResult(
        T_star=T,
        C_star=C_df,
        A_df=A_df,
        contributions_total=a,
        returns_by_account=R_df,
        wealth_by_account=wealth_df,
        t_hit_by_account=t_hit_ser,
        margin_at_T=margin_ser,
        feasible=feas,
    )

# ===========================================================================
# Manual quick test (integration smoke tests)
# ===========================================================================
if __name__ == "__main__":
    from datetime import date
    import numpy as np
    import pandas as pd

    from .income import FixedIncome, VariableIncome, IncomeModel
    from .investment import simulate_portfolio
    from .goals import Goal
    from .optimization import MinTimeAllocationInput, min_time_with_allocation

    # Example: 2 goals (housing + emergency), 24 months horizon
    accounts = ["housing", "emergency"]
    income = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
        variable=VariableIncome(base=200_000.0, sigma=0.00),
    )

    inp = MinTimeAllocationInput(
        accounts=accounts,
        targets_by_account={"housing": 20_000_000.0, "emergency": 6_000_000.0},
        start_wealth_by_account=0.0,
        income_model=income,
        start=date(2025, 9, 1),
        alpha_fixed=0.35,
        beta_variable=1.0,
        returns_by_account={"housing": 0.004, "emergency": 0.002},
        T_max=36,
        t_min="auto",        # <-- test new feature
        backend="greedy",    # try greedy backend
    )

    res = min_time_with_allocation(inp)
    print("[MinTimeAllocation] T_star:", res.T_star)
    print("[MinTimeAllocation] feasible:", res.feasible)
    print("[MinTimeAllocation] margins:")
    print(res.margin_at_T)
    print(res.C_star.head())
