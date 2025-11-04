"""General utilities for FinOpt (MVP)

Contents
--------
- Validation helpers
- Rate conversions (annual ↔ monthly, compounded)
- Array/Series helpers (ensure_1d, to_series, index builders)
- Finance helpers (drawdown, CAGR)
- Scenario helpers (rescale_returns, bootstrap_returns, set_random_seed)
- Reporting helpers (summary_metrics)
- Matplotlib formatters (millions_formatter, format_currency)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    # Validation
    "check_non_negative",
    # Rates
    "annual_to_monthly",
    "monthly_to_annual",
    # Arrays / Series / Index
    "ensure_1d",
    "to_series",
    "month_index",
    "align_index_like",
    "normalize_start_month",
    # Finance
    "drawdown",
    "compute_cagr",
    # Scenarios / randomness
    "set_random_seed",
    "rescale_returns",
    "bootstrap_returns",
    # Reporting
    "summary_metrics",
    # Matplotlib formatters
    "millions_formatter",
    "format_currency",
]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_non_negative(name: str, value: float) -> None:
    """Raise if *value* is negative (strict)."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative (got {value}).")


# ---------------------------------------------------------------------------
# Rate conversions (compounded)
# ---------------------------------------------------------------------------

def annual_to_monthly(r_annual: float) -> float:
    """Convert nominal annual rate to equivalent compounded monthly rate.

    Uses: (1 + r_a) ** (1/12) - 1. Accepts negative values as well.
    """
    return float((1.0 + r_annual) ** (1.0 / 12.0) - 1.0)


def monthly_to_annual(r_monthly: float) -> float:
    """Convert nominal monthly rate to equivalent compounded annual rate.

    Uses: (1 + r_m) ** 12 - 1. Accepts negative values as well.
    """
    return float((1.0 + r_monthly) ** 12.0 - 1.0)


# ---------------------------------------------------------------------------
# Array / Series helpers
# ---------------------------------------------------------------------------
ArrayLike = Sequence[float] | np.ndarray | pd.Series


def ensure_1d(a: ArrayLike, *, name: str = "array") -> np.ndarray:
    """Convert input to a 1-D float NumPy array with helpful error messages."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def to_series(a: ArrayLike, index: Optional[pd.Index], *, name: str = "value") -> pd.Series:
    """Return a Pandas Series from array-like *a* with provided index (optional)."""
    arr = ensure_1d(a, name=name)
    if index is None:
        # Build a simple RangeIndex if none is provided
        return pd.Series(arr, name=name)
    if len(index) != arr.shape[0]:
        raise ValueError("index length must match data length.")
    return pd.Series(arr, index=index, name=name)


def month_index(start: Optional[date], months: int) -> pd.DatetimeIndex:
    """Construct a first-of-month DatetimeIndex for *months* periods.

    If *start* is None, uses the current month as the first period.
    """
    if months <= 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if start is None:
        today = pd.Timestamp.today().normalize()
        first = pd.Timestamp(today.year, today.month, 1)
    else:
        first = pd.Timestamp(start.year, start.month, 1)
    return pd.date_range(start=first, periods=months, freq="MS")


def align_index_like(months: int, like: Optional[pd.Index | pd.Series | pd.DataFrame]) -> pd.DatetimeIndex:
    """Infer a DatetimeIndex of length *months* from *like* if possible.

    - If *like* (or its `.index`) is a DatetimeIndex long enough, reuse it.
    - Otherwise, build a default monthly index starting today.
    """
    if isinstance(like, pd.DatetimeIndex) and len(like) >= months:
        return like[:months]
    if like is not None and hasattr(like, "index"):
        idx = getattr(like, "index")
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= months:
            return idx[:months]
    # Fallback
    return month_index(start=None, months=months)


def normalize_start_month(start: Optional[date] | int) -> int:
    """
    Map start date or month to offset 0..11 (Jan=1→0).
    If None, defaults to 0 (January).
    """
    if start is None:
        return 0
    if isinstance(start, date):
        month = start.month
    else:
        month = int(start)
    if not (1 <= month <= 12):
        raise ValueError("start month must be in 1..12")
    return (month - 1) % 12


# ---------------------------------------------------------------------------
# Finance helpers
# ---------------------------------------------------------------------------

def drawdown(series: pd.Series) -> pd.Series:
    """Return drawdown series: (W - cummax(W)) / cummax(W).

    Returns zeros for non-positive running maxima to avoid division by zero.
    """
    if series.empty:
        return series.copy()
    s = series.astype(float)
    running_max = s.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (s - running_max) / running_max
        dd[running_max <= 0] = 0.0
    dd.name = getattr(series, "name", None) or "drawdown"
    return dd


def compute_cagr(wealth: pd.Series, *, periods_per_year: int = 12) -> float:
    """Compute CAGR from a wealth series.

    Uses the first strictly-positive observation as the starting base to
    avoid division by zero when W0 == 0 in contribution-driven processes.
    """
    if not isinstance(wealth, pd.Series) or wealth.empty:
        return 0.0
    W = wealth.astype(float).values
    start_val = next((x for x in W if x > 0), 0.0)
    end_val = float(W[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    years = len(W) / float(periods_per_year)
    return float((end_val / start_val) ** (1.0 / years) - 1.0) if years > 0 else 0.0


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def set_random_seed(seed: Optional[int]) -> None:
    """Set NumPy and Python's `random` seeds for reproducibility (if given)."""
    if seed is None:
        return
    import random

    np.random.seed(int(seed))
    random.seed(int(seed))


def rescale_returns(path: ArrayLike, *, target_mean: float, target_vol: float) -> np.ndarray:
    """Rescale an arithmetic-returns path to match target mean and volatility.

    Useful to normalize historical or simulated returns.
    """
    r = ensure_1d(path, name="returns")
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma == 0:
        # Pure shift to target_mean if zero vol
        return np.full_like(r, fill_value=float(target_mean))
    z = (r - mu) / sigma
    return z * float(target_vol) + float(target_mean)


def bootstrap_returns(history: ArrayLike, months: int, *, seed: Optional[int] = None) -> np.ndarray:
    """Simple IID bootstrap of arithmetic returns from a historical sample."""
    r = ensure_1d(history, name="history")
    if r.size == 0 or months <= 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=int(months))
    return r[idx].astype(float, copy=False)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summary_metrics(results: Mapping[str, object]) -> pd.DataFrame:
    """Build a metrics table from a dict of ScenarioResult-like objects.

    Duck-typing: each value must have `.metrics` with attributes
    (final_wealth, total_contributions, cagr, vol, max_drawdown).
    """
    rows = []
    for name, res in results.items():
        metrics = getattr(res, "metrics", None)
        if metrics is None:
            continue
        rows.append(
            {
                "scenario": name,
                "final_wealth": getattr(metrics, "final_wealth", np.nan),
                "total_contributions": getattr(metrics, "total_contributions", np.nan),
                "cagr": getattr(metrics, "cagr", np.nan),
                "vol": getattr(metrics, "vol", np.nan),
                "max_drawdown": getattr(metrics, "max_drawdown", np.nan),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[
            "final_wealth", "total_contributions", "cagr", "vol", "max_drawdown"
        ])
    df = pd.DataFrame(rows).set_index("scenario").sort_index()
    return df


# ---------------------------------------------------------------------------
# Matplotlib formatters
# ---------------------------------------------------------------------------

def millions_formatter(x, pos):
    """
    Format axis values as millions for matplotlib FuncFormatter.
    
    Converts large monetary values to compact millions notation:
    - 25_000_000 → "25M"
    - 12_500_000 → "12.5M"
    - 0 → "0"
    
    Parameters
    ----------
    x : float
        Value to format (in raw units, e.g., CLP).
    pos : int
        Tick position (unused, required by FuncFormatter signature).
    
    Returns
    -------
    str
        Formatted string with "M" suffix.
    
    Examples
    --------
    >>> from matplotlib.ticker import FuncFormatter
    >>> ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    
    Notes
    -----
    Designed for use with matplotlib's FuncFormatter. For text annotations
    where currency symbols are needed, use format_currency() instead.
    """
    if x == 0:
        return '0'
    val = x / 1e6
    return f'{val:.0f}M' if val == int(val) else f'{val:.1f}M'


def format_currency(value, decimals=1, symbol='$', unit='M'):
    """
    Format currency values for text annotations and labels.
    
    Standardizes monetary display across plots with customizable precision
    and symbols. Default behavior: millions with 1 decimal place.
    
    Parameters
    ----------
    value : float
        Monetary value in raw units (e.g., CLP).
    decimals : int, default 1
        Number of decimal places to display.
    symbol : str, default '$'
        Currency symbol prefix.
    unit : str, default 'M'
        Unit suffix (typically 'M' for millions).
    
    Returns
    -------
    str
        Formatted currency string.
    
    Examples
    --------
    >>> format_currency(25_000_000)
    '$25.0M'
    >>> format_currency(25_000_000, decimals=0)
    '$25M'
    >>> format_currency(5_500_000, decimals=2)
    '$5.50M'
    
    Notes
    -----
    - Always divides by 1e6 (millions scaling)
    - Use in text annotations, legends, or titles where FuncFormatter
      cannot be applied
    - Consistent with millions_formatter() for axis ticks
    - Companion function to millions_formatter() for complete formatting solution
    """
    val = value / 1e6
    if decimals == 0:
        return f'{symbol}{val:.0f}{unit}'
    return f'{symbol}{val:.{decimals}f}{unit}'


# ---------------------------------------------------------------------------
# Return-path generators
# ---------------------------------------------------------------------------

def fixed_rate_path(months: int, r_monthly: float) -> np.ndarray:
    """Return a constant arithmetic monthly return path of length `months`."""
    if months <= 0:
        return np.zeros(0, dtype=float)
    return np.full(int(months), float(r_monthly), dtype=float)


def lognormal_iid(
    months: int,
    *,
    mu: float,
    sigma: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """IID arithmetic returns derived from lognormal gross returns.

    Samples gross returns G_t ~ LogNormal(mu, sigma) and maps to arithmetic:
        r_t = G_t - 1
    This guarantees r_t > -1 (no impossible bankruptcies).

    Note: (mu, sigma) are parameters of the normal in log-gross space,
    not arithmetic mean/volatility.
    """
    if months <= 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(seed)
    gross = rng.lognormal(mean=float(mu), sigma=float(sigma), size=int(months))
    r = gross - 1.0
    return r.astype(float, copy=False)


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
    *,
    contributions: Optional[pd.Series | np.ndarray] = None,
    periods_per_year: int = 12,
) -> PortfolioMetrics:
    """Compute key metrics for a simulated wealth path.

    Metrics
    -------
    - final_wealth: W_T
    - total_contributions: sum(A_t) if provided; otherwise 0
    - cagr: uses utils.compute_cagr (avoids division by 0 when W0=0)
    - vol: standard deviation of approximate monthly returns computed
           from wealth changes (robust to W[t]==0)
    - max_drawdown: min(drawdown(W))

    Note
    ----
    When W includes contributions, 'vol' based on ΔW/W_{t-1} is approximate.
    For pure risk analysis, use exogenous returns r_t instead.
    """
    if not isinstance(wealth, pd.Series) or wealth.empty:
        return PortfolioMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    W = wealth.astype(float)
    final_w = float(W.iloc[-1])

    # Total contributions (if provided)
    if contributions is None:
        total_contrib = 0.0
    else:
        a = np.asarray(contributions, dtype=float)
        total_contrib = float(np.nansum(a))

    # CAGR
    cagr_val = compute_cagr(W, periods_per_year=periods_per_year)

    # Vol of approximate "returns" using relative changes in W
    # Handles zeros by avoiding invalid divisions
    W_prev = W.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pseudo_r = (W - W_prev) / W_prev
    pseudo_r = pseudo_r.replace([np.inf, -np.inf], np.nan).dropna()
    vol_val = float(pseudo_r.std(ddof=1)) if not pseudo_r.empty else 0.0

    # Maximum drawdown
    dd = drawdown(W)
    max_dd = float(dd.min()) if not dd.empty else 0.0

    return PortfolioMetrics(
        final_wealth=final_w,
        total_contributions=total_contrib,
        cagr=cagr_val,
        vol=vol_val,
        max_drawdown=max_dd,
    )