"""Income modeling module for FinOpt

Purpose
-------
Entry point of cash flows in FinOpt. Models where the money comes from
(fixed salary, variable income) and how it evolves over time. Provides
clean monthly series that downstream modules (simulation, investment,
optimization) consume to derive contributions and wealth trajectories.

Key components
--------------
- FixedIncome:
    Deterministic monthly base with optional annual growth (e.g. salary
    with inflation adjustments).
- VariableIncome:
    Monthly base with optional seasonality (12 factors), Gaussian noise,
    and floor/cap guardrails (e.g. freelancing, bonuses).
- IncomeModel:
    Facade that combines multiple streams, projects total monthly income
    as a pandas Series/DataFrame, and derives contribution paths via a
    simple rule:
        a_t = alpha * y_fixed_t + beta * y_variable_t

Design principles
-----------------
- Deterministic by default; any randomness is explicit and reproducible
  via numpy.random.Generator with a fixed seed.
- Calendar-first outputs: returns are indexed by first-of-month
  DatetimeIndex for reporting and alignment with scenarios.
- Separation of concerns: income modeling is distinct from portfolio
  returns (`investment.py`), scenario orchestration (`simulation.py`),
  and optimization (`optimization.py`).
- Reuses helpers from `utils.py` for validation, rate conversions, and
  index construction.

Example
-------
>>> from datetime import date
>>> from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
>>> income = IncomeModel(
...     fixed=FixedIncome(base=1_400_000.0, annual_growth=0.02),
...     variable=VariableIncome(base=200_000.0, sigma=0.1, seed=42),
... )
>>> df = income.project(months=12, start=date(2025, 1, 1), as_dataframe=True)
>>> contrib = income.contributions(12, alpha_fixed=0.35, beta_variable=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Reuse common utilities
from .utils import (
    check_non_negative,
    annual_to_monthly,
    month_index,
)

__all__ = [
    "FixedIncome",
    "VariableIncome",
    "IncomeModel",
    "IncomeMetrics",
]


@dataclass(frozen=True)
class IncomeMetrics:
    months: int
    total_fixed: float
    total_variable: float
    total_income: float
    mean_fixed: float
    mean_variable: float
    mean_total: float
    std_variable: float
    coefvar_variable: float
    fixed_share: float
    variable_share: float
    min_variable: float
    max_variable: float
    pct_variable_below_threshold: float  # in [0,1], or NaN if no threshold


# ---------------------------------------------------------------------------
# Income Streams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedIncome:
    """Deterministic stream of fixed monthly income (e.g., salary).

    Purpose
    -------
    Models a predictable, recurring cash inflow such as a salary or stipend.
    Optionally applies an annual growth rate (e.g., inflation adjustment,
    contractual raises) which is internally converted to an equivalent
    compounded monthly rate.

    Parameters
    ----------
    base : float
        Monthly base income amount at t=0.
    annual_growth : float, default 0.0
        Nominal annual growth rate (e.g., 0.05 for +5% per year). Converted
        internally to a compounded monthly rate:
            m = (1 + annual_growth) ** (1/12) - 1
        The projected series then follows:
            y_t = base * (1 + m)^t
    name : str, optional
        Identifier for labeling outputs (e.g., column name in DataFrame).

    Notes
    -----
    - Income values are always non-negative; validation is enforced at init.
    - The projection is fully deterministic (no randomness).
    - Used inside `IncomeModel` to form part of the total monthly income.

    Example
    -------
    >>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.03)
    >>> fi.project(6)
    array([1400000.0, 1403485.0, 1406988.0, ..., 1421123.0])
    """

    base: float
    annual_growth: float = 0.0
    name: str = "fixed"

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)

    def project(self, months: int) -> np.ndarray:
        """Project monthly fixed income for a given horizon.

        Growth is compounded monthly using the equivalent monthly rate.
        """
        if months <= 0:
            return np.zeros(0, dtype=float)
        m = annual_to_monthly(self.annual_growth)
        # Sequence: base * (1+m) ** t for t=0..months-1
        t = np.arange(months, dtype=float)
        return self.base * np.power(1.0 + m, t)


@dataclass(frozen=True)
class VariableIncome:
    """Variable (irregular) income stream with seasonality and noise.

    Purpose
    -------
    Models cash inflows that fluctuate across months, such as tutoring,
    freelance work, commissions, or bonuses. Captures:
    - Long-term trend via optional annual growth.
    - Intra-year pattern via a 12-month seasonality vector.
    - Random fluctuations via Gaussian noise.
    - Guardrails (floor/cap) to bound extreme values.

    Parameters
    ----------
    base : float
        Baseline monthly income before applying growth, seasonality, and noise.
    seasonality : Optional[Iterable[float]], default None
        Length-12 vector of multiplicative factors (Jan–Dec).
        Example: [1.0, 0.8, 1.2, ...] where 1.0 = neutral.
        If None, no seasonality is applied.
    sigma : float, default 0.0
        Standard deviation of Gaussian noise as a *fraction* of the mean
        for each month. Example: sigma=0.2 ⇒ ±20% variation (approx).
    floor : float, optional
        Lower bound applied after noise. Ensures income does not fall below
        this value. If None, no lower bound is used.
    cap : float, optional
        Upper bound applied after noise. Ensures income does not exceed
        this value. If None, no upper bound is used.
    annual_growth : float, default 0.0
        Nominal annual growth rate applied to the `base` before seasonality.
        Internally converted to compounded monthly rate:
            m = (1 + annual_growth) ** (1/12) - 1
    name : str, default "variable"
        Identifier for labeling outputs (e.g., column name).
    seed : Optional[int]
        Random generator seed for reproducible noise.

    Notes
    -----
    - Validates that `base` and `sigma` are non-negative.
    - Requires `seasonality` length = 12 with all values ≥ 0.
    - If both `floor` and `cap` are given, must satisfy floor ≤ cap.
    - Returns are always non-negative (clamped at zero as last step).
    - The projection is deterministic if `sigma=0`; stochastic otherwise,
      but reproducible when `seed` is provided.

    Example
    -------
    >>> vi = VariableIncome(
    ...     base=200_000.0,
    ...     seasonality=[1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.0, 0.8, 0.9, 1.0, 1.05, 1.15],
    ...     sigma=0.15,
    ...     floor=50_000.0,
    ...     cap=400_000.0,
    ...     annual_growth=0.02,
    ...     seed=42
    ... )
    >>> vi.project(6)
    array([200000., 183000., 224000., 210500., 240300., 230100.])
    """

    base: float
    seasonality: Optional[Iterable[float]] = None
    sigma: float = 0.0
    floor: Optional[float] = None
    cap: Optional[float] = None
    annual_growth: float = 0.0
    name: str = "variable"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)
        check_non_negative("sigma", self.sigma)
        if self.floor is not None and self.cap is not None and self.floor > self.cap:
            raise ValueError("floor cannot be greater than cap.")
        if self.seasonality is not None:
            vals = tuple(float(x) for x in self.seasonality)
            if len(vals) != 12:
                raise ValueError("seasonality must have length 12 for Jan..Dec.")
            if any(v < 0 for v in vals):
                raise ValueError("seasonality factors must be non-negative.")

    @staticmethod
    def _normalize_start_month(start_month: Optional[int]) -> int:
        """Map 1..12 to offset 0..11 (Jan=1→0). None→0 (no shift)."""
        if start_month is None:
            return 0
        if not (1 <= int(start_month) <= 12):
            raise ValueError("start_month must be in 1..12 if provided.")
        return (int(start_month) - 1) % 12

    def project(self, months: int, *, start_month: Optional[int] = None) -> np.ndarray:
        """Project variable income with optional calendar‑aware seasonality.

        Parameters
        ----------
        months : int
            Horizon length.
        start_month : Optional[int], default None
            Calendar month number of the **first** observation (1=Jan,…,12=Dec).
            If provided and `seasonality` is not None, the seasonality vector
            [Jan..Dec] se rota para alinear su primer valor con `start_month`.
            Si es None → no se rota (equivalente a empezar en enero).
        """
        if months <= 0:
            return np.zeros(0, dtype=float)

        rng = np.random.default_rng(self.seed)
        m = annual_to_monthly(self.annual_growth)
        t = np.arange(months, dtype=float)
        base_path = self.base * np.power(1.0 + m, t)

        # Calendar-aware seasonal means
        if self.seasonality is None:
            means = base_path.copy()
        else:
            s = tuple(float(x) for x in self.seasonality)  # length 12, Jan..Dec
            offset = self._normalize_start_month(start_month)  # 0..11
            means = np.empty(months, dtype=float)
            for k in range(months):
                # index rotates over 12 months with calendar offset
                idx = (offset + k) % 12
                means[k] = base_path[k] * s[idx]

        # Add noise (if any)
        if self.sigma == 0.0:
            noisy = means
        else:
            noise = rng.normal(loc=0.0, scale=self.sigma, size=months)
            noisy = means * (1.0 + noise)

        # Guardrails
        if self.floor is not None:
            noisy = np.maximum(noisy, self.floor)
        if self.cap is not None:
            noisy = np.minimum(noisy, self.cap)

        return np.maximum(noisy, 0.0)

# ---------------------------------------------------------------------------
# Income Model (Facade)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncomeModel:
    """Facade that combines fixed and variable income streams.

    Purpose
    -------
    Bridges individual income sources (salary, side jobs) with the rest
    of FinOpt. Produces monthly projections of total income and converts
    them into contribution series used for investment simulations.

    Parameters
    ----------
    fixed : FixedIncome
        Deterministic income stream (e.g., salary) with optional annual growth.
    variable : VariableIncome
        Irregular income stream with optional seasonality, noise, and growth.
    name_fixed : str, default "fixed"
        Label for the fixed-income component in DataFrame outputs.
    name_variable : str, default "variable"
        Label for the variable-income component in DataFrame outputs.

    Methods
    -------
    project(months, start=None, as_dataframe=False)
        Project monthly income for the given horizon.
        - Returns a Series (total only) or a DataFrame
          [fixed, variable, total] if as_dataframe=True.
    contributions_from_proportions(months, alpha_fixed, beta_variable, start=None)
        Compute monthly contributions as a weighted fraction of each stream:
            contrib_t = alpha_fixed * fixed_t + beta_variable * variable_t
        Negative contributions are floored at zero.
    to_dict()
        Serialize parameters into a dictionary (for saving configs).
    from_dict(payload)
        Class method to rebuild an IncomeModel from a dictionary.

    Notes
    -----
    - Deterministic unless `variable.sigma > 0` (noise); reproducible when
      `seed` is provided.
    - `start` sets the DatetimeIndex and, if `variable.seasonality` is provided,
    aligns the seasonality to the calendar (the first simulated month uses the
    factor of `start.month`).
    - Designed to integrate seamlessly with `simulation.py` (scenario
      orchestration) and `investment.py` (capital accumulation).

    Example
    -------
    >>> from datetime import date
    >>> income = IncomeModel(
    ...     fixed=FixedIncome(base=1_400_000.0, annual_growth=0.02),
    ...     variable=VariableIncome(base=200_000.0, sigma=0.1, seed=123),
    ... )
    >>> income.project(months=6, start=date(2025, 1, 1), as_dataframe=True)
             fixed  variable     total
    2025-01  1400000   210000.0  1610000.0
    2025-02  1402323   190000.0  1592323.0
    ...
    >>> contrib = income.contributions_from_proportions(
    ...     months=6, alpha_fixed=0.35, beta_variable=1.0
    ... )
    """

    fixed: FixedIncome
    variable: VariableIncome
    name_fixed: str = field(default="fixed", init=True)
    name_variable: str = field(default="variable", init=True)

    def project(
        self,
        months: int,
        start: Optional[date] = None,
        as_dataframe: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Project incomes for a given number of months.

        Parameters
        ----------
        months : int
            Horizon in months (> 0).
        start : Optional[date]
            If given, used to build a calendar index (1st of each month).
        as_dataframe : bool, default False
            If True, returns a DataFrame with columns [fixed, variable, total].
            If False, returns a Series with the total only.
        """
        idx = month_index(start=start, months=max(months, 0))
        if months <= 0:
            if as_dataframe:
                return pd.DataFrame(columns=[self.name_fixed, self.name_variable, "total"], index=idx)
            return pd.Series(dtype=float, index=idx, name="total")

        start_month = start.month if start is not None else None

        fixed_path = self.fixed.project(months)
        variable_path = self.variable.project(months, start_month=start_month)
        total = fixed_path + variable_path

        if as_dataframe:
            df = pd.DataFrame(
                {
                    self.name_fixed: fixed_path,
                    self.name_variable: variable_path,
                    "total": total,
                },
                index=idx,
            )
            return df
        return pd.Series(total, index=idx, name="total")

    def contributions(
        self,
        months: int,
        alpha_fixed: float,
        beta_variable: float,
        start: Optional[date] = None,
    ) -> pd.Series:
        """Compute monthly contributions given fixed/variable proportions.

        Contribution per month t:
            contrib_t = alpha_fixed * fixed_t + beta_variable * variable_t
        """
        idx = month_index(start=start, months=max(months, 0))
        if months <= 0:
            return pd.Series(dtype=float, index=idx, name="contribution")

        start_month = start.month if start is not None else None

        fixed_path = self.fixed.project(months)
        variable_path = self.variable.project(months, start_month=start_month)
        contrib = alpha_fixed * fixed_path + beta_variable * variable_path
        contrib = np.maximum(contrib, 0.0)
        return pd.Series(contrib, index=idx, name="contribution")
    
    def plot(
        self,
        months: int,
        start: Optional[date] = None,
        kind: str = "lines",            # "lines" | "stacked_area"
        ax=None,
        figsize: tuple = (10, 5),
        title: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        ylabel_left: str = "Fixed/Total (CLP)",
        ylabel_right: str = "Variable (CLP)",
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        dual_axis: str | bool = "auto",  # "auto" | True | False
        dual_axis_ratio: float = 3.0,    # threshold to auto-activate dual axis
    ):
        """
        Plot monthly projections of fixed, variable, and total income.

        Parameters
        ----------
        months : int
            Projection horizon in months (> 0).
        start : Optional[date]
            Start date used to build a calendar index (1st of each month).
        kind : {"lines", "stacked_area"}, default "lines"
            Plot style. Note: dual-axis is only supported for "lines".
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        figsize : tuple, default (10, 5)
            Figure size when `ax` is None.
        title : str, optional
            Plot title.
        legend : bool, default True
            Show legend.
        grid : bool, default True
            Show grid on left axis.
        ylabel_left : str, default "Fixed/Total (CLP)"
            Y-label for left axis (fixed + total).
        ylabel_right : str, default "Variable (CLP)"
            Y-label for right axis (variable).
        save_path : str, optional
            If provided, save the figure to this path (PNG, PDF, etc.).
        return_fig_ax : bool, default False
            If True, return (fig, ax). Otherwise, return nothing.
        dual_axis : {"auto", True, False}, default "auto"
            - "auto": enable twin axis if scales differ by `dual_axis_ratio`.
            - True: force twin axis (left: fixed+total, right: variable).
            - False: single axis (all series in the same scale).
        dual_axis_ratio : float, default 3.0
            If max(left)/max(right) or vice versa exceeds this, auto-activate twin axis.

        Notes
        -----
        - Internally uses `project(..., as_dataframe=True)`.
        - Dual axis is only applied with `kind="lines"` (stacked areas with dual axes
          are visually misleading).

        Examples
        --------
        >>> income.plot(months=24, start=date(2025, 1, 1), kind="lines")
        >>> income.plot(months=24, start=date(2025, 1, 1), kind="lines", dual_axis=True)
        >>> income.plot(months=24, start=date(2025, 1, 1), kind="stacked_area")
        """
        import matplotlib.pyplot as plt  # local import to avoid hard dependency if unused

        # Get DataFrame with [fixed, variable, total]
        df = self.project(months=months, start=start, as_dataframe=True)

        # Handle empty horizon
        if df.shape[0] == 0:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data (months <= 0)", ha="center", va="center", transform=ax.transAxes)
            if return_fig_ax:
                return (ax.figure, ax)
            return

        # Create fig/ax if not provided
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        fixed_col = self.name_fixed
        var_col = self.name_variable

        # Decide dual-axis usage
        use_dual = False
        if kind == "lines":
            if dual_axis == True:
                use_dual = True
            elif dual_axis == "auto":
                left_max = max(float(np.nanmax(df[fixed_col])), float(np.nanmax(df["total"])))
                right_max = float(np.nanmax(df[var_col]))
                # avoid div-by-zero
                if right_max > 0 and left_max > 0:
                    ratio = max(left_max / right_max, right_max / left_max)
                    use_dual = ratio >= dual_axis_ratio
        else:
            # For stacked_area, disallow dual axis to avoid misleading visuals
            if dual_axis is True:
                raise ValueError("Dual axis is only supported for kind='lines'.")

        lines = []
        labels = []

        if kind == "lines" and use_dual:
            # Left axis: fixed + total
            left_line_fixed, = ax.plot(df.index, df[fixed_col], label=fixed_col)
            left_line_total, = ax.plot(df.index, df["total"], label="total")
            ax.set_ylabel(ylabel_left)

            # Right axis: variable
            ax_r = ax.twinx()
            right_line_var, = ax_r.plot(df.index, df[var_col], linestyle="--", label=var_col)
            ax_r.set_ylabel(ylabel_right)

            # Collect handles for a unified legend
            lines.extend([left_line_fixed, left_line_total, right_line_var])
            labels.extend([fixed_col, "total", var_col])

            # Styling
            if grid:
                ax.grid(True, linestyle="--", alpha=0.4)

        elif kind == "lines" and not use_dual:
            l1, = ax.plot(df.index, df[fixed_col], label=fixed_col)
            l2, = ax.plot(df.index, df[var_col], label=var_col)
            l3, = ax.plot(df.index, df["total"], label="total")
            lines.extend([l1, l2, l3])
            labels.extend([fixed_col, var_col, "total"])
            ax.set_ylabel(ylabel_left if ylabel_left else "Amount")
            if grid:
                ax.grid(True, linestyle="--", alpha=0.4)

        elif kind == "stacked_area":
            # Single axis stacked areas: fixed + variable, total as line
            ax.fill_between(df.index, 0.0, df[fixed_col], alpha=0.5, label=fixed_col)
            ax.fill_between(
                df.index,
                df[fixed_col],
                df[fixed_col] + df[var_col],
                alpha=0.5,
                label=var_col,
            )
            ltot, = ax.plot(df.index, df["total"], label="total")
            lines.append(ltot)
            labels.append("total")
            ax.set_ylabel(ylabel_left if ylabel_left else "Amount")
            if grid:
                ax.grid(True, linestyle="--", alpha=0.4)

        else:
            raise ValueError("kind must be 'lines' or 'stacked_area'.")

        # Common decorations
        if title:
            ax.set_title(title)
        ax.set_xlabel("Month")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        if legend:
            # Put a combined legend (works for both single and dual-axis)
            ax.legend(lines, labels, loc="best")

        if save_path:
            (fig or ax.figure).savefig(save_path, bbox_inches="tight", dpi=150)

        if return_fig_ax:
            return (fig or ax.figure, ax)
        
        # --- Add cumulative income annotation ---
        total_fixed = float(df[fixed_col].sum())
        total_variable = float(df[var_col].sum())
        total_income = float(df["total"].sum())

        # Texto con formato CLP (miles separados por puntos)
        def fmt_clp(x):
            return "$" + f"{x:,.0f}".replace(",", ".")

        textstr = (
            f"Total Fixed: {fmt_clp(total_fixed)}\n"
            f"Total Variable: {fmt_clp(total_variable)}\n"
            f"Total Income: {fmt_clp(total_income)}"
        )

        # Cuadro de texto en la esquina superior izquierda
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

    def summary(
        self,
        months: int,
        start: Optional[date] = None,
        round_digits: int = 2,
    ) -> pd.Series:
        """
        Return a compact summary of projected incomes for the horizon.

        Parameters
        ----------
        months : int
            Projection horizon in months (> 0).
        start : Optional[date]
            Start date to align calendar (affects variable seasonality).
        round_digits : int, default 2
            Decimal rounding for the returned Series.

        Returns
        -------
        pd.Series
            Keys: ["months", "total_income", "total_fixed", "total_variable",
                   "mean_total", "mean_fixed", "mean_variable",
                   "fixed_share", "variable_share", "std_variable", "coefvar_variable"].
        """
        df = self.project(months=months, start=start, as_dataframe=True)
        if df.empty:
            return pd.Series(
                {
                    "months": 0,
                    "total_income": 0.0,
                    "total_fixed": 0.0,
                    "total_variable": 0.0,
                    "mean_total": 0.0,
                    "mean_fixed": 0.0,
                    "mean_variable": 0.0,
                    "fixed_share": 0.0,
                    "variable_share": 0.0,
                    "std_variable": 0.0,
                    "coefvar_variable": 0.0,
                }
            )

        fixed_col = self.name_fixed
        var_col = self.name_variable

        total_fixed = float(df[fixed_col].sum())
        total_variable = float(df[var_col].sum())
        total_income = float(df["total"].sum())

        mean_fixed = float(df[fixed_col].mean())
        mean_variable = float(df[var_col].mean())
        mean_total = float(df["total"].mean())

        std_variable = float(df[var_col].std(ddof=1)) if len(df) > 1 else 0.0
        coefvar_variable = (std_variable / mean_variable) if mean_variable > 0 else 0.0

        fixed_share = (total_fixed / total_income) if total_income > 0 else 0.0
        variable_share = (total_variable / total_income) if total_income > 0 else 0.0

        out = pd.Series(
            {
                "months": int(len(df)),
                "total_income": total_income,
                "total_fixed": total_fixed,
                "total_variable": total_variable,
                "mean_total": mean_total,
                "mean_fixed": mean_fixed,
                "mean_variable": mean_variable,
                "fixed_share": fixed_share,
                "variable_share": variable_share,
                "std_variable": std_variable,
                "coefvar_variable": coefvar_variable,
            }
        )
        return out.round(round_digits)

    def income_metrics(
        self,
        months: int,
        start: Optional[date] = None,
        variable_threshold: Optional[float] = None,
    ) -> IncomeMetrics:
        """
        Compute detailed statistical metrics of projected incomes.

        Parameters
        ----------
        months : int
            Projection horizon in months (> 0).
        start : Optional[date]
            Start date to align calendar (affects variable seasonality).
        variable_threshold : Optional[float]
            If provided, compute fraction of months where variable income
            falls below this threshold (e.g., a safety floor).

        Returns
        -------
        IncomeMetrics
            Dataclass with totals, means, dispersion and shares.
        """
        df = self.project(months=months, start=start, as_dataframe=True)
        if df.empty:
            return IncomeMetrics(
                months=0,
                total_fixed=0.0,
                total_variable=0.0,
                total_income=0.0,
                mean_fixed=0.0,
                mean_variable=0.0,
                mean_total=0.0,
                std_variable=0.0,
                coefvar_variable=0.0,
                fixed_share=0.0,
                variable_share=0.0,
                min_variable=0.0,
                max_variable=0.0,
                pct_variable_below_threshold=float("nan"),
            )

        fixed_col = self.name_fixed
        var_col = self.name_variable

        total_fixed = float(df[fixed_col].sum())
        total_variable = float(df[var_col].sum())
        total_income = float(df["total"].sum())

        mean_fixed = float(df[fixed_col].mean())
        mean_variable = float(df[var_col].mean())
        mean_total = float(df["total"].mean())

        std_variable = float(df[var_col].std(ddof=1)) if len(df) > 1 else 0.0
        coefvar_variable = (std_variable / mean_variable) if mean_variable > 0 else 0.0

        fixed_share = (total_fixed / total_income) if total_income > 0 else 0.0
        variable_share = (total_variable / total_income) if total_income > 0 else 0.0

        min_variable = float(df[var_col].min())
        max_variable = float(df[var_col].max())

        if variable_threshold is None:
            pct_below = float("nan")
        else:
            below = (df[var_col] < float(variable_threshold)).sum()
            pct_below = float(below) / float(len(df)) if len(df) > 0 else float("nan")

        return IncomeMetrics(
            months=int(len(df)),
            total_fixed=total_fixed,
            total_variable=total_variable,
            total_income=total_income,
            mean_fixed=mean_fixed,
            mean_variable=mean_variable,
            mean_total=mean_total,
            std_variable=std_variable,
            coefvar_variable=coefvar_variable,
            fixed_share=fixed_share,
            variable_share=variable_share,
            min_variable=min_variable,
            max_variable=max_variable,
            pct_variable_below_threshold=pct_below,
        )

    # -------------------------- Serialization helpers ----------------------
    def to_dict(self) -> dict:
        return {
            "fixed": {
                "base": self.fixed.base,
                "annual_growth": self.fixed.annual_growth,
                "name": self.fixed.name,
            },
            "variable": {
                "base": self.variable.base,
                "seasonality": None if self.variable.seasonality is None else list(self.variable.seasonality),
                "sigma": self.variable.sigma,
                "floor": self.variable.floor,
                "cap": self.variable.cap,
                "annual_growth": self.variable.annual_growth,
                "name": self.variable.name,
                "seed": self.variable.seed,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IncomeModel":
        fx = payload.get("fixed", {})
        vr = payload.get("variable", {})
        fixed = FixedIncome(
            base=float(fx.get("base", 0.0)),
            annual_growth=float(fx.get("annual_growth", 0.0)),
            name=str(fx.get("name", "fixed")),
        )
        variable = VariableIncome(
            base=float(vr.get("base", 0.0)),
            seasonality=vr.get("seasonality"),
            sigma=float(vr.get("sigma", 0.0)),
            floor=vr.get("floor"),
            cap=vr.get("cap"),
            annual_growth=float(vr.get("annual_growth", 0.0)),
            name=str(vr.get("name", "variable")),
            seed=vr.get("seed"),
        )
        return cls(fixed=fixed, variable=variable)
    
# ---------------------------------------------------------------------------
# Quick sanity check block (manual execution only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date

    model = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.4),
        variable=VariableIncome(base=200_000.0, sigma=0.01),
    )

    # Projection (DataFrame with [fixed, variable, total] columns)
    print("=== Projection (6 months) ===")
    df = model.project(months=6, start=date(2025, 9, 1), as_dataframe=True)
    print(df)

    # Contributions using α, β rule
    print("\n=== Contributions (α=0.35, β=1.0) ===")
    contrib = model.contributions(
        months=6, alpha_fixed=0.35, beta_variable=1.0, start=date(2025, 9, 1)
    )
    print(contrib)

    # Compact summary as a Series
    print("\n=== Summary() ===")
    s = model.summary(months=6, start=date(2025, 9, 1), round_digits=2)
    print(s)

    # Detailed metrics (dataclass)
    print("\n=== Income Metrics (variable threshold = 150k) ===")
    im = model.income_metrics(months=6, start=date(2025, 9, 1), variable_threshold=150_000.0)
    print(im)

    # Basic sanity checks
    print("\n=== Sanity checks ===")
    total_from_parts = float(df["fixed"].sum() + df["variable"].sum())
    print("Sum(fixed)+Sum(variable) == Sum(total)?",
          abs(total_from_parts - float(df["total"].sum())) < 1e-6)
    print("fixed_share + variable_share ≈ 1?",
          abs(s["fixed_share"] + s["variable_share"] - 1.0) < 1e-6)

