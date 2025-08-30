"""
Income modeling module for FinOpt.

Purpose
-------
Entry point for modeling cash flows in FinOpt. Captures where the money comes from
(fixed salary, variable income) and how it evolves over time. Produces clean,
monthly series that downstream modules (simulation, investment, optimization)
consume to derive contributions and wealth trajectories.

Key components
--------------
- FixedIncome:
    Deterministic monthly income with optional annual growth (e.g., salary
    with inflation or contractual raises). Produces a predictable, non-negative
    series via compounded monthly growth.

- VariableIncome:
    Monthly income with optional seasonality (12-month factors), Gaussian noise,
    floor/cap guardrails, and optional annual growth (e.g., freelancing, bonuses).
    Allows alignment to calendar months and reproducible randomness via a fixed seed.

- IncomeModel:
    Facade that combines fixed and variable income streams. Projects total
    monthly income as a pandas Series or DataFrame, and computes contribution
    paths as weighted fractions of each stream. Default weights are 30% of fixed
    income and 100% of variable income, adjustable via `monthly_contribution`.

Design principles
-----------------
- Deterministic by default; randomness is explicit and reproducible via
  numpy.random.Generator with a fixed seed.
- Calendar-aware outputs: Series/DataFrames indexed by first-of-month
  for reporting and alignment with scenarios.
- Separation of concerns: income modeling is distinct from portfolio
  returns (`investment.py`), scenario orchestration (`simulation.py`),
  and optimization (`optimization.py`).
- Reuses helpers from `utils.py` for validation, rate conversions, and
  index construction.

Example
-------
>>> from datetime import date
>>> from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
>>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.02)
>>> vi = VariableIncome(base=200_000.0, sigma=0.0)  # deterministic example
>>> income = IncomeModel(fixed=fi, variable=vi)
>>> df = income.project(months=12, start=date(2025, 1, 1), as_dataframe=True)
>>> contrib = income.contributions(months=12)
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
    normalize_start_month
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
    """
    Deterministic monthly income stream with optional annual growth and salary raises.

    Purpose
    -------
    Models predictable, recurring cash inflows such as salaries, stipends,
    or pensions. Optionally applies an annual growth rate, which is internally
    converted to a compounded monthly rate, producing a deterministic projection.
    
    Supports scheduled salary raises at specific dates that are automatically
    converted to month offsets based on the projection start date.

    Parameters
    ----------
    base : float
        Monthly base income amount at t=0. Must be non-negative.
    annual_growth : float, default 0.0
        Nominal annual growth rate (e.g., 0.05 for +5% per year). Internally
        converted to a monthly compounded rate:
            m = (1 + annual_growth) ** (1/12) - 1
        The projected income series follows:
            y_t = base * (1 + m)^t
    salary_raises : Optional[Dict[date, float]], default None
        Dictionary mapping raise dates to absolute raise amounts.
        Raises are applied from the first month that includes the specified date.
        Example: {date(2025, 7, 1): 200_000, date(2025, 12, 15): 150_000}
    name : str, default "fixed"
        Identifier for labeling outputs (e.g., column name in DataFrames).

    Methods
    -------
    project(months, start=None)
        Returns a deterministic monthly income series for the given horizon.

    Notes
    -----
    - Income values are always non-negative; enforced during initialization.
    - The projection is fully deterministic (no stochasticity).
    - Salary raises are applied based on calendar dates relative to start date.
    - Can be combined with variable income streams in `IncomeModel`.

    Example
    -------
    >>> from datetime import date
    >>> fi = FixedIncome(
    ...     base=1_400_000.0, 
    ...     annual_growth=0.03,
    ...     salary_raises={date(2025, 7, 1): 200_000}
    ... )
    >>> fi.project(12, start=date(2025, 1, 1))
    array([1400000.0, 1403485.0, ..., 1620501.0, ...])  # raise applied from month 6
    """

    base: float
    annual_growth: float = 0.0
    salary_raises: Optional[Dict[date, float]] = None
    name: str = "fixed"

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)

    def project(self, months: int, start: Optional[date] = None) -> np.ndarray:
        """
        Deterministic monthly income stream with optional annual growth 
        and scheduled salary raises.

        Purpose
        -------
        Models predictable, recurring cash inflows such as salaries, stipends,
        or pensions. Supports two mechanisms for income evolution:

        1. **Annual growth**: a fixed nominal rate converted to compounded 
        monthly growth.
        2. **Salary raises**: absolute increases at specific calendar dates,
        applied permanently from the corresponding month onward.

        Parameters
        ----------
        base : float
            Monthly base income amount at t=0. Must be non-negative.
        annual_growth : float, default 0.0
            Nominal annual growth rate (e.g., 0.05 for +5% per year). Internally
            converted to a monthly compounded rate:
                m = (1 + annual_growth) ** (1/12) - 1
            The projected income series follows:
                y_t = base * (1 + m)^t
            unless modified by salary raises.
        salary_raises : Optional[Dict[date, float]], default None
            Dictionary mapping raise dates to absolute raise amounts.
            Each raise increases the current salary level permanently,
            with future growth compounding on the new base.
            Example: {date(2025, 7, 1): 200_000, date(2025, 12, 15): 150_000}
        name : str, default "fixed"
            Identifier for labeling outputs (e.g., column name in DataFrames).

        Methods
        -------
        project(months, start=None)
            Returns a deterministic monthly income series for the given horizon.

        Notes
        -----
        - Income values are always non-negative; enforced during initialization.
        - The projection is fully deterministic (no stochasticity).
        - Salary raises are triggered from the month containing the specified date.
        - Growth is applied multiplicatively per month, compounding on the
        updated base (after raises).
        - Can be combined with variable income streams in `IncomeModel`.

        Example
        -------
        >>> from datetime import date
        >>> fi = FixedIncome(
        ...     base=1_400_000.0,
        ...     annual_growth=0.03,
        ...     salary_raises={date(2025, 7, 1): 200_000}
        ... )
        >>> fi.project(12, start=date(2025, 1, 1))
        array([1400000.0, 1403485.0, ..., 1605021.0, ...])  
        # raise applied starting July 2025
    """
        if months <= 0:
            return np.zeros(0, dtype=float)

        # If no raises, use simple projection
        if not self.salary_raises:
            m = annual_to_monthly(self.annual_growth)
            t = np.arange(months, dtype=float)
            return self.base * np.power(1.0 + m, t)

        # With raises, we need start date and month-by-month calculation
        if start is None:
            raise ValueError("start date is required when salary_raises are specified")

        # Convert raises to month offsets
        raise_schedule = {}
        for raise_date, raise_amount in self.salary_raises.items():
            raise_month = self._date_to_month_offset(start, raise_date)
            if 0 <= raise_month < months:
                raise_schedule[raise_month] = raise_amount

        # Month-by-month calculation
        monthly_rate = annual_to_monthly(self.annual_growth)
        income_path = np.zeros(months, dtype=float)
        current_salary = self.base

        for month in range(months):
            # Apply raise if scheduled for this month
            if month in raise_schedule:
                current_salary += raise_schedule[month]
            
            # Set income for this month
            income_path[month] = current_salary
            
            # Apply monthly growth for next month
            if month < months - 1:  # Don't grow after the last month
                current_salary *= (1.0 + monthly_rate)

        return income_path

    def _date_to_month_offset(self, start_date: date, target_date: date) -> int:
        """
        Calculate month offset between two dates.
        
        Parameters
        ----------
        start_date : date
            Reference start date (month 0).
        target_date : date
            Target date for the raise.
            
        Returns
        -------
        int
            Month offset (0-indexed). Can be negative if target_date < start_date.
        """
        year_diff = target_date.year - start_date.year
        month_diff = target_date.month - start_date.month
        return year_diff * 12 + month_diff


@dataclass(frozen=True)
class VariableIncome:
    """
    Variable monthly income stream with optional seasonality, noise, and growth.

    Purpose
    -------
    Models irregular cash inflows that fluctuate across months, such as tutoring,
    freelance work, commissions, or bonuses. The model captures:

    - Long-term trend via optional annual growth.
    - Intra-year seasonality through a 12-month multiplicative vector.
    - Random fluctuations using Gaussian noise.
    - Floor and cap guardrails to bound extreme values.
    - Non-negativity of resulting projected income.

    Parameters
    ----------
    base : float
        Baseline monthly income before applying growth, seasonality, and noise.
    seasonality : Optional[Iterable[float]], default None
        Length-12 vector of multiplicative factors (Jan–Dec). Values must be non-negative.
        Example: [1.0, 0.9, 1.1, ...], where 1.0 = neutral. If None, no seasonality is applied.
    sigma : float, default 0.0
        Standard deviation of Gaussian noise as a fraction of the mean for each month.
        Example: sigma=0.2 ⇒ ±20% variation approximately.
    floor : Optional[float], default None
        Minimum income after noise. Ensures income does not fall below this value.
    cap : Optional[float], default None
        Maximum income after noise. Ensures income does not exceed this value.
    annual_growth : float, default 0.0
        Nominal annual growth rate applied to `base` before seasonality.
        Converted internally to a compounded monthly rate:
            m = (1 + annual_growth) ** (1/12) - 1
    name : str, default "variable"
        Identifier for labeling outputs (e.g., column name in DataFrames).
    seed : Optional[int]
        Random seed for reproducible noise.

    Methods
    -------
    project(months, start=None, seed=None)
        Returns a projected income array for the specified number of months,
        optionally aligned to a start month and including seasonality and noise.

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
    
    def project(self, months: int, *,  start: Optional[date] | int = None, seed: int = None) -> np.ndarray:
        """
        Project a variable monthly income series over a given horizon.

        Generates a projected income array incorporating optional annual growth,
        seasonality, Gaussian noise, and guardrails (floor/cap). The series can
        be aligned to any starting calendar month.

        Parameters
        ----------
        months : int
            Number of months to project. Must be non-negative.
        start : Optional[date or int], default None
            Starting date or month for the projection. If date, uses start.month.
            If int, must be 1..12. If None, January is assumed.
        seed : Optional[int], default None
            Random seed to control Gaussian noise for reproducibility. If None,
            a non-deterministic random sequence is used.

        Returns
        -------
        np.ndarray
            Array of length `months` containing the projected income for each month.
            Values respect the specified floor, cap, and non-negativity constraints.

        Notes
        -----
        - If `months <= 0`, returns an empty array.
        - The base income is adjusted for annual growth using a compounded monthly rate.
        - Gaussian noise is applied multiplicatively to simulate realistic income fluctuations.
        - Seasonality, if provided, is applied multiplicatively and rotates according to `start_month`.
        """

        if months <= 0:
            return np.zeros(0, dtype=float)

        rng = np.random.default_rng(seed)
        m = annual_to_monthly(self.annual_growth)
        t = np.arange(months, dtype=float)
        base_path = self.base * np.power(1.0 + m, t)

        # Calendar-aware seasonal means
        if self.seasonality is None:
            means = base_path.copy()
        else:
            s = tuple(float(x) for x in self.seasonality)
            offset = normalize_start_month(start)
            means = np.empty(months, dtype=float)
            for k in range(months):
                idx = (offset + k) % 12
                means[k] = base_path[k] * s[idx]

        # Add noise
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


@dataclass(frozen=False)
class IncomeModel:
    """
    Unified monthly income projection combining fixed and variable streams.

    Description
    -----------
    This class integrates deterministic (fixed) and irregular (variable) income sources
    into a single framework. It provides:

    - Monthly projections of total income.
    - Monthly contribution estimates for investment or savings.
    - Calendar-aware handling of seasonal effects and contribution schedules.

    The design ensures consistency with financial modeling practices, allowing
    flexible specification of contribution rates and alignment with any starting month.

    Parameters
    ----------
    fixed : FixedIncome
        Deterministic income stream (e.g., salary) with optional annual growth.
    variable : VariableIncome
        Irregular income stream, capturing seasonality, stochastic noise, and growth.
    name_fixed : str, default "fixed"
        Column name or label for the fixed-income component in outputs.
    name_variable : str, default "variable"
        Column name or label for the variable-income component in outputs.
    monthly_contribution : Optional[dict], default None
        Dictionary specifying **annual contribution fractions per stream**:
        {"fixed": array_like[12], "variable": array_like[12]}.
        Each array corresponds to January–December fractions of income to contribute.
        If None, default values are applied:
            - fixed: 30% of fixed income per month
            - variable: 100% of variable income per month
        Fractions are rotated based on the starting month (`start`) and repeated cyclically
        to cover any projection horizon.

    Methods
    -------
    project(months, start=None, as_dataframe=False)
        Returns projected monthly income over the given horizon, optionally as a DataFrame.
    contributions(months, start=None)
        Returns projected monthly contributions by applying the contribution fractions
        to the fixed and variable income streams.
    to_dict()
        Serializes the model configuration and parameters into a dictionary.
    from_dict(payload)
        Reconstructs an `IncomeModel` instance from a dictionary payload.

    Notes
    -----
    - The `start` parameter defines the calendar alignment for both income projections
      and rotation of contribution fractions. For `VariableIncome`, it sets the
      first month for seasonality.
    - Contribution fractions are automatically rotated according to the starting month
      and repeated every 12 months to cover longer horizons.
    - Contributions are floored at zero; negative values are never returned.
    - Both `project` and `contributions` handle `months <= 0` gracefully,
      returning empty Series or DataFrames with proper indices.
    - The class supports deterministic fixed incomes and stochastic variable incomes
      with optional seasonality and noise.

    Example
    -------
    >>> from datetime import date
    >>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.02)
    >>> vi = VariableIncome(base=200_000.0, sigma=0.1, seed=123)
    >>> income_model = IncomeModel(fixed=fi, variable=vi)
    >>> df = income_model.project(months=6, start=date(2025,1,1), as_dataframe=True)
    >>> df
               fixed   variable      total
    2025-01  1400000  210000.0  1610000.0
    2025-02  1402323  190000.0  1592323.0
    >>> contrib = income_model.contributions(months=6)
    >>> contrib
    2025-01    602000.0
    2025-02    547697.0
    ...
    """
    
    fixed: FixedIncome
    variable: VariableIncome
    name_fixed: str = field(default="fixed", init=True)
    name_variable: str = field(default="variable", init=True)
    monthly_contribution: dict = None 

    def project(
        self,
        months: int,
        start: Optional[date] = None,
        as_dataframe: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Project combined fixed and variable income streams over a specified horizon.

        Generates monthly income projections by summing the deterministic fixed
        income (`FixedIncome`) and the variable income (`VariableIncome`) streams.
        Optionally aligns the projection with a calendar index starting from a given
        date and returns either a Series of total income or a DataFrame with
        individual components.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0. If `months <= 0`, an empty
            Series or DataFrame is returned with appropriate calendar index.
        start : Optional[date], default None
            If provided, used to build a calendar index (1st of each month). Also
            determines the starting month for seasonality in `VariableIncome`.
            If None, defaults to month 1 (January) for seasonality alignment.
        as_dataframe : bool, default False
            If True, returns a DataFrame with columns:
                - fixed: projected fixed income
                - variable: projected variable income
                - total: sum of fixed and variable incomes
            If False, returns a Series containing the total income only.

        Returns
        -------
        pd.Series or pd.DataFrame
            - Series of length `months` with total projected income if `as_dataframe=False`.
            - DataFrame with columns [fixed, variable, total] if `as_dataframe=True`.
            - Index is a monthly calendar based on `start` if provided, else integer range.

        Notes
        -----
        - Negative or zero `months` return empty structures with correct indices.
        - For `VariableIncome`, the starting month (`start.month`) is used to
        rotate seasonality factors.
        - Projection is deterministic for the fixed component, stochastic for
        the variable component if its `sigma` > 0.
        - Useful for calculating contributions, investment planning, and
        combining multiple income sources in simulations.

        Examples
        --------
        >>> from datetime import date
        >>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.02)
        >>> vi = VariableIncome(base=200_000.0, sigma=0.1, seed=123)
        >>> income_model = IncomeModel(fixed=fi, variable=vi)
        >>> income_model.project(months=6, start=date(2025, 1, 1), as_dataframe=True)
                fixed   variable      total
        2025-01  1400000  210000.0  1610000.0
        2025-02  1402323  190000.0  1592323.0
        >>> income_model.project(months=6, as_dataframe=False)
        0    1610000.0
        1    1592323.0
        2    ...
        Name: total, dtype: float64
        """
        idx = month_index(start=start, months=max(months, 0))
        if months <= 0:
            if as_dataframe:
                return pd.DataFrame(columns=[self.name_fixed, self.name_variable, "total"], index=idx)
            return pd.Series(dtype=float, index=idx, name="total")

        fixed_path = self.fixed.project(months, start=start)
        variable_path = self.variable.project(months, start=start)
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
        start: Optional[date] = None,
    ) -> pd.Series:
        """
        Compute monthly contributions from fixed and variable income streams.

        Description
        -----------
        Calculates monthly contributions by applying **monthly fractional rates** to the 
        projected fixed and variable incomes. Contribution fractions are specified
        annually (12 months) and are rotated based on the starting month to align with 
        the calendar. If no fractions are provided, default values are used:
            - fixed income: 30% each month
            - variable income: 100% each month

        Contributions are floored at zero, ensuring non-negative values, and can be
        used for investment planning, savings projections, or cash-flow simulations.

        Parameters
        ----------
        months : int
            Number of months to compute contributions. Must be >= 0.
            If `months <= 0`, returns an empty Series with the correct index.
        start : Optional[date], default None
            Calendar start date for contributions. Determines:
            - The first month for seasonality alignment of `VariableIncome`.
            - The rotation of the 12-month contribution fractions.
            If None, January (month 1) is assumed.

        Returns
        -------
        pd.Series
            Series of length `months` with computed contributions per month.
            Index is a monthly calendar based on `start` if provided; otherwise,
            an integer range. Contributions are non-negative.

        Notes
        -----
        - Monthly contribution for month `t` is computed as:
            contrib_t = fixed_fraction_t * fixed_income_t
                        + variable_fraction_t * variable_income_t
        where fractions rotate annually based on `start`.
        - If `self.monthly_contribution` is None, defaults (30% fixed, 100% variable)
        are used.
        - Both fixed and variable incomes are projected using their respective `project` methods.
        - Negative contributions are automatically floored to zero.

        Raises
        ------
        ValueError
            If `self.monthly_contribution` is provided but either the fixed or variable
            fraction arrays do not have length 12.

        Examples
        --------
        >>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.02)
        >>> vi = VariableIncome(base=200_000.0, sigma=0.0)
        >>> income_model = IncomeModel(fixed=fi, variable=vi)
        >>> income_model.contributions(months=6, start=date(2025,1,1))
        2025-01    620000.0
        2025-02    624200.0
        2025-03    628400.0
        2025-04    632600.0
        2025-05    636800.0
        2025-06    641000.0
        Name: contribution, dtype: float64
        """
        idx = month_index(start=start, months=max(months, 0))
        if months <= 0:
            return pd.Series(dtype=float, index=idx, name="contribution")

        # Project incomes
        fixed_path = self.fixed.project(months, start=start)
        variable_path = self.variable.project(months, start=start)

        # Initialize contribution fractions (12-month lists)
        if self.monthly_contribution is None:
            fixed_fractions = np.full(12, 0.3, dtype=float)
            variable_fractions = np.ones(12, dtype=float)
        else:
            fixed_fractions = np.asarray(self.monthly_contribution["fixed"], dtype=float)
            variable_fractions = np.asarray(self.monthly_contribution["variable"], dtype=float)
            if len(fixed_fractions) != 12 or len(variable_fractions) != 12:
                raise ValueError("monthly_contribution lists must have length 12.")

        # Rotate fractions according to start and repeat cyclically
        offset = normalize_start_month(start)
        fixed_fractions_full = np.array([fixed_fractions[(offset + k) % 12] for k in range(months)])
        variable_fractions_full = np.array([variable_fractions[(offset + k) % 12] for k in range(months)])

        # Compute contributions
        contrib = fixed_fractions_full * fixed_path + variable_fractions_full * variable_path
        contrib = np.maximum(contrib, 0.0)

        return pd.Series(contrib, index=idx, name="contribution")

    def plot_income(
        self,
        months: int,
        start: Optional[date] = None,
        ax=None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        ylabel_left: str = "Fixed/Total (CLP)",
        ylabel_right: str = "Variable (CLP)",
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        dual_axis: str | bool = "auto",
        dual_axis_ratio: float = 3.0,
        show_confidence_band: bool = True,
        confidence: float = 0.9,
        n_simulations: int = 500,
        colors: dict | None = None,  # e.g., {"fixed": "blue", "variable": "orange", "total": "black"}
    ):
        """
        Plot monthly projections of fixed, variable, and total income.

        Supports stochastic variable income with optional confidence bands and automatic dual-axis scaling.

        Parameters
        ----------
        months : int
            Projection horizon in months (> 0).
        start : date, optional
            Start date for projection to align calendar months.
        ax : matplotlib.axes.Axes, optional
            Existing Axes object to plot on. If None, a new figure and axes are created.
        figsize : tuple, default (12, 6)
            Figure size (width, height) in inches if a new figure is created.
        title : str, optional
            Plot title.
        legend : bool, default True
            Whether to display legend.
        grid : bool, default True
            Whether to show grid.
        ylabel_left : str, default "Fixed/Total (CLP)"
            Label for left y-axis.
        ylabel_right : str, default "Variable (CLP)"
            Label for right y-axis when dual-axis is used.
        save_path : str, optional
            File path to save the figure. If None, figure is not saved.
        return_fig_ax : bool, default False
            If True, returns a tuple (fig, ax) for further customization.
        dual_axis : {"auto", True, False}, default "auto"
            Controls whether to use a second y-axis for variable income.
            "auto": activate if scales differ significantly (controlled by dual_axis_ratio).
        dual_axis_ratio : float, default 3.0
            Threshold ratio between left and right axis maxima to auto-activate dual-axis.
        show_confidence_band : bool, default True
            Whether to display confidence intervals for stochastic variable income.
        confidence : float, default 0.9
            Confidence level for the bands (only used if show_confidence_band=True).
        n_simulations : int, default 500
            Number of stochastic simulations to compute confidence bands.
        colors : dict, optional
            Mapping of line colors. Keys: "fixed", "variable", "total". Defaults to
            {"fixed": "blue", "variable": "orange", "total": "black"}.

        Returns
        -------
        None or tuple
            If return_fig_ax=True, returns (fig, ax). Otherwise, returns None.

        Notes
        ------
        - Automatically formats x-axis month labels and displays cumulative totals.
        - Confidence bands are shaded regions for variable income simulations.
        """

        import matplotlib.pyplot as plt
        import numpy as np

        colors = colors or {"fixed": "blue", "variable": "orange", "total": "black"}
        fixed_col, var_col = self.name_fixed, self.name_variable

        fixed_arr = self.fixed.project(months=months, start=start)
        if show_confidence_band and hasattr(self, "variable") and self.variable.sigma > 0:
            sims = np.array([self.variable.project(months=months, start=start, seed=i) for i in range(n_simulations)])
            var_mean = sims.mean(axis=0)
            lower_perc = np.percentile(sims, (1-confidence)/2*100, axis=0)
            upper_perc = np.percentile(sims, (1+confidence)/2*100, axis=0)
        else:
            var_mean = self.variable.project(months=months, start=start)
            lower_perc = upper_perc = var_mean

        total = fixed_arr + var_mean
        idx = month_index(start=start, months=max(months, 0))
        if len(idx) == 0:
            fig, ax = plt.subplots(figsize=figsize) if ax is None else ax
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data (months <= 0)", ha="center", va="center", transform=ax.transAxes)
            if return_fig_ax: return (ax.figure, ax)
            return

        fig = None
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Determine if dual-axis is needed
        use_dual = False
        if dual_axis is True:
            use_dual = True
        elif dual_axis == "auto":
            left_max = max(np.nanmax(fixed_arr), np.nanmax(total))
            right_max = np.nanmax(var_mean)
            use_dual = (max(left_max / right_max, right_max / left_max) >= dual_axis_ratio) if left_max*right_max>0 else False

        lines, labels = [], []
        if use_dual:
            l_fixed, = ax.plot(idx, fixed_arr, label=fixed_col, color=colors.get("fixed", "blue"))
            l_total, = ax.plot(idx, total, label="total", color=colors.get("total", "black"))
            ax.set_ylabel(ylabel_left)
            lines.extend([l_fixed, l_total])
            labels.extend([fixed_col, "total"])

            ax_r = ax.twinx()
            l_var, = ax_r.plot(idx, var_mean, linestyle="--", label=var_col, color=colors.get("variable", "orange"))
            if show_confidence_band:
                ax_r.fill_between(idx, lower_perc, upper_perc, color=colors.get("variable", "orange"), alpha=0.2)
                ax.fill_between(idx, lower_perc + fixed_arr, upper_perc + fixed_arr,
                                color=colors.get("total", "black"), alpha=0.15, label="Total CI")
            ax_r.set_ylabel(ylabel_right)
            lines.append(l_var)
            labels.append(var_col)
        else:
            l_fixed, = ax.plot(idx, fixed_arr, label=fixed_col, color=colors.get("fixed", "blue"))
            l_var, = ax.plot(idx, var_mean, label=var_col, color=colors.get("variable", "orange"), linestyle="--")
            l_total, = ax.plot(idx, total, label="total", color=colors.get("total", "black"))
            lines.extend([l_fixed, l_var, l_total])
            labels.extend([fixed_col, var_col, "total"])
            if show_confidence_band:
                ax.fill_between(idx, lower_perc, upper_perc, color=colors.get("variable", "orange"), 
                                alpha=0.2, label=f"{var_col} CI")
                ax.fill_between(idx, lower_perc + fixed_arr, upper_perc + fixed_arr,
                                color=colors.get("total", "black"), alpha=0.15, label="Total CI")
            ax.set_ylabel(ylabel_left)

        if grid: ax.grid(True, linestyle="--", alpha=0.4)
        if title: ax.set_title(title)
        ax.set_xlabel("Month")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        if legend: ax.legend(lines, labels, loc="best")

        if save_path: (fig or ax.figure).savefig(save_path, bbox_inches="tight", dpi=150)

        total_fixed_sum, total_var_sum, total_sum = fixed_arr.sum(), var_mean.sum(), fixed_arr.sum() + var_mean.sum()
        ax.text(0.02, 0.98,
                f"Total Fixed: ${total_fixed_sum:,.0f}\nTotal Variable: ${total_var_sum:,.0f}\nTotal Income: ${total_sum:,.0f}".replace(",", "."),
                transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        if return_fig_ax: return (fig or ax.figure, ax)

    def plot_contributions(
        self,
        months: int,
        start: Optional[date] = None,
        ax=None,
        figsize: tuple = (12, 6),
        title: str = "Projected Monthly Contributions",
        legend: bool = True,
        grid: bool = True,
        ylabel: str = "Total Contribution (CLP)",
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        colors: dict | None = None,
        show_confidence_band: bool = True,
        confidence: float = 0.9,
        n_simulations: int = 500,
    ):
        """
        Plot total monthly contributions with optional confidence bands for stochastic variable income.

        This method visualizes the **total monthly contributions** computed from both fixed and
        variable income streams. If the variable income has stochasticity (non-zero sigma), the
        method can simulate multiple contribution paths and display a confidence interval
        representing the variability.

        Parameters
        ----------
        months : int
            Number of months to project contributions. Must be >= 0.
        start : datetime.date, optional
            Start date for the projection. Used to align months with the calendar.
        ax : matplotlib.axes.Axes, optional
            Existing matplotlib Axes object to plot on. If None, a new figure and axes are created.
        figsize : tuple, default (12, 6)
            Size of the figure (width, height) in inches if a new figure is created.
        title : str, default "Projected Monthly Contributions"
            Title of the plot.
        legend : bool, default True
            Whether to display the legend.
        grid : bool, default True
            Whether to display gridlines.
        ylabel : str, default "Total Contribution (CLP)"
            Label for the y-axis.
        save_path : str, optional
            Path to save the figure. If None, the figure is not saved.
        return_fig_ax : bool, default False
            If True, returns the tuple (fig, ax) for further customization.
        colors : dict, optional
            Dictionary of colors for plotting. Keys:
                - "total": line color for total contributions.
                - "ci": color for confidence band (if shown).
            Defaults to {"total": "blue", "ci": "orange"}.
        show_confidence_band : bool, default True
            Whether to display a confidence band for stochastic contributions.
        confidence : float, default 0.9
            Confidence level for the band (between 0 and 1).
        n_simulations : int, default 500
            Number of stochastic simulations used to compute the confidence band.

        Returns
        -------
        None or tuple
            Returns (fig, ax) if `return_fig_ax=True`; otherwise, returns None.

        Notes
        -----
        - Confidence bands are only displayed if `show_confidence_band=True` and the variable
        income is stochastic.
        - Contributions are floored at zero, so negative values do not appear in the plot.
        - The x-axis is rotated for better readability of month labels.

        """

        import matplotlib.pyplot as plt
        import numpy as np

        colors = colors or {"total": "blue", "ci": "orange"}

        idx = month_index(start=start, months=max(months, 0))
        if months <= 0 or len(idx) == 0:
            fig, ax = plt.subplots(figsize=figsize) if ax is None else ax
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data (months <= 0)", ha="center", va="center", transform=ax.transAxes)
            if return_fig_ax: return (ax.figure, ax)
            return

        # Deterministic or stochastic contributions
        if show_confidence_band and hasattr(self.variable, "sigma") and self.variable.sigma > 0:
            sims = np.array([self.contributions(months, start=start) for _ in range(n_simulations)])
            contrib_mean = sims.mean(axis=0)
            lower = np.percentile(sims, (1-confidence)/2*100, axis=0)
            upper = np.percentile(sims, (1+confidence)/2*100, axis=0)
        else:
            contrib_mean = self.contributions(months, start=start)
            lower = upper = contrib_mean

        fig = None
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Plot total contribution
        ax.plot(idx, contrib_mean, color=colors["total"], label="Total Contribution")
        if show_confidence_band:
            ax.fill_between(idx, lower, upper, color=colors["ci"], alpha=0.2,
                            label=f"{int(confidence*100)}% CI")

        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if grid: ax.grid(True, linestyle="--", alpha=0.4)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        if legend: ax.legend(loc="best")

        # Total sum annotation
        ax.text(0.02, 0.98,
                f"Total Contributions: ${contrib_mean.sum():,.0f}".replace(",", "."),
                transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        if save_path: (fig or ax.figure).savefig(save_path, bbox_inches="tight", dpi=150)
        if return_fig_ax: return (fig or ax.figure, ax)

    def plot(
        self,
        mode: str = "income",  # "income" | "contributions"
        months: int = 12,
        start: Optional[date] = None,
        ax=None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
        legend: bool = True,
        grid: bool = True,
        ylabel_left: Optional[str] = None,
        ylabel_right: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        dual_axis: str | bool = "auto",
        dual_axis_ratio: float = 3.0,
        show_confidence_band: bool = True,
        confidence: float = 0.9,
        n_simulations: int = 500,
        colors: dict | None = None,
    ):
        """
        Unified plot wrapper to call either `plot_income` or `plot_contributions`.

        Parameters
        ----------
        mode : str
            "income" to plot income, "contributions" to plot contributions.
        Other parameters are passed directly to the corresponding plotting method.
        """
        if mode == "income":
            return self.plot_income(
                months=months,
                start=start,
                ax=ax,
                figsize=figsize,
                title=title,
                legend=legend,
                grid=grid,
                ylabel_left=ylabel_left or "Fixed/Total (CLP)",
                ylabel_right=ylabel_right or "Variable (CLP)",
                save_path=save_path,
                return_fig_ax=return_fig_ax,
                dual_axis=dual_axis,
                dual_axis_ratio=dual_axis_ratio,
                show_confidence_band=show_confidence_band,
                confidence=confidence,
                n_simulations=n_simulations,
                colors=colors,
            )
        elif mode == "contributions":
            return self.plot_contributions(
                months=months,
                start=start,
                ax=ax,
                figsize=figsize,
                title=title,
                legend=legend,
                grid=grid,
                ylabel=ylabel_left or "Total Contribution (CLP)",  # Only one y-axis for contributions
                save_path=save_path,
                return_fig_ax=return_fig_ax,
                colors=colors,
                show_confidence_band=show_confidence_band,
                confidence=confidence,
                n_simulations=n_simulations,
            )
        else:
            raise ValueError("Invalid kind. Must be 'income' or 'contributions'.")

    def summary(
        self,
        months: int,
        start: Optional[date] = None,
        variable_threshold: Optional[float] = None,
        round_digits: int = 2,
    ) -> pd.Series:
        """
        Return a compact summary of projected incomes for the horizon.

        This method delegates computation to `income_metrics` and converts the
        dataclass result into a Series with optional rounding.

        Parameters
        ----------
        months : int
            Projection horizon in months (> 0).
        start : Optional[date]
            Start date to align calendar (affects variable seasonality).
        variable_threshold : Optional[float]
            If provided, include fraction of months where variable income is below this threshold.
        round_digits : int, default 2
            Decimal rounding for the returned Series.

        Returns
        -------
        pd.Series
            Keys: ["months", "total_income", "total_fixed", "total_variable",
                "mean_total", "mean_fixed", "mean_variable",
                "fixed_share", "variable_share", "std_variable", 
                "coefvar_variable", "min_variable", "max_variable", 
                "pct_variable_below_threshold"].
        """
        metrics = self.income_metrics(
            months=months,
            start=start,
            variable_threshold=variable_threshold,
        )
        out = pd.Series(vars(metrics))
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
                "salary_raises": None if self.fixed.salary_raises is None 
                                else {d.isoformat(): v for d, v in self.fixed.salary_raises.items()},
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
            salary_raises=None if fx.get("salary_raises") is None else {
                date.fromisoformat(k): float(v) for k, v in fx["salary_raises"].items()
            },
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

    
    def __repr__(self) -> str:
        try:
            metrics = self.income_metrics(months=12, start=date(2025, 1, 1))  # ejemplo: horizonte 12 meses
            return (
                f"{self.__class__.__name__}(horizon=12 months, "
                f"total_income={metrics.total_income:.2f}, "
                f"total_fixed={metrics.total_fixed:.2f}, "
                f"total_variable={metrics.total_variable:.2f}, "
                f"mean_total={metrics.mean_total:.2f}, "
                f"fixed_share={metrics.fixed_share*100:.1f}%, "
                f"variable_share={metrics.variable_share*100:.1f}%)"
            )
        except Exception:
            return f"{self.__class__.__name__}(IncomeModel instance)"

# ---------------------------------------------------------------------------
# Quick sanity check block (manual execution only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date

    # Create income model
    model = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.04),
        variable=VariableIncome(base=200_000.0, sigma=0.01, seed=42),
    )

    months = 6
    start_date = date(2025, 9, 1)

    # Projection (DataFrame with [fixed, variable, total] columns)
    print("=== Projection (6 months) ===")
    df = model.project(months=months, start=start_date, as_dataframe=True)
    print(df)

    # Contributions using explicit monthly_contribution (12-month lists, rotated)
    model.monthly_contribution = {
        "fixed": [0.35] * 12,      # 12-month annual fractions
        "variable": [1.0] * 12,    # 12-month annual fractions
    }
    print("\n=== Contributions (fixed=0.35, variable=1.0, rotated) ===")
    contrib = model.contributions(months=months, start=start_date)
    print(contrib)

    # Compact summary as a Series
    print("\n=== Summary() ===")
    s = model.summary(months=months, start=start_date, round_digits=2)
    print(s)

    # Detailed metrics (dataclass)
    print("\n=== Income Metrics (variable threshold = 150k) ===")
    im = model.income_metrics(months=months, start=start_date, variable_threshold=150_000.0)
    print(im)

    # Basic sanity checks
    print("\n=== Sanity checks ===")
    total_from_parts = float(df["fixed"].sum() + df["variable"].sum())
    print("Sum(fixed)+Sum(variable) == Sum(total)?",
          abs(total_from_parts - float(df["total"].sum())) < 1e-6)
    print("fixed_share + variable_share ≈ 1?",
          abs(s["fixed_share"] + s["variable_share"] - 1.0) < 1e-6)



