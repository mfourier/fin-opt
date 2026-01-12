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
>>> df = income.project(months=12, start=date(2025, 1, 1), output="dataframe")
>>> contrib = income.contributions(months=12)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, Optional, Literal, Union, TYPE_CHECKING, List

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from .expenses import ExpenseModel

# Reuse common utilities
from .utils import (
    check_non_negative,
    annual_to_monthly,
    month_index,
    normalize_start_month,
    millions_formatter,
    format_currency,
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
    project(months, start=None, output="array", n_sims=1)
        Returns deterministic monthly income projection for the given horizon.
        Supports multiple output formats and optional replication for API
        consistency with VariableIncome (all simulations are identical).

    Notes
    -----
    - Income values are always non-negative; enforced during initialization.
    - The projection is fully deterministic (no stochasticity).
    - Salary raises are applied based on calendar dates relative to start date.
    - When n_sims > 1, the deterministic projection is replicated across simulations.
    - Can be combined with variable income streams in `IncomeModel`.

    Examples
    --------
    >>> from datetime import date
    >>> fi = FixedIncome(
    ...     base=1_400_000.0,
    ...     annual_growth=0.03,
    ...     salary_raises={date(2025, 7, 1): 200_000}
    ... )

    # Single projection
    >>> fi.project(12, start=date(2025, 1, 1))
    array([1400000.0, 1403485.0, ..., 1620501.0, ...])  # raise applied from month 6

    # Multiple simulations
    >>> fi.project(12, start=date(2025, 1, 1), n_sims=100)
    array([[1400000.0, 1403485.0, ...],
        [1400000.0, 1403485.0, ...],
        ...])  # shape: (100, 12), all rows identical

    # Series output with calendar index
    >>> fi.project(12, start=date(2025, 1, 1), output="series")
    2025-01-01    1400000.0
    2025-02-01    1403485.0
    ...
    Name: fixed, dtype: float64
    """
    base: float
    annual_growth: float = 0.0
    salary_raises: Optional[Dict[date, float]] = None
    name: str = "fixed"

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        output: Literal["array", "series"] = "array",
        n_sims: int = 1,
    ) -> np.ndarray | pd.Series:
        """
        Project deterministic monthly income stream with optional growth and raises.

        Generates a deterministic projection incorporating annual growth (compounded
        monthly) and scheduled salary raises at specific calendar dates. Since
        FixedIncome is deterministic, all simulations (when n_sims > 1) are identical.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0. If `months <= 0`, returns
            empty array or Series.
        start : Optional[date], default None
            Start date for the projection. Required when `salary_raises` are specified.
            Used to convert raise dates to month offsets and to build calendar index
            when `output='series'`.
        output : Literal["array", "series"], default "array"
            Output format:
            - "array": returns np.ndarray (no calendar index)
            - "series": returns pd.Series with monthly calendar index
                    (only valid when n_sims=1)
        n_sims : int, default 1
            Number of simulations. For API consistency with VariableIncome.
            Since FixedIncome is deterministic, all simulations are identical.
            When n_sims > 1, returns shape (n_sims, months) with replicated rows.

        Returns
        -------
        np.ndarray or pd.Series
            Projected monthly income for the given horizon.
            - If n_sims=1 and output="array": np.ndarray of shape (months,)
            - If n_sims>1 and output="array": np.ndarray of shape (n_sims, months)
            - If n_sims=1 and output="series": pd.Series indexed by first-of-month dates
            - If n_sims>1 and output="series": raises ValueError

        Raises
        ------
        ValueError
            If `salary_raises` are specified but `start` is None.
            If `output` is not 'array' or 'series'.
            If `output='series'` and `n_sims > 1`.
            If `n_sims` is not a positive integer.
        """
        # Validate n_sims
        if not isinstance(n_sims, int) or n_sims < 1:
            raise ValueError(f"n_sims must be a positive integer, got: {n_sims}")

        # Validate output FIRST (before any logic that uses it)
        if output not in ("array", "series"):
            raise ValueError(f"output must be 'array' or 'series', got: {output}")

        if months <= 0:
            arr = np.zeros((n_sims, 0) if n_sims > 1 else 0, dtype=float)
            if output == "array":
                return arr
            else:  # series (validated above, safe)
                if n_sims > 1:
                    raise ValueError("output='series' incompatible with n_sims > 1")
                idx = month_index(start=start, months=0)
                return pd.Series(arr, index=idx, name=self.name)

        # Generate base projection (deterministic, shape: (months,))
        if not self.salary_raises:
            # Vectorized: no raises case
            m = annual_to_monthly(self.annual_growth)
            t = np.arange(months, dtype=float)
            arr = self.base * np.power(1.0 + m, t)
        else:
            # With raises: sequential calculation (small overhead, typically <10 raises)
            if start is None:
                raise ValueError("start date is required when salary_raises are specified")

            # Convert raises to month offsets
            raise_schedule = {}
            for raise_date, raise_amount in self.salary_raises.items():
                raise_month = self._date_to_month_offset(start, raise_date)
                if 0 <= raise_month < months:
                    raise_schedule[raise_month] = raise_amount

            # Month-by-month calculation (preserves compounding semantics)
            monthly_rate = annual_to_monthly(self.annual_growth)
            arr = np.zeros(months, dtype=float)
            current_salary = self.base

            for month in range(months):
                if month in raise_schedule:
                    current_salary += raise_schedule[month]
                arr[month] = current_salary
                if month < months - 1:
                    current_salary *= (1.0 + monthly_rate)
        
        # Replicate for n_sims > 1 (deterministic replication)
        if n_sims > 1:
            arr = np.tile(arr, (n_sims, 1))
        
        # Format output
        if output == "array":
            return arr
        elif output == "series":
            if n_sims > 1:
                raise ValueError(
                    f"output='series' requires n_sims=1 (got n_sims={n_sims}). "
                    "Use output='array' for multiple simulations."
                )
            idx = month_index(start=start, months=months)
            return pd.Series(arr, index=idx, name=self.name)
        else:
            raise ValueError(f"output must be 'array' or 'series', got: {output}")


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
    seed : Optional[int], default None
        Random seed for reproducible noise generation. Controls the RNG stream
        for all simulations when using n_sims > 1.

    Methods
    -------
    project(months, start=None, seed=None, output="array", n_sims=1)
        Returns projected income for specified months with vectorized support for
        multiple independent simulations. Supports calendar alignment via `start`
        and multiple output formats (array or Series).

    Performance
    -----------
    Vectorized generation via `n_sims` parameter eliminates Python-level loops:
    - `project(months=240, n_sims=500)` is ~100x faster than 500 sequential calls
    - Single memory allocation and NumPy vectorization throughout
    - Recommended for Monte Carlo simulations and uncertainty quantification

    Examples
    --------
    >>> vi = VariableIncome(
    ...     base=200_000.0,
    ...     seasonality=[1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.0, 0.8, 0.9, 1.0, 1.05, 1.15],
    ...     sigma=0.15,
    ...     floor=50_000.0,
    ...     cap=400_000.0,
    ...     annual_growth=0.02,
    ...     seed=42
    ... )

    # Single realization (backward compatible)
    >>> vi.project(6)
    array([200000., 183000., 224000., 210500., 240300., 230100.])

    # Multiple simulations (vectorized)
    >>> sims = vi.project(6, n_sims=100)
    >>> sims.shape
    (100, 6)
    >>> sims.mean(axis=0)  # Average across simulations
    array([200000., 182500., ...])

    # Calendar-aligned Series output (n_sims=1 only)
    >>> vi.project(6, start=date(2025, 1, 1), output="series")
    2025-01-01    200000.0
    2025-02-01    183000.0
    ...
    Name: variable, dtype: float64
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
            # Convert to tuple for validation and counting
            s = tuple(float(x) for x in self.seasonality)
            if len(s) != 12:
                raise ValueError("seasonality must have length 12 for Jan..Dec.")
            if any(v < 0 for v in s):
                raise ValueError("seasonality factors must be non-negative.")
            # Store validated tuple (requires workaround for frozen dataclass)
            object.__setattr__(self, 'seasonality', s)

    def project(
        self,
        months: int,
        *,
        start: Optional[date | int] = None,
        seed: Optional[int] = None,
        output: Literal["array", "series"] = "array",
        n_sims: int = 1,
    ) -> np.ndarray | pd.Series:
        """
        Project variable monthly income with seasonality, noise, and growth.

        Generates income projection incorporating annual growth, seasonal factors,
        Gaussian noise, and floor/cap guardrails. Can be aligned to any starting
        calendar month for proper seasonality rotation. Supports vectorized
        generation of multiple independent simulations.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0. If `months <= 0`, returns
            empty array or Series.
        start : Optional[date | int], default None
            Starting month for seasonality alignment:
            - If date: uses start.month (1-12)
            - If int: must be 1-12 (1=January)
            - If None: assumes January
        seed : Optional[int], default None
            Random seed for Gaussian noise. If None, uses instance seed.
            If both are None, generates non-deterministic noise.
            When n_sims > 1, seed controls reproducibility of the entire batch.
        output : Literal["array", "series"], default "array"
            Output format:
            - "array": returns np.ndarray
            - "series": returns pd.Series with monthly calendar index
                    (only valid when n_sims=1)
        n_sims : int, default 1
            Number of independent simulations to generate. When n_sims > 1:
            - Returns shape (n_sims, months) array
            - Each simulation uses independent noise draws from the same RNG stream
            - output='series' raises ValueError (incompatible with multiple sims)

        Returns
        -------
        np.ndarray or pd.Series
            Projected monthly income for the given horizon.
            - If n_sims=1 and output="array": np.ndarray of shape (months,)
            - If n_sims>1 and output="array": np.ndarray of shape (n_sims, months)
            - If n_sims=1 and output="series": pd.Series indexed by first-of-month dates
            - If n_sims>1 and output="series": raises ValueError

        Raises
        ------
        ValueError
            If `output` is not 'array' or 'series'.
            If `output='series'` and `n_sims > 1`.
            If `n_sims` is not a positive integer.

        Notes
        -----
        - Annual growth converted to monthly rate: m = (1 + annual_growth)^(1/12) - 1
        - Seasonality applied multiplicatively after growth
        - Noise applied multiplicatively: income * (1 + N(0, sigma))
        - Floor/cap applied after noise, then non-negativity enforced
        - Seasonality rotates based on `start` and repeats cyclically
        - When n_sims > 1, all simulations share the same deterministic base_path
        and seasonal pattern, but have independent noise realizations

        Examples
        --------
        >>> vi = VariableIncome(
        ...     base=200_000.0,
        ...     seasonality=[1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 
        ...                  1.0, 0.8, 0.9, 1.0, 1.05, 1.15],
        ...     sigma=0.15,
        ...     floor=50_000.0,
        ...     seed=42
        ... )
        
        # Single simulation
        >>> vi.project(6)
        array([200000., 183000., 224000., 210500., 240300., 230100.])
        
        # Multiple simulations (vectorized)
        >>> vi.project(6, n_sims=100)
        array([[200000., 183000., ...],
            [195000., 180000., ...],
            ...])  # shape: (100, 6)
        
        # Series output with calendar index (n_sims=1 only)
        >>> vi.project(6, start=date(2025, 1, 1), output="series")
        2025-01-01    200000.0
        2025-02-01    183000.0
        ...
        Name: variable, dtype: float64
        """
        # Validate n_sims
        if not isinstance(n_sims, int) or n_sims < 1:
            raise ValueError(f"n_sims must be a positive integer, got: {n_sims}")

        # Validate output FIRST (before any logic that uses it)
        if output not in ("array", "series"):
            raise ValueError(f"output must be 'array' or 'series', got: {output}")
            
        # Use method seed if provided, else instance seed
        rng_seed = seed if seed is not None else self.seed
        
        # Handle edge case: months <= 0
        if months <= 0:
            arr = np.zeros((n_sims, 0) if n_sims > 1 else 0, dtype=float)
            if output == "array":
                return arr
            elif output == "series":
                if n_sims > 1:
                    raise ValueError("output='series' incompatible with n_sims > 1")
                idx = month_index(start=start, months=0)
                return pd.Series(arr, index=idx, name=self.name)
            else:
                raise ValueError(f"output must be 'array' or 'series', got: {output}")
        
        # 1. Base path: deterministic growth trajectory (shape: (months,))
        m = annual_to_monthly(self.annual_growth)
        t = np.arange(months, dtype=float)
        base_path = self.base * np.power(1.0 + m, t)
        
        # 2. Seasonality: vectorized application (shape: (months,))
        if self.seasonality is None:
            means = base_path
        else:
            s = np.array(self.seasonality, dtype=float)
            offset = normalize_start_month(start)
            seasonal_idx = (offset + np.arange(months)) % 12
            means = base_path * s[seasonal_idx]
        
        # 3. Noise: vectorized generation (shape: (n_sims, months) or (months,))
        if self.sigma == 0.0:
            # No noise: replicate or return directly
            if n_sims == 1:
                noisy = means
            else:
                noisy = np.tile(means, (n_sims, 1))
        else:
            rng = np.random.default_rng(rng_seed)
            # Generate all simulations at once
            noise = rng.normal(loc=0.0, scale=self.sigma, size=(n_sims, months))
            # Broadcasting: (1, months) * (n_sims, months) -> (n_sims, months)
            noisy = means[None, :] * (1.0 + noise)
            
            if n_sims == 1:
                noisy = noisy[0, :]  # Squeeze for backward compatibility
        
        # 4. Guardrails: work with any shape via broadcasting
        if self.floor is not None:
            noisy = np.maximum(noisy, self.floor)
        if self.cap is not None:
            noisy = np.minimum(noisy, self.cap)
        
        arr = np.maximum(noisy, 0.0)
        
        # 5. Format output
        if output == "array":
            return arr
        elif output == "series":
            if n_sims > 1:
                raise ValueError(
                    f"output='series' requires n_sims=1 (got n_sims={n_sims}). "
                    "Use output='array' for multiple simulations."
                )
            idx = month_index(start=start, months=months)
            return pd.Series(arr, index=idx, name=self.name)
        else:
            raise ValueError(f"output must be 'array' or 'series', got: {output}")

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
    project(months, start=None, output="series", seed=None)
        Returns projected monthly income as Series (total) or DataFrame (breakdown).
    contributions(months, start=None, seed=None, output="series")
        Returns projected monthly contributions as array or Series.
    to_dict()
        Serializes the model configuration and parameters into a dictionary.
    from_dict(payload)
        Reconstructs an `IncomeModel` instance from a dictionary payload.

    Notes
    -----
    - The `start` parameter defines the calendar alignment for both income projections
      and rotation of contribution fractions. For `VariableIncome`, it sets the
      first month for seasonality.
    - The `seed` parameter controls reproducibility of variable income realizations.
      If None, uses the variable stream's instance seed. If both None, non-deterministic.
    - Contribution fractions are automatically rotated according to the starting month
      and repeated every 12 months to cover longer horizons.
    - Contributions are floored at zero; negative values are never returned.
    - Both `project` and `contributions` handle `months <= 0` gracefully,
      returning empty structures with proper indices.

    Example
    -------
    >>> from datetime import date
    >>> fi = FixedIncome(base=1_400_000.0, annual_growth=0.02)
    >>> vi = VariableIncome(base=200_000.0, sigma=0.1, seed=123)
    >>> income_model = IncomeModel(fixed=fi, variable=vi)
    
    # Total income as Series
    >>> income_model.project(months=6, start=date(2025,1,1))
    2025-01-01    1610000.0
    ...
    
    # Detailed breakdown as DataFrame
    >>> df = income_model.project(months=6, start=date(2025,1,1), output="dataframe")
    >>> df
                  fixed   variable      total
    2025-01-01  1400000  210000.0  1610000.0
    2025-02-01  1402323  190000.0  1592323.0
    
    # Contributions
    >>> contrib = income_model.contributions(months=6, start=date(2025,1,1))
    >>> contrib
    2025-01-01    620000.0
    2025-02-01    624200.0
    ...
    """
    
    fixed: Union[FixedIncome, List[FixedIncome]]
    variable: Optional[Union[VariableIncome, List[VariableIncome]]] = None
    name_fixed: str = field(default="fixed", init=True)
    name_variable: str = field(default="variable", init=True)
    monthly_contribution: dict = None
    expenses: Optional["ExpenseModel"] = None  # Optional expense model for net income

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        output: Literal["series", "dataframe", "array"] = "series",
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> pd.Series | pd.DataFrame | dict[str, np.ndarray]:
        """
        Project combined fixed and variable income streams over a specified horizon.

        Generates monthly income projections by summing the deterministic fixed
        income and the variable income streams. Supports vectorized generation of
        multiple independent simulations for uncertainty quantification.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0.
        start : Optional[date], default None
            Start date for calendar alignment and seasonality rotation.
        output : Literal["series", "dataframe", "array"], default "series"
            Output format:
            - "series": total income as pd.Series (n_sims=1 only)
            - "dataframe": breakdown DataFrame (n_sims=1 only)
            - "array": dict with arrays for each component
        seed : Optional[int], default None
            Random seed for variable income.
        n_sims : int, default 1
            Number of independent simulations. When n_sims > 1, only output="array" is valid.

        Returns
        -------
        pd.Series, pd.DataFrame, or dict[str, np.ndarray]
            - n_sims=1, output="series": Series of total income
            - n_sims=1, output="dataframe": DataFrame with [fixed, variable, total]
            - n_sims=1, output="array": dict with shape (months,) arrays
            - n_sims>1, output="array": dict with shape (n_sims, months) arrays
            - If months <= 0: returns empty structure matching the output format

        Raises
        ------
        ValueError
            If `n_sims` is not a positive integer.
            If `output` is not 'series', 'dataframe', or 'array'.
            If `n_sims > 1` and `output != "array"`.

        Notes
        -----
        - When n_sims > 1, fixed component is replicated (deterministic) while
        variable component has independent noise realizations per simulation.
        - Array output returns dict with keys: 'fixed', 'variable', 'total'.
        - Seed controls reproducibility of all variable income simulations.

        Examples
        --------
        # Single simulation as Series (backward compatible)
        >>> income.project(6, start=date(2025, 1, 1))
        2025-01-01    1610000.0
        ...

        # Detailed breakdown as DataFrame
        >>> income.project(6, start=date(2025, 1, 1), output="dataframe")
                        fixed   variable      total
        2025-01-01  1400000  210000.0  1610000.0
        ...

        # Single simulation with array output
        >>> result = income.project(6, output="array")
        >>> result['total'].shape
        (6,)

        # Multiple simulations (vectorized)
        >>> result = income.project(6, n_sims=100, output="array")
        >>> result['variable'].shape
        (100, 6)
        >>> result['total'].mean(axis=0)  # Mean across simulations
        array([1610000., ...])
        """
        # Validate n_sims
        if not isinstance(n_sims, int) or n_sims < 1:
            raise ValueError(f"n_sims must be a positive integer, got: {n_sims}")
        
        # Validate output
        if output not in ("series", "dataframe", "array"):
            raise ValueError(f"output must be 'series', 'dataframe', or 'array', got: {output}")
        
        # Validate n_sims > 1 constraint
        if n_sims > 1 and output != "array":
            raise ValueError(
                f"output='{output}' requires n_sims=1. "
                f"Use output='array' with n_sims={n_sims}."
            )
        
        idx = month_index(start=start, months=max(months, 0))
        
        if months <= 0:
            if output == "array":
                shape = (n_sims, 0) if n_sims > 1 else (0,)
                return {
                    self.name_fixed: np.zeros(shape, dtype=float),
                    self.name_variable: np.zeros(shape, dtype=float),
                    "total": np.zeros(shape, dtype=float),
                }
            elif output == "dataframe":
                return pd.DataFrame(
                    columns=[self.name_fixed, self.name_variable, "total"],
                    index=idx
                )
            else:  # series
                return pd.Series(dtype=float, index=idx, name="total")
        
        # Get arrays: shape (n_sims, months) or (months,)
        if isinstance(self.fixed, list):
            # Sum up list of fixed incomes
            projections = [
                f.project(months, start=start, output="array", n_sims=n_sims)
                for f in self.fixed
            ]
            fixed_arr = sum(projections)
        else:
            fixed_arr = self.fixed.project(months, start=start, output="array", n_sims=n_sims)

        if self.variable is not None:
            if isinstance(self.variable, list):
                # Sum up list of variable incomes with independent seeds
                projections = []
                for i, v in enumerate(self.variable):
                    s = seed + i if seed is not None else None
                    projections.append(v.project(
                        months, start=start, seed=s, output="array", n_sims=n_sims
                    ))
                variable_arr = sum(projections)
            else:
                variable_arr = self.variable.project(
                    months, start=start, seed=seed, output="array", n_sims=n_sims
                )
        else:
            # No variable income - return zeros
            shape = (n_sims, months) if n_sims > 1 else (months,)
            variable_arr = np.zeros(shape, dtype=float)
        total_arr = fixed_arr + variable_arr
        
        if output == "array":
            return {
                self.name_fixed: fixed_arr,
                self.name_variable: variable_arr,
                "total": total_arr,
            }
        elif output == "dataframe":
            return pd.DataFrame(
                {self.name_fixed: fixed_arr, self.name_variable: variable_arr, "total": total_arr},
                index=idx,
            )
        else:  # series
            return pd.Series(total_arr, index=idx, name="total")

    def contributions(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        output: Literal["array", "series"] = "series",
        n_sims: int = 1,
    ) -> np.ndarray | pd.Series:
        """
        Compute monthly contributions from fixed and variable income streams.

        Calculates monthly contributions by applying monthly fractional rates to the 
        projected fixed and variable incomes. Contribution fractions are specified
        as 12-month arrays and rotated based on the starting month. Supports vectorized
        generation of multiple independent simulations.

        Parameters
        ----------
        months : int
            Number of months to compute contributions. Must be >= 0.
        start : Optional[date], default None
            Calendar start date for contributions. Determines:
            - Seasonality alignment of `VariableIncome`
            - Rotation of the 12-month contribution fractions
            If None, January is assumed.
        seed : Optional[int], default None
            Random seed for variable income. If None, uses variable stream's
            instance seed. If both None, non-deterministic.
        output : Literal["array", "series"], default "series"
            Output format:
            - "array": returns np.ndarray
            - "series": returns pd.Series with monthly calendar index (n_sims=1 only)
        n_sims : int, default 1
            Number of independent simulations. When n_sims > 1, only output="array" is valid.

        Returns
        -------
        np.ndarray or pd.Series
            Monthly contributions for the given horizon.
            - If n_sims=1 and output="array": np.ndarray of shape (months,)
            - If n_sims>1 and output="array": np.ndarray of shape (n_sims, months)
            - If n_sims=1 and output="series": pd.Series indexed by first-of-month dates
            - Contributions are non-negative (floored at zero)

        Raises
        ------
        ValueError
            If `n_sims` is not a positive integer.
            If `output` is not 'array' or 'series'.
            If `output='series'` and `n_sims > 1`.
            If `self.monthly_contribution` is provided but either the fixed or variable
            fraction arrays do not have length 12.

        Notes
        -----
        - Monthly contribution: contrib_t = α_fixed_t * income_fixed_t + α_variable_t * income_variable_t
        - Fractions rotate annually based on `start` and repeat cyclically
        - Negative contributions are floored to zero
        - When n_sims > 1, variable income has independent noise per simulation

        Examples
        --------
        # Single simulation as Series (backward compatible)
        >>> income.contributions(6, start=date(2025, 1, 1))
        2025-01-01    620000.0
        ...
        
        # Array output
        >>> income.contributions(6, start=date(2025, 1, 1), output="array")
        array([620000., 624200., 628400., 632600., 636800., 641000.])
        
        # Multiple simulations (vectorized)
        >>> contrib = income.contributions(6, n_sims=100, output="array")
        >>> contrib.shape
        (100, 6)
        >>> contrib.mean(axis=0)  # Mean contribution per month
        array([620000., ...])
        
        # Custom fractions
        >>> income.monthly_contribution = {"fixed": [0.35]*12, "variable": [1.0]*12}
        >>> income.contributions(6, start=date(2025, 1, 1))
        """
        # Validate n_sims
        if not isinstance(n_sims, int) or n_sims < 1:
            raise ValueError(f"n_sims must be a positive integer, got: {n_sims}")
        
        # Validate output
        if output not in ("array", "series"):
            raise ValueError(f"output must be 'array' or 'series', got: {output}")
        
        # Validate n_sims > 1 constraint
        if n_sims > 1 and output != "array":
            raise ValueError(
                f"output='series' requires n_sims=1. "
                f"Use output='array' with n_sims={n_sims}."
            )
        
        if months <= 0:
            shape = (n_sims, 0) if n_sims > 1 else (0,)
            arr = np.zeros(shape, dtype=float)
            if output == "array":
                return arr
            else:  # series
                idx = month_index(start=start, months=0)
                return pd.Series(arr, index=idx, name="contribution")

        # Project incomes as arrays: shape (n_sims, months) or (months,)
        if isinstance(self.fixed, list):
            # Sum up list of fixed incomes
            projections = [
                f.project(months, start=start, output="array", n_sims=n_sims)
                for f in self.fixed
            ]
            fixed_arr = sum(projections)
        else:
            fixed_arr = self.fixed.project(months, start=start, output="array", n_sims=n_sims)

        if self.variable is not None:
            if isinstance(self.variable, list):
                # Sum up list of variable incomes with independent seeds
                projections = []
                for i, v in enumerate(self.variable):
                    s = seed + i if seed is not None else None
                    projections.append(v.project(
                        months, start=start, seed=s, output="array", n_sims=n_sims
                    ))
                variable_arr = sum(projections)
            else:
                variable_arr = self.variable.project(months, start=start, seed=seed, output="array", n_sims=n_sims)
        else:
            shape = (n_sims, months) if n_sims > 1 else (months,)
            variable_arr = np.zeros(shape, dtype=float)

        # Initialize contribution fractions (12-month arrays)
        if self.monthly_contribution is None:
            fixed_fractions = np.full(12, 0.3, dtype=float)
            variable_fractions = np.ones(12, dtype=float)
        else:
            fixed_fractions = np.asarray(self.monthly_contribution["fixed"], dtype=float)
            variable_fractions = np.asarray(self.monthly_contribution["variable"], dtype=float)
            if len(fixed_fractions) != 12 or len(variable_fractions) != 12:
                raise ValueError("monthly_contribution lists must have length 12.")

        # Rotate fractions vectorized: shape (months,)
        offset = normalize_start_month(start)
        fixed_frac_full = fixed_fractions[(offset + np.arange(months)) % 12]
        var_frac_full = variable_fractions[(offset + np.arange(months)) % 12]

        # Compute contributions with broadcasting
        if n_sims == 1:
            # fixed_frac_full: (months,), fixed_arr: (months,)
            contrib_arr = fixed_frac_full * fixed_arr + var_frac_full * variable_arr
        else:
            # fixed_frac_full: (months,), fixed_arr: (n_sims, months)
            # Broadcasting: (1, months) * (n_sims, months) → (n_sims, months)
            contrib_arr = (
                fixed_frac_full[None, :] * fixed_arr + 
                var_frac_full[None, :] * variable_arr
            )
        
        contrib_arr = np.maximum(contrib_arr, 0.0)

        # Return in requested format
        if output == "array":
            return contrib_arr
        else:  # series (validated above, safe)
            idx = month_index(start=start, months=months)
            return pd.Series(contrib_arr, index=idx, name="contribution")

    # ---------------------------------------------------------------------------
    # Extended methods for expense integration
    # ---------------------------------------------------------------------------

    def gross_income(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> np.ndarray:
        """
        Total income before expenses (I_t = y_fixed + y_variable).

        This is equivalent to project(output="array")["total"].

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment.
        seed : int, optional
            Random seed for variable income.
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        np.ndarray
            If n_sims=1: shape (months,)
            If n_sims>1: shape (n_sims, months)
        """
        result = self.project(months, start=start, seed=seed, output="array", n_sims=n_sims)
        return result["total"]

    def net_income(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> np.ndarray:
        """
        Income after expenses: Net_t = I_t - C_t.

        Can be negative if expenses exceed income in some months.

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment.
        seed : int, optional
            Random seed for stochastic components.
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        np.ndarray
            If n_sims=1: shape (months,)
            If n_sims>1: shape (n_sims, months)

        Notes
        -----
        If no expense model is configured, returns gross_income (no expense deduction).
        """
        gross = self.gross_income(months, start=start, seed=seed, n_sims=n_sims)
        
        if self.expenses is None:
            return gross
        
        # Use different seed offset for expenses to ensure independence
        expense_seed = seed + 1000 if seed is not None else None
        expense_result = self.expenses.project(
            months, start=start, seed=expense_seed, output="array", n_sims=n_sims
        )
        expense_total = expense_result["total"]
        
        # Handle dimension consistency
        if n_sims == 1 and gross.ndim == 1 and expense_total.ndim == 1:
            return gross - expense_total
        elif n_sims == 1:
            # Flatten if needed
            return gross.flatten() - expense_total.flatten()
        else:
            return gross - expense_total

    def disposable_income(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> np.ndarray:
        """
        Non-negative disposable income: D_t = max(0, I_t - C_t).

        Clamps net income to zero (cannot spend more than earned for savings).

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment.
        seed : int, optional
            Random seed for stochastic components.
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        np.ndarray
            If n_sims=1: shape (months,)
            If n_sims>1: shape (n_sims, months)
            All values >= 0.
        """
        net = self.net_income(months, start=start, seed=seed, n_sims=n_sims)
        return np.maximum(net, 0.0)

    def contributions_from_disposable(
        self,
        months: int,
        *,
        savings_rate: Union[float, np.ndarray] = 0.3,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> np.ndarray:
        """
        Contributions as fraction of disposable income: A_t = savings_rate * D_t.

        Alternative to contributions() that first deducts expenses.

        Parameters
        ----------
        months : int
            Number of months to project.
        savings_rate : float or ndarray, default 0.3
            Fraction of disposable income to save.
            - float: constant rate for all months
            - ndarray shape (12,): monthly rotating rates (Jan=0, ..., Dec=11)
            - ndarray shape (months,): specific rate per month
        start : date, optional
            Start date for calendar alignment.
        seed : int, optional
            Random seed for stochastic components.
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        np.ndarray
            If n_sims=1: shape (months,)
            If n_sims>1: shape (n_sims, months)

        Notes
        -----
        This method is useful when you want to model:
        1. Gross income (salary)
        2. Minus expenses (rent, food, etc.)
        3. Equals disposable income
        4. Times savings_rate = contributions

        Example
        -------
        >>> model = IncomeModel(fixed=fi, variable=vi, expenses=em)
        >>> A = model.contributions_from_disposable(24, savings_rate=0.4)
        """
        D = self.disposable_income(months, start=start, seed=seed, n_sims=n_sims)
        
        # Handle savings_rate
        if isinstance(savings_rate, (int, float)):
            rate = float(savings_rate)
            return rate * D
        
        savings_rate = np.asarray(savings_rate)
        
        if savings_rate.shape == (12,):
            # Rotate based on start month
            offset = normalize_start_month(start)
            rate_full = savings_rate[(offset + np.arange(months)) % 12]
        else:
            rate_full = savings_rate[:months]
        
        # Broadcast for n_sims
        if n_sims == 1 or D.ndim == 1:
            return rate_full * D
        else:
            return rate_full[None, :] * D

    def plot_income(
        self,
        months: int,
        start: Optional[date] = None,
        ax=None,
        figsize: tuple = (12, 6),
        title: Optional[str] = r"Projected Monthly Income $(Y_t)$",
        legend: bool = True,
        grid: bool = True,
        ylabel_left: str = "Fixed/Total (CLP)",
        ylabel_right: str = "Variable (CLP)",
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        dual_axis: str | bool = "auto",
        dual_axis_ratio: float = 3.0,
        show_trajectories: bool = True,
        show_confidence_band: bool = True,
        trajectory_alpha: float = 0.07,
        confidence: float = 0.9,
        n_simulations: int = 500,
        colors: dict | None = None,
    ):
        """
        Plot projected monthly income streams (fixed, variable, total) with optional Monte Carlo trajectories and confidence bands.

        Visualizes the evolution of fixed income (deterministic), variable income (potentially stochastic),
        and their sum over a specified horizon. Supports two visualization modes: individual stochastic
        trajectories (Monte Carlo style) and/or statistical confidence intervals. Automatically handles
        dual-axis rendering when income scales differ significantly.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0. If `months <= 0`, returns an empty
            plot with a "no data" message.
        start : datetime.date, optional
            Start date for the projection. Used to align months with the calendar, determine
            seasonality rotation for variable income, and apply salary raises at specific dates
            for fixed income. If None, January (month 1) is assumed for seasonality.
        ax : matplotlib.axes.Axes, optional
            Existing Axes object to plot on. If None, a new figure and axes are created.
        figsize : tuple, default (12, 6)
            Figure size (width, height) in inches when creating a new figure.
        title : str, optional
            Plot title. If None, no title is displayed.
        legend : bool, default True
            Whether to display the legend.
        grid : bool, default True
            Whether to display gridlines.
        ylabel_left : str, default "Fixed/Total (CLP)"
            Label for the left y-axis (fixed and total income in single-axis mode,
            or only these streams in dual-axis mode).
        ylabel_right : str, default "Variable (CLP)"
            Label for the right y-axis when dual-axis is active.
        save_path : str, optional
            File path to save the figure. If None, the figure is not saved.
        return_fig_ax : bool, default False
            If True, returns a tuple (fig, ax) for further customization.
        dual_axis : {"auto", True, False}, default "auto"
            Controls whether to use a second y-axis for variable income:
            - "auto": activates if max(left_scale/right_scale, right_scale/left_scale) >= dual_axis_ratio
            - True: always use dual-axis
            - False: always use single-axis
        dual_axis_ratio : float, default 3.0
            Threshold ratio for automatic dual-axis activation when dual_axis="auto".
            Higher values require larger scale differences to trigger dual-axis.
        show_trajectories : bool, default True
            Whether to display individual Monte Carlo trajectories for stochastic variable income.
            Trajectories are plotted with low alpha for transparency. Only active when
            variable income has non-zero sigma.
        show_confidence_band : bool, default False
            Whether to display confidence intervals as shaded bands (legacy visualization).
            Can be combined with show_trajectories. Only active when variable income
            has non-zero sigma.
        trajectory_alpha : float, default 0.08
            Transparency level for individual trajectories (0=invisible, 1=opaque).
            When combining with confidence bands, reduce this value (e.g., 0.02-0.05)
            for better band visibility.
        confidence : float, default 0.9
            Confidence level for the band (between 0 and 1). Only used if
            show_confidence_band=True. Example: 0.95 produces a 95% confidence interval.
        n_simulations : int, default 500
            Number of Monte Carlo simulations to generate for variable income. Used for
            computing both trajectories and confidence bands. When combining trajectories
            and bands, reduce this value (e.g., 100-200) to prevent visual saturation.
        colors : dict, optional
            Dictionary mapping line colors. Keys: "fixed", "variable", "total".
            Defaults to {"fixed": "blue", "variable": "orange", "total": "black"}.

        Returns
        -------
        None or tuple
            If `return_fig_ax=True`, returns (fig, ax) tuple for further customization.
            In dual-axis mode, `ax` is the left axis; the right axis is created internally.
            Otherwise, returns None.

        Examples
        --------
        >>> from datetime import date
        >>> income = IncomeModel(
        ...     fixed=FixedIncome(base=1_400_000, annual_growth=0.03),
        ...     variable=VariableIncome(base=200_000, sigma=0.15, seed=42)
        ... )

        # Default: Monte Carlo trajectories with auto dual-axis
        >>> income.plot_income(months=24, start=date(2025, 1, 1))

        # Single-axis, no trajectories (deterministic view)
        >>> income.plot_income(
        ...     months=24,
        ...     start=date(2025, 1, 1),
        ...     dual_axis=False,
        ...     show_trajectories=False
        ... )

        # Legacy: confidence bands only
        >>> income.plot_income(
        ...     months=24,
        ...     start=date(2025, 1, 1),
        ...     show_trajectories=False,
        ...     show_confidence_band=True,
        ...     confidence=0.95
        ... )

        # Hybrid: trajectories + bands (adjust parameters for visibility)
        >>> income.plot_income(
        ...     months=36,
        ...     start=date(2025, 1, 1),
        ...     show_trajectories=True,
        ...     show_confidence_band=True,
        ...     n_simulations=150,
        ...     trajectory_alpha=0.03,
        ...     dual_axis=True
        ... )

        # Custom colors and forced dual-axis
        >>> income.plot_income(
        ...     months=24,
        ...     colors={"fixed": "darkgreen", "variable": "purple", "total": "red"},
        ...     dual_axis=True,
        ...     save_path="income_projection.png"
        ... )
        """
        import matplotlib.pyplot as plt
        import numpy as np

        colors = colors or {"fixed": "black", "variable": "gray", "total": "blue"}
        fixed_col, var_col = self.name_fixed, self.name_variable

        # Fixed income: single projection (deterministic, replicated internally if needed)
        if isinstance(self.fixed, list):
             projections = [
                f.project(months=months, start=start, output="array") 
                for f in self.fixed
            ]
             fixed_arr = sum(projections)
        else:
            fixed_arr = self.fixed.project(months=months, start=start, output="array")  # shape: (months,)
        
        # Variable income: vectorized generation
        sims = None
        has_variable = False
        if self.variable is not None:
            if isinstance(self.variable, list):
                has_variable = any(hasattr(v, "sigma") and v.sigma > 0 for v in self.variable)
            else:
                has_variable = hasattr(self.variable, "sigma") and self.variable.sigma > 0

        if (show_trajectories or show_confidence_band) and has_variable:
            if isinstance(self.variable, list):
                projections = []
                for i, v in enumerate(self.variable):
                    # Use unique seeds based on instance to ensure diversity but reproducibility
                    # Ideally we accept a seed in plot(), but here we synthesize
                    s = i * 1234
                    projections.append(v.project(
                        months=months, 
                        start=start, 
                        seed=s,
                        output="array", 
                        n_sims=n_simulations
                    ))
                sims = sum(projections)
            else:
                sims = self.variable.project(
                    months=months, 
                    start=start, 
                    seed=None,  
                    output="array", 
                    n_sims=n_simulations
                )  # shape: (n_simulations, months)
            
            var_mean = sims.mean(axis=0)  # shape: (months,)
            
            if show_confidence_band:
                lower_perc = np.percentile(sims, (1-confidence)/2*100, axis=0)
                upper_perc = np.percentile(sims, (1+confidence)/2*100, axis=0)
            else:
                lower_perc = upper_perc = None
        else:
            # Deterministic or no variable income or no trajectories needed
            if self.variable is not None:
                if isinstance(self.variable, list):
                    projections = [
                        v.project(months=months, start=start, output="array")
                        for v in self.variable
                    ]
                    var_mean = sum(projections)
                else:
                    var_mean = self.variable.project(months=months, start=start, output="array")
            else:
                var_mean = np.zeros(months, dtype=float)
            lower_perc = upper_perc = None

        total_mean = fixed_arr + var_mean
        idx = month_index(start=start, months=max(months, 0))
        
        if len(idx) == 0:
            fig, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data (months <= 0)", ha="center", va="center", transform=ax.transAxes)
            if return_fig_ax: return (ax.figure, ax)
            return

        fig = None
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Determine dual-axis
        use_dual = False
        if dual_axis is True:
            use_dual = True
        elif dual_axis == "auto":
            left_max = max(np.nanmax(fixed_arr), np.nanmax(total_mean))
            right_max = np.nanmax(var_mean)
            use_dual = (max(left_max / right_max, right_max / left_max) >= dual_axis_ratio) if left_max*right_max>0 else False

        lines, labels = [], []
        ax_right = None
        
        if use_dual:
            # --- Dual-axis mode ---
            ax_right = ax.twinx()
            
            # 1. Plot trajectories FIRST (background)
            if show_trajectories and sims is not None:
                for i in range(n_simulations):
                    # Variable trajectories on right axis: sims[i, :] shape (months,)
                    ax_right.plot(idx, sims[i, :], color='gray', alpha=trajectory_alpha, 
                            linewidth=0.8, zorder=1)
                    # Total trajectories on left axis
                    ax.plot(idx, fixed_arr + sims[i, :], color='gray', 
                        alpha=trajectory_alpha*0.7, linewidth=0.8, zorder=1)
            
            # 2. Plot confidence bands (if enabled)
            if show_confidence_band and lower_perc is not None:
                ax_right.fill_between(idx, lower_perc, upper_perc, 
                                color=colors.get("variable", "orange"), 
                                alpha=0.2, zorder=2)
                ax.fill_between(idx, lower_perc + fixed_arr, upper_perc + fixed_arr,
                            color=colors.get("total", "black"), 
                            alpha=0.15, zorder=2)
            
            # 3. Plot mean lines (foreground)
            l_fixed, = ax.plot(idx, fixed_arr, label=fixed_col, 
                            color=colors.get("fixed", "blue"), 
                            linewidth=2.5, zorder=3)
            l_total, = ax.plot(idx, total_mean, label="total", 
                            color=colors.get("total", "black"), 
                            linewidth=2.5, zorder=3)
            l_var, = ax_right.plot(idx, var_mean, linestyle="--", label=var_col, 
                            color=colors.get("variable", "orange"), 
                            linewidth=2.5, zorder=3)
            
            ax.set_ylabel(ylabel_left)
            ax_right.set_ylabel(ylabel_right)
            lines.extend([l_fixed, l_total, l_var])
            labels.extend([fixed_col, "total", var_col])
            
        else:
            # --- Single-axis mode ---
            
            # 1. Plot trajectories FIRST (background)
            if show_trajectories and sims is not None:
                for i in range(n_simulations):
                    ax.plot(idx, sims[i, :], color='gray', alpha=trajectory_alpha, 
                        linewidth=0.8, zorder=1, label='_nolegend_')
                    ax.plot(idx, fixed_arr + sims[i, :], color='gray', 
                        alpha=trajectory_alpha*0.7, linewidth=0.8, 
                        zorder=1, label='_nolegend_')
            
            # 2. Plot confidence bands (if enabled)
            if show_confidence_band and lower_perc is not None:
                ax.fill_between(idx, lower_perc, upper_perc, 
                            color=colors.get("variable", "orange"), 
                            alpha=0.2, label=f"{var_col} CI", 
                            zorder=2)
                ax.fill_between(idx, lower_perc + fixed_arr, upper_perc + fixed_arr,
                            color=colors.get("total", "black"), 
                            alpha=0.15, label="Total CI", 
                            zorder=2)
            
            # 3. Plot mean lines (foreground)
            l_fixed, = ax.plot(idx, fixed_arr, label=fixed_col, 
                            color=colors.get("fixed", "blue"), 
                            linewidth=2.5, zorder=3)
            l_var, = ax.plot(idx, var_mean, label=var_col, 
                            color=colors.get("variable", "orange"), 
                            linestyle="--", linewidth=2.5, zorder=3)
            l_total, = ax.plot(idx, total_mean, label="total", 
                            color=colors.get("total", "black"), 
                            linewidth=2.5, zorder=3)
            
            ax.set_ylabel(ylabel_left)
            lines.extend([l_fixed, l_var, l_total])
            labels.extend([fixed_col, var_col, "total"])

        # ========== Formatting ==========
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        if ax_right is not None:
            ax_right.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        
        if grid: ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        if title: ax.set_title(title)
        ax.set_xlabel("Month")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        if legend: ax.legend(lines, labels, loc="best")

        # Total annotation
        total_fixed_sum = fixed_arr.sum()
        total_var_sum = var_mean.sum()
        total_sum = total_fixed_sum + total_var_sum
        ax.text(0.02, 0.98,
                f"Total Fixed: ${total_fixed_sum:,.0f}".replace(",", ".") + "\n" +
                f"Total Variable: ${total_var_sum:,.0f}".replace(",", ".") + "\n" +
                f"Total Income: ${total_sum:,.0f}".replace(",", "."),
                transform=ax.transAxes, fontsize=9, verticalalignment="top", 
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), zorder=10)

        if save_path: (fig or ax.figure).savefig(save_path, bbox_inches="tight", dpi=150)
        if return_fig_ax: return (fig or ax.figure, ax)

    def plot_contributions(
        self,
        months: int,
        start: Optional[date] = None,
        ax=None,
        figsize: tuple = (12, 6),
        title: str = r"Projected Monthly Contributions $(A_t)$",
        legend: bool = True,
        grid: bool = True,
        ylabel: str = "Total Contribution (CLP)",
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        colors: dict | None = None,
        show_trajectories: bool = True,
        show_confidence_band: bool = False,
        trajectory_alpha: float = 0.08,
        confidence: float = 0.9,
        n_simulations: int = 500,
    ):
        """
        Plot projected monthly contributions with optional Monte Carlo trajectories and confidence bands.

        Visualizes total monthly contributions derived from fixed and variable income streams,
        applying the contribution fractions specified in `self.monthly_contribution`. Supports
        two visualization modes: individual stochastic trajectories (Monte Carlo style) and/or
        statistical confidence intervals.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0. If `months <= 0`, returns an empty
            plot with a "no data" message.
        start : datetime.date, optional
            Start date for the projection. Used to align months with the calendar and
            rotate contribution fractions according to the specified starting month.
            If None, January (month 1) is assumed.
        ax : matplotlib.axes.Axes, optional
            Existing Axes object to plot on. If None, a new figure and axes are created.
        figsize : tuple, default (12, 6)
            Figure size (width, height) in inches when creating a new figure.
        title : str, default "Projected Monthly Contributions"
            Plot title.
        legend : bool, default True
            Whether to display the legend.
        grid : bool, default True
            Whether to display gridlines.
        ylabel : str, default "Total Contribution (CLP)"
            Label for the y-axis.
        save_path : str, optional
            File path to save the figure. If None, the figure is not saved.
        return_fig_ax : bool, default False
            If True, returns a tuple (fig, ax) for further customization.
        colors : dict, optional
            Dictionary mapping line colors. Keys: "total" (mean line), "ci" (confidence band).
            Defaults to {"total": "blue", "ci": "orange"}.
        show_trajectories : bool, default True
            Whether to display individual Monte Carlo trajectories for stochastic contributions.
            Trajectories are plotted with low alpha for transparency. Only active when
            variable income has non-zero sigma.
        show_confidence_band : bool, default False
            Whether to display confidence intervals as shaded bands (legacy visualization).
            Can be combined with show_trajectories. Only active when variable income
            has non-zero sigma.
        trajectory_alpha : float, default 0.08
            Transparency level for individual trajectories (0=invisible, 1=opaque).
            When combining with confidence bands, reduce this value (e.g., 0.02-0.05)
            for better band visibility.
        confidence : float, default 0.9
            Confidence level for the band (between 0 and 1). Only used if
            show_confidence_band=True. Example: 0.95 produces a 95% confidence interval.
        n_simulations : int, default 500
            Number of Monte Carlo simulations to generate. Used for computing both
            trajectories and confidence bands. When combining trajectories and bands,
            reduce this value (e.g., 100-200) to prevent visual saturation.

        Returns
        -------
        None or tuple
            If `return_fig_ax=True`, returns (fig, ax) tuple for further customization.
            Otherwise, returns None.

        Examples
        --------
        >>> from datetime import date
        >>> income = IncomeModel(
        ...     fixed=FixedIncome(base=1_400_000, annual_growth=0.02),
        ...     variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
        ... )
        >>> income.monthly_contribution = {"fixed": [0.3]*12, "variable": [1.0]*12}

        # Default: Monte Carlo trajectories only
        >>> income.plot_contributions(months=24, start=date(2025, 1, 1))

        # Legacy: confidence band only
        >>> income.plot_contributions(
        ...     months=24,
        ...     start=date(2025, 1, 1),
        ...     show_trajectories=False,
        ...     show_confidence_band=True
        ... )

        # Hybrid: trajectories + bands (adjust parameters for visibility)
        >>> income.plot_contributions(
        ...     months=24,
        ...     start=date(2025, 1, 1),
        ...     show_trajectories=True,
        ...     show_confidence_band=True,
        ...     n_simulations=150,
        ...     trajectory_alpha=0.03,
        ...     confidence=0.95
        ... )

        # Custom colors and save
        >>> income.plot_contributions(
        ...     months=36,
        ...     start=date(2025, 1, 1),
        ...     colors={"total": "green", "ci": "lightblue"},
        ...     save_path="contributions_projection.png"
        ... )
        """
        import matplotlib.pyplot as plt
        import numpy as np

        colors = colors or {"total": "blue", "ci": "orange"}
        idx = month_index(start=start, months=max(months, 0))
        
        if months <= 0 or len(idx) == 0:
            fig, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data (months <= 0)", ha="center", va="center", transform=ax.transAxes)
            if return_fig_ax: return (ax.figure, ax)
            return

        # Generate simulations if needed (VECTORIZED)
        sims = None
        has_variable = self.variable is not None and hasattr(self.variable, "sigma") and self.variable.sigma > 0
        if (show_trajectories or show_confidence_band) and has_variable:
            sims = self.contributions(
                months, 
                start=start, 
                seed=None, 
                output="array", 
                n_sims=n_simulations
            )  # shape: (n_simulations, months)
            
            contrib_mean = sims.mean(axis=0)  # shape: (months,)
            
            if show_confidence_band:
                lower = np.percentile(sims, (1-confidence)/2*100, axis=0)
                upper = np.percentile(sims, (1+confidence)/2*100, axis=0)
            else:
                lower = upper = None
        else:
            # Deterministic case
            contrib_mean = self.contributions(months, start=start, output="array")
            lower = upper = None

        fig = None
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # 1. Plot trajectories FIRST (background)
        if show_trajectories and sims is not None:
            for i in range(n_simulations):
                ax.plot(idx, sims[i, :], color='gray', alpha=trajectory_alpha, 
                    linewidth=0.8, zorder=1, label='_nolegend_')
        
        # 2. Plot confidence band (if enabled)
        if show_confidence_band and lower is not None:
            ax.fill_between(idx, lower, upper, color=colors["ci"], alpha=0.2,
                        label=f"{int(confidence*100)}% CI", zorder=2)
        
        # 3. Plot mean line (foreground)
        ax.plot(idx, contrib_mean, color=colors["total"], label="Mean Contribution", 
            linewidth=2.5, zorder=3)

        # ========== Formatting ==========
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if grid: ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        if legend: ax.legend(loc="best")

        # Total annotation
        total_contrib = contrib_mean.sum()
        ax.text(0.02, 0.98,
                f"Total Contributions: ${total_contrib:,.0f}".replace(",", "."),
                transform=ax.transAxes, fontsize=9, verticalalignment="top", 
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), zorder=10)

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
        show_trajectories: bool = True,
        show_confidence_band: bool = False,
        trajectory_alpha: float = 0.08,
        confidence: float = 0.9,
        n_simulations: int = 500,
        colors: dict | None = None,
    ):
        """
        Unified plot wrapper to call either `plot_income` or `plot_contributions`.
        
        Parameters
        ----------
        mode : str
            "income" to plot income streams, "contributions" to plot contributions.
        show_trajectories : bool, default True
            Whether to display individual Monte Carlo trajectories for stochastic components.
        trajectory_alpha : float, default 0.08
            Transparency level for individual trajectories (0=invisible, 1=opaque).
        show_confidence_band : bool, default False
            Whether to display confidence intervals as shaded bands (legacy option).
            Can be combined with show_trajectories.
        n_simulations : int, default 500
            Number of Monte Carlo simulations. When combining trajectories and bands,
            reduce this value (e.g., 100-200) for better visibility.
        dual_axis : str | bool, default "auto"
            Only applies to mode="income". Controls second y-axis for variable income.
        dual_axis_ratio : float, default 3.0
            Only applies to mode="income". Threshold ratio for auto dual-axis activation.
        ylabel_left : str, optional
            Left y-axis label. Defaults depend on mode.
        ylabel_right : str, optional
            Right y-axis label (only for dual-axis income plots).
        
        Other parameters are passed directly to the corresponding plotting method.
        
        Returns
        -------
        None or tuple
            If return_fig_ax=True, returns (fig, ax). Otherwise, returns None.
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
                show_trajectories=show_trajectories,
                show_confidence_band=show_confidence_band,
                trajectory_alpha=trajectory_alpha,
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
                ylabel=ylabel_left or "Total Contribution (CLP)",
                save_path=save_path,
                return_fig_ax=return_fig_ax,
                colors=colors,
                show_trajectories=show_trajectories,
                show_confidence_band=show_confidence_band,
                trajectory_alpha=trajectory_alpha,
                confidence=confidence,
                n_simulations=n_simulations,
            )
        else:
            raise ValueError("Invalid mode. Must be 'income' or 'contributions'.")

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
        df = self.project(months=months, start=start, output="dataframe")
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
        result = {
            "fixed": {
                "base": self.fixed.base,
                "annual_growth": self.fixed.annual_growth,
                "salary_raises": None if self.fixed.salary_raises is None 
                                else {d.isoformat(): v for d, v in self.fixed.salary_raises.items()},
                "name": self.fixed.name,
            },
        }
        if self.variable is not None:
            result["variable"] = {
                "base": self.variable.base,
                "seasonality": None if self.variable.seasonality is None else list(self.variable.seasonality),
                "sigma": self.variable.sigma,
                "floor": self.variable.floor,
                "cap": self.variable.cap,
                "annual_growth": self.variable.annual_growth,
                "name": self.variable.name,
                "seed": self.variable.seed,
            }
        return result

    @classmethod
    def from_dict(cls, payload: dict) -> "IncomeModel":
        fx = payload.get("fixed", {})
        vr = payload.get("variable")
        fixed = FixedIncome(
            base=float(fx.get("base", 0.0)),
            annual_growth=float(fx.get("annual_growth", 0.0)),
            salary_raises=None if fx.get("salary_raises") is None else {
                date.fromisoformat(k): float(v) for k, v in fx["salary_raises"].items()
            },
            name=str(fx.get("name", "fixed")),
        )
        variable = None
        if vr is not None:
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
            metrics = self.income_metrics(months=12, start=date(2025, 1, 1))
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
    df = model.project(months=months, start=start_date, output="dataframe")
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

    # Test vectorization
    print("\n=== Vectorization tests ===")

    # Multiple simulations for variable income
    print("Testing n_sims > 1:")
    sims = model.variable.project(months=6, n_sims=100, output="array")
    print(f"Variable sims shape: {sims.shape}")  # Expected: (100, 6)
    assert sims.shape == (100, 6), "Shape mismatch for variable income"

    # Multiple simulations for contributions
    contrib_sims = model.contributions(months=6, start=start_date, output="array", n_sims=100)
    print(f"Contributions sims shape: {contrib_sims.shape}")  # Expected: (100, 6)
    assert contrib_sims.shape == (100, 6), "Shape mismatch for contributions"

    # Verify independence of simulations
    assert not np.allclose(sims[0, :], sims[1, :]), "Simulations should be independent"
    print("✓ All vectorization tests passed")

    # Performance comparison (optional)
    print("\n=== Performance comparison ===")
    import time

    # Old way (simulated)
    start_time = time.time()
    old_way = np.array([model.variable.project(6, seed=None, output="array") for _ in range(100)])
    old_time = time.time() - start_time

    # New way
    start_time = time.time()
    new_way = model.variable.project(6, n_sims=100, output="array")
    new_time = time.time() - start_time

    print(f"Old way (loop): {old_time*1000:.2f} ms")
    print(f"New way (vectorized): {new_time*1000:.2f} ms")
    print(f"Speedup: {old_time/new_time:.1f}×")