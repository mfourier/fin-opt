"""Income modeling module for FinOpt

Key concepts
------------
- FixedIncome: deterministic monthly base with optional growth.
- VariableIncome: monthly base with seasonal pattern and noise.
- IncomeModel: orchestrates multiple streams and produces a monthly
  projection as a pandas Series/DataFrame with a DatetimeIndex.

Design goals
------------
- Deterministic by default, reproducible when stochasticity is used
  (via numpy.random.Generator with explicit seed).
- Reuses shared helpers from utils.py for validation, rate conversions
  and index construction.
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
]


# ---------------------------------------------------------------------------
# Income Streams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedIncome:
    """Deterministic fixed income stream.

    Parameters
    ----------
    base : float
        Monthly base amount.
    annual_growth : float, default 0.0
        Annual nominal growth (e.g., 0.05 for +5%/year). Converted to an
        equivalent monthly rate internally.
    name : str, optional
        Identifier used in outputs.
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
    """Variable income with optional seasonality and Gaussian noise.

    Parameters
    ----------
    base : float
        Monthly base before seasonality and noise.
    seasonality : Optional[Iterable[float]]
        Length-12 multiplicative factors for Jan..Dec (1.0 = neutral).
        If None, seasonality is neutral.
    sigma : float, default 0.0
        Std. dev. of Gaussian noise as *fraction* of the month mean.
    floor : float, optional
        Lower bound (after noise). If None, no floor is applied.
    cap : float, optional
        Upper bound (after noise). If None, no cap is applied.
    annual_growth : float, default 0.0
        Annual growth applied to the base before seasonality.
    name : str, default "variable"
        Identifier used in outputs.
    seed : Optional[int]
        RNG seed for reproducible noise.
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

    def _seasonal_factor(self, month_index_: int) -> float:
        if self.seasonality is None:
            return 1.0
        # month_index_: 0..11 repeating
        return float(tuple(self.seasonality)[month_index_ % 12])

    def project(self, months: int) -> np.ndarray:
        if months <= 0:
            return np.zeros(0, dtype=float)
        rng = np.random.default_rng(self.seed)
        m = annual_to_monthly(self.annual_growth)
        t = np.arange(months, dtype=float)
        base_path = self.base * np.power(1.0 + m, t)

        means = np.empty(months, dtype=float)
        for k in range(months):
            means[k] = base_path[k] * self._seasonal_factor(k)

        if self.sigma == 0.0:
            noisy = means
        else:
            noise = rng.normal(loc=0.0, scale=self.sigma, size=months)
            noisy = means * (1.0 + noise)

        if self.floor is not None:
            noisy = np.maximum(noisy, self.floor)
        if self.cap is not None:
            noisy = np.minimum(noisy, self.cap)

        # Ensure non-negative income
        return np.maximum(noisy, 0.0)


# ---------------------------------------------------------------------------
# Income Model (Facade)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncomeModel:
    """Combines fixed and variable income streams and projects monthly totals.

    Notes
    -----
    - The model is deterministic unless `VariableIncome.sigma > 0`.
    - `start` is used only to create a friendly DatetimeIndex; numerical
      projections do not depend on calendar specifics.
    """

    fixed: FixedIncome
    variable: VariableIncome
    name_fixed: str = field(default="fixed", init=True)
    name_variable: str = field(default="variable", init=True)

    def project_monthly(
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

        fixed_path = self.fixed.project(months)
        variable_path = self.variable.project(months)
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

    def contributions_from_proportions(
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
        fixed_path = self.fixed.project(months)
        variable_path = self.variable.project(months)
        contrib = alpha_fixed * fixed_path + beta_variable * variable_path
        contrib = np.maximum(contrib, 0.0)
        return pd.Series(contrib, index=idx, name="contribution")

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
    model = IncomeModel(
        fixed=FixedIncome(base=1_400_000.0, annual_growth=0.0),
        variable=VariableIncome(base=200_000.0, sigma=0.0),
    )
    s = model.project_monthly(months=6, start=date(2025, 9, 1), as_dataframe=True)
    print(s)
    contrib = model.contributions_from_proportions(
        months=6, alpha_fixed=0.35, beta_variable=1.0, start=date(2025, 9, 1)
    )
    print(contrib)
