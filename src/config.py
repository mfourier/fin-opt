"""
Configuration management module for FinOpt.

Purpose
-------
Centralized configuration using Pydantic models for type-safe parameter management,
validation, and serialization. Supports environment variables, JSON/YAML configs,
and programmatic defaults.

Design Principles
-----------------
- Type-safe: Pydantic enforces types and validates ranges
- Immutable: Frozen models prevent accidental mutation
- Serializable: Easy conversion to/from JSON/YAML for config files
- Environment-aware: Supports .env files for sensitive parameters
- Defaults: Sensible defaults for all parameters

Example
-------
>>> from finopt.config import SimulationConfig, OptimizationConfig
>>> sim_config = SimulationConfig(n_sims=1000, seed=42)
>>> opt_config = OptimizationConfig(T_max=120, solver="ECOS")
>>>
>>> # Serialize to dict/JSON
>>> config_dict = sim_config.model_dump()
>>> json_str = sim_config.model_dump_json()
>>>
>>> # Load from dict/JSON
>>> loaded = SimulationConfig.model_validate(config_dict)
>>> loaded_json = SimulationConfig.model_validate_json(json_str)
"""

from __future__ import annotations
from typing import Optional, Literal, List
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

__all__ = [
    "SimulationConfig",
    "OptimizationConfig",
    "IncomeConfig",
    "AccountConfig",
    "AppSettings",
]


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """
    Configuration for Monte Carlo simulation parameters.

    Attributes
    ----------
    n_sims : int
        Number of Monte Carlo scenarios (100-10,000).
    seed : int, optional
        Random seed for reproducibility. If None, uses random seed.
    cache_enabled : bool
        Enable SHA256-based caching of simulation results.
    verbose : bool
        Print progress messages during simulation.

    Examples
    --------
    >>> config = SimulationConfig(n_sims=500, seed=42, cache_enabled=True)
    >>> config.n_sims
    500
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_sims: int = Field(
        default=500,
        ge=100,
        le=10_000,
        description="Number of Monte Carlo scenarios"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable simulation result caching"
    )
    verbose: bool = Field(
        default=True,
        description="Print progress messages"
    )


# ---------------------------------------------------------------------------
# Optimization Configuration
# ---------------------------------------------------------------------------

class OptimizationConfig(BaseModel):
    """
    Configuration for portfolio optimization parameters.

    Attributes
    ----------
    T_max : int
        Maximum horizon to search (12-600 months).
    T_min : int
        Minimum horizon to start search (1-T_max).
    solver : str
        CVXPY solver backend: "ECOS", "SCS", "CLARABEL", "MOSEK".
    objective : str
        Optimization objective: "risky", "balanced", "conservative", "risky_turnover".
    search_strategy : str
        Horizon search method: "binary" or "linear".
    tolerance : float
        Constraint violation tolerance (1e-6 to 1e-2).
    verbose : bool
        Print solver progress messages.
    warm_start : bool
        Use warm-start from previous horizon in search.
    max_iterations : int
        Maximum solver iterations (for iterative solvers).

    Examples
    --------
    >>> config = OptimizationConfig(T_max=120, solver="ECOS", objective="balanced")
    >>> config.T_max
    120
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    T_max: int = Field(
        default=240,
        ge=12,
        le=600,
        description="Maximum optimization horizon (months)"
    )
    T_min: int = Field(
        default=12,
        ge=1,
        description="Minimum optimization horizon (months)"
    )
    solver: Literal["ECOS", "SCS", "CLARABEL", "MOSEK"] = Field(
        default="ECOS",
        description="CVXPY solver backend"
    )
    objective: Literal["risky", "balanced", "conservative", "risky_turnover"] = Field(
        default="balanced",
        description="Optimization objective function"
    )
    search_strategy: Literal["binary", "linear"] = Field(
        default="binary",
        description="Horizon search strategy"
    )
    tolerance: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Constraint violation tolerance"
    )
    verbose: bool = Field(
        default=True,
        description="Print solver progress"
    )
    warm_start: bool = Field(
        default=True,
        description="Enable warm-start between horizon searches"
    )
    max_iterations: int = Field(
        default=1000,
        ge=10,
        le=10_000,
        description="Maximum solver iterations"
    )

    @field_validator("T_min")
    @classmethod
    def validate_T_min(cls, v, info):
        """Ensure T_min <= T_max."""
        T_max = info.data.get("T_max", 240)
        if v > T_max:
            raise ValueError(f"T_min ({v}) must be <= T_max ({T_max})")
        return v


# ---------------------------------------------------------------------------
# Income Configuration
# ---------------------------------------------------------------------------

class FixedIncomeConfig(BaseModel):
    """Configuration for fixed income stream (salary)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base: float = Field(
        ge=0,
        description="Base monthly income amount"
    )
    annual_growth: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Annual growth rate (e.g., 0.03 for 3%)"
    )
    salary_raises: Optional[dict] = Field(
        default=None,
        description="Dict of date to raise amounts for scheduled raises"
    )


class VariableIncomeConfig(BaseModel):
    """Configuration for variable income stream (bonuses, freelancing)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base: float = Field(
        ge=0,
        description="Base monthly income amount"
    )
    sigma: float = Field(
        default=0.0,
        ge=0,
        le=2.0,
        description="Volatility (standard deviation as fraction of base)"
    )
    annual_growth: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Annual growth rate"
    )
    seasonality: Optional[List[float]] = Field(
        default=None,
        description="12-month seasonality factors (must sum to 12)"
    )
    floor: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum income floor (absolute value)"
    )
    cap: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum income cap (absolute value)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    @field_validator("seasonality")
    @classmethod
    def validate_seasonality(cls, v):
        """Ensure seasonality has 12 factors summing to 12."""
        if v is not None:
            if len(v) != 12:
                raise ValueError(f"Seasonality must have 12 factors, got {len(v)}")
            total = sum(v)
            if not (11.9 <= total <= 12.1):  # Allow small floating point error
                raise ValueError(f"Seasonality factors must sum to 12, got {total:.2f}")
        return v


class IncomeConfig(BaseModel):
    """Combined configuration for fixed + variable income."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixed: Optional[FixedIncomeConfig] = Field(
        default=None,
        description="Fixed income configuration"
    )
    variable: Optional[VariableIncomeConfig] = Field(
        default=None,
        description="Variable income configuration"
    )
    contribution_rate_fixed: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Fraction of fixed income contributed to portfolio"
    )
    contribution_rate_variable: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Fraction of variable income contributed to portfolio"
    )


# ---------------------------------------------------------------------------
# Account Configuration
# ---------------------------------------------------------------------------

class AccountConfig(BaseModel):
    """
    Configuration for portfolio account with return/volatility parameters.

    Attributes
    ----------
    name : str
        Account identifier (e.g., "Conservative", "Aggressive")
    annual_return : float
        Expected annual return (e.g., 0.08 for 8%)
    annual_volatility : float
        Annual return volatility (standard deviation)
    initial_wealth : float
        Starting account balance

    Examples
    --------
    >>> account = AccountConfig(
    ...     name="Emergency",
    ...     annual_return=0.04,
    ...     annual_volatility=0.05,
    ...     initial_wealth=1_000_000
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=50,
        description="Account name/identifier"
    )
    annual_return: float = Field(
        ge=-0.5,
        le=1.0,
        description="Expected annual return"
    )
    annual_volatility: float = Field(
        ge=0,
        le=2.0,
        description="Annual return volatility"
    )
    initial_wealth: float = Field(
        default=0,
        ge=0,
        description="Initial account balance"
    )


# ---------------------------------------------------------------------------
# Application Settings (Environment Variables)
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """
    Global application settings loaded from environment variables.

    Supports .env files for local development. Environment variables
    should be prefixed with FINOPT_ (e.g., FINOPT_DEBUG=true).

    Attributes
    ----------
    debug : bool
        Enable debug mode with verbose logging
    log_level : str
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
    cache_dir : Path
        Directory for caching simulation results
    max_cache_size_mb : int
        Maximum cache size in megabytes

    Examples
    --------
    >>> settings = AppSettings()
    >>> settings.debug
    False

    # With .env file:
    # FINOPT_DEBUG=true
    # FINOPT_CACHE_DIR=/tmp/finopt-cache
    >>> settings = AppSettings(_env_file=".env")
    >>> settings.debug
    True
    """

    model_config = ConfigDict(
        env_prefix="FINOPT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "finopt",
        description="Cache directory for simulation results"
    )
    max_cache_size_mb: int = Field(
        default=500,
        ge=10,
        le=10_000,
        description="Maximum cache size in MB"
    )
    enable_telemetry: bool = Field(
        default=False,
        description="Enable anonymous usage telemetry"
    )
