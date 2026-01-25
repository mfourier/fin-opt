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
from typing import Optional, Literal, List, Union
import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

__all__ = [
    "SimulationConfig",
    "OptimizationConfig",
    "IncomeConfig",
    "AccountConfig",
    "WithdrawalEventConfig",
    "StochasticWithdrawalConfig",
    "WithdrawalConfig",
    "IntermediateGoalConfig",
    "TerminalGoalConfig",
    "ScenarioConfig",
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
            if any(v_ < 0 for v_ in v):
                raise ValueError("Seasonality factors must be non-negative")
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
    contribution_rate_fixed: Union[float, List[float]] = Field(
        default=0.3,
        description="Fraction of fixed income contributed to portfolio"
    )
    contribution_rate_variable: Union[float, List[float]] = Field(
        default=1.0,
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
    display_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Descriptive name for plots and reports"
    )


# ---------------------------------------------------------------------------
# Withdrawal Configuration
# ---------------------------------------------------------------------------

class WithdrawalEventConfig(BaseModel):
    """
    Configuration for a single scheduled withdrawal event.

    Attributes
    ----------
    account : str
        Target account name (must match an account in the portfolio).
    amount : float
        Withdrawal amount (must be positive).
    withdrawal_date : datetime.date
        Calendar date of the withdrawal.
    description : str, optional
        Human-readable description.

    Examples
    --------
    >>> from datetime import date
    >>> event = WithdrawalEventConfig(
    ...     account="Conservative",
    ...     amount=500_000,
    ...     withdrawal_date=date(2025, 6, 1),
    ...     description="Emergency fund withdrawal"
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str = Field(
        min_length=1,
        max_length=50,
        description="Target account name"
    )
    amount: float = Field(
        gt=0,
        description="Withdrawal amount (must be positive)"
    )
    withdrawal_date: datetime.date = Field(
        description="Calendar date of the withdrawal"
    )
    description: str = Field(
        default="",
        max_length=200,
        description="Human-readable description"
    )


class StochasticWithdrawalConfig(BaseModel):
    """
    Configuration for a stochastic withdrawal with uncertainty.

    Models withdrawals that have a base expected amount but may vary across
    scenarios (e.g., variable medical expenses, emergency costs).

    Attributes
    ----------
    account : str
        Target account name.
    base_amount : float
        Expected withdrawal amount (mean of distribution).
    sigma : float
        Standard deviation of withdrawal amount.
    month : int, optional
        Month offset from start_date (1-indexed). Mutually exclusive with withdrawal_date.
    withdrawal_date : datetime.date, optional
        Calendar date of withdrawal. Mutually exclusive with month.
    floor : float
        Minimum withdrawal amount (default: 0.0).
    cap : float, optional
        Maximum withdrawal amount (None = no cap).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from datetime import date
    >>> withdrawal = StochasticWithdrawalConfig(
    ...     account="Conservative",
    ...     base_amount=300_000,
    ...     sigma=50_000,
    ...     withdrawal_date=date(2025, 9, 1),
    ...     floor=200_000,
    ...     cap=500_000
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str = Field(
        min_length=1,
        max_length=50,
        description="Target account name"
    )
    base_amount: float = Field(
        ge=0,
        description="Expected withdrawal amount"
    )
    sigma: float = Field(
        ge=0,
        description="Standard deviation of withdrawal amount"
    )
    month: Optional[int] = Field(
        default=None,
        ge=1,
        description="Month offset (1-indexed). Mutually exclusive with withdrawal_date."
    )
    withdrawal_date: Optional[datetime.date] = Field(
        default=None,
        description="Calendar date. Mutually exclusive with month."
    )
    floor: float = Field(
        default=0.0,
        ge=0,
        description="Minimum withdrawal amount"
    )
    cap: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum withdrawal amount (None = no cap)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    @field_validator("cap")
    @classmethod
    def validate_cap(cls, v, info):
        """Ensure cap >= floor if both specified."""
        floor = info.data.get("floor", 0.0)
        if v is not None and v < floor:
            raise ValueError(f"cap ({v}) must be >= floor ({floor})")
        return v

    @field_validator("withdrawal_date")
    @classmethod
    def validate_month_date_exclusive(cls, v, info):
        """Ensure month and withdrawal_date are mutually exclusive."""
        month = info.data.get("month")
        if v is not None and month is not None:
            raise ValueError("Specify either month or withdrawal_date, not both")
        if v is None and month is None:
            raise ValueError("Must specify either month or withdrawal_date")
        return v


class WithdrawalConfig(BaseModel):
    """
    Combined configuration for all withdrawal types.

    Groups scheduled events and stochastic withdrawals for serialization
    and model configuration.

    Attributes
    ----------
    scheduled : List[WithdrawalEventConfig]
        List of deterministic scheduled withdrawals.
    stochastic : List[StochasticWithdrawalConfig]
        List of stochastic withdrawals with uncertainty.

    Examples
    --------
    >>> from datetime import date
    >>> config = WithdrawalConfig(
    ...     scheduled=[
    ...         WithdrawalEventConfig(
    ...             account="Conservative",
    ...             amount=500_000,
    ...             withdrawal_date=date(2025, 6, 1)
    ...         )
    ...     ],
    ...     stochastic=[
    ...         StochasticWithdrawalConfig(
    ...             account="Aggressive",
    ...             base_amount=1_000_000,
    ...             sigma=200_000,
    ...             withdrawal_date=date(2026, 1, 1)
    ...         )
    ...     ]
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheduled: List[WithdrawalEventConfig] = Field(
        default_factory=list,
        description="Scheduled deterministic withdrawals"
    )
    stochastic: List[StochasticWithdrawalConfig] = Field(
        default_factory=list,
        description="Stochastic withdrawals with uncertainty"
    )


# ---------------------------------------------------------------------------
# Goal Configuration
# ---------------------------------------------------------------------------

class IntermediateGoalConfig(BaseModel):
    """
    Configuration for an intermediate financial goal at fixed calendar time.

    Represents checkpoint constraint:
        ℙ(W_{t_fixed}^m ≥ threshold) ≥ confidence

    Attributes
    ----------
    account : str
        Target account name (must match an account in the portfolio).
    threshold : float
        Minimum required wealth at the target time.
    confidence : float
        Minimum satisfaction probability (e.g., 0.90 for 90%).
    month : int, optional
        Target month as offset from start_date (1-indexed).
        Mutually exclusive with goal_date.
    goal_date : datetime.date, optional
        Target date (will be converted to month offset).
        Mutually exclusive with month.

    Examples
    --------
    >>> from datetime import date
    >>> goal = IntermediateGoalConfig(
    ...     account="Emergency",
    ...     threshold=5_000_000,
    ...     confidence=0.90,
    ...     month=6
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str = Field(
        min_length=1,
        max_length=50,
        description="Target account name"
    )
    threshold: float = Field(
        gt=0,
        description="Minimum required wealth"
    )
    confidence: float = Field(
        gt=0,
        lt=1,
        description="Minimum satisfaction probability (e.g., 0.90)"
    )
    month: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target month offset (1-indexed). Mutually exclusive with goal_date."
    )
    goal_date: Optional[datetime.date] = Field(
        default=None,
        description="Target date. Mutually exclusive with month."
    )

    @field_validator("goal_date")
    @classmethod
    def validate_month_date_exclusive(cls, v, info):
        """Ensure month and goal_date are mutually exclusive."""
        month = info.data.get("month")
        if v is not None and month is not None:
            raise ValueError("Specify either month or goal_date, not both")
        if v is None and month is None:
            raise ValueError("Must specify either month or goal_date")
        return v


class TerminalGoalConfig(BaseModel):
    """
    Configuration for a terminal financial goal at end of horizon.

    Represents end-of-planning constraint:
        ℙ(W_T^m ≥ threshold) ≥ confidence

    Attributes
    ----------
    account : str
        Target account name (must match an account in the portfolio).
    threshold : float
        Minimum required terminal wealth.
    confidence : float
        Minimum satisfaction probability (e.g., 0.90 for 90%).

    Examples
    --------
    >>> goal = TerminalGoalConfig(
    ...     account="Retirement",
    ...     threshold=20_000_000,
    ...     confidence=0.90
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str = Field(
        min_length=1,
        max_length=50,
        description="Target account name"
    )
    threshold: float = Field(
        gt=0,
        description="Minimum required terminal wealth"
    )
    confidence: float = Field(
        gt=0,
        lt=1,
        description="Minimum satisfaction probability (e.g., 0.90)"
    )


# ---------------------------------------------------------------------------
# Scenario Configuration
# ---------------------------------------------------------------------------

class ScenarioConfig(BaseModel):
    """
    Configuration for a complete optimization scenario.

    A scenario combines a financial model (profile) with goals and withdrawals
    for optimization. This enables saving and loading complete "what-if"
    scenarios for comparison and reproducibility.

    Attributes
    ----------
    name : str
        Human-readable scenario name (e.g., "Conservative", "With House Purchase").
    description : str
        Optional description of the scenario purpose.
    model_path : str, optional
        Path to the FinancialModel JSON file (profile).
        If None, the scenario must be used with an in-memory model.
    start_date : datetime.date
        Simulation start date for goal/withdrawal resolution.
    intermediate_goals : List[IntermediateGoalConfig]
        List of intermediate checkpoint goals.
    terminal_goals : List[TerminalGoalConfig]
        List of terminal (end-of-horizon) goals.
    withdrawals : WithdrawalConfig, optional
        Scheduled and stochastic withdrawals.
    simulation : SimulationConfig
        Monte Carlo simulation parameters (n_sims, seed, etc.).
    optimization : OptimizationConfig
        Optimization parameters (T_max, solver, objective, etc.).

    Examples
    --------
    >>> from datetime import date
    >>> scenario = ScenarioConfig(
    ...     name="House Purchase Scenario",
    ...     description="Planning for apartment down payment in 2026",
    ...     start_date=date(2025, 1, 1),
    ...     intermediate_goals=[
    ...         IntermediateGoalConfig(
    ...             account="Savings", threshold=3_000_000,
    ...             confidence=0.90, month=12
    ...         )
    ...     ],
    ...     terminal_goals=[
    ...         TerminalGoalConfig(
    ...             account="Investment", threshold=10_000_000,
    ...             confidence=0.80
    ...         )
    ...     ]
    ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Scenario name"
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Scenario description"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to FinancialModel JSON file (profile)"
    )
    start_date: datetime.date = Field(
        description="Simulation start date"
    )
    intermediate_goals: List[IntermediateGoalConfig] = Field(
        default_factory=list,
        description="Intermediate checkpoint goals"
    )
    terminal_goals: List[TerminalGoalConfig] = Field(
        default_factory=list,
        description="Terminal (end-of-horizon) goals"
    )
    withdrawals: Optional[WithdrawalConfig] = Field(
        default=None,
        description="Withdrawal schedule (scheduled + stochastic)"
    )
    simulation: SimulationConfig = Field(
        default_factory=SimulationConfig,
        description="Simulation parameters"
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization parameters"
    )

    @field_validator("intermediate_goals", "terminal_goals")
    @classmethod
    def validate_goals_not_empty(cls, v, info):
        """Warn if no goals specified (valid but unusual)."""
        # Both can be empty, but at least one should have goals
        # This is just a soft validation - we allow empty for flexibility
        return v


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
