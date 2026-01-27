"""
Serialization module for FinOpt model persistence.

Purpose
-------
Provides JSON/YAML serialization and deserialization for FinOpt models,
enabling configuration persistence, sharing, and version control.

Supports serialization of:
- IncomeModel (fixed + variable streams)
- Account configurations
- Portfolio return correlations
- Optimization results
- Full FinancialModel configurations
- Goals (IntermediateGoal, TerminalGoal)
- Withdrawals (WithdrawalModel)
- Complete scenarios (model + goals + withdrawals + parameters)

Design Principles
-----------------
- Type-safe: Uses Pydantic configs for validation
- Human-readable: JSON/YAML formats for easy editing
- Reproducible: Includes seeds and all parameters
- Modular: Can serialize individual components or full models
- Backward compatible: Validates schema versions

Example
-------
>>> from finopt import FinancialModel, Account, IncomeModel, FixedIncome
>>> from finopt.serialization import save_model, load_model
>>> from pathlib import Path
>>>
>>> # Create model
>>> income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
>>> accounts = [Account.from_annual("Emergency", 0.04, 0.05)]
>>> model = FinancialModel(income, accounts)
>>>
>>> # Save to JSON
>>> save_model(model, Path("config.json"))
>>>
>>> # Load from JSON
>>> loaded_model = load_model(Path("config.json"))
"""

from __future__ import annotations
from typing import Dict, Any, List, Union, Optional, Literal, TYPE_CHECKING
from pathlib import Path
from datetime import date
import json
import warnings

import numpy as np

from .config import (
    AccountConfig,
    IncomeConfig,
    FixedIncomeConfig,
    VariableIncomeConfig,
    WithdrawalEventConfig,
    StochasticWithdrawalConfig,
    WithdrawalConfig,
    IntermediateGoalConfig,
    TerminalGoalConfig,
    ScenarioConfig,
    SimulationConfig,
    OptimizationConfig,
)

if TYPE_CHECKING:
    from .model import FinancialModel
    from .portfolio import Account
    from .income import IncomeModel, FixedIncome, VariableIncome
    from .optimization import OptimizationResult
    from .withdrawal import WithdrawalEvent, StochasticWithdrawal, WithdrawalModel, WithdrawalSchedule
    from .goals import IntermediateGoal, TerminalGoal

__all__ = [
    "save_model",
    "load_model",
    "save_optimization_result",
    "load_optimization_result",
    "save_scenario",
    "load_scenario",
    "account_to_dict",
    "account_from_dict",
    "income_to_dict",
    "income_from_dict",
    "withdrawal_to_dict",
    "withdrawal_from_dict",
    "goals_to_dict",
    "goals_from_dict",
    "SCHEMA_VERSION",
]


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# Account Serialization
# ---------------------------------------------------------------------------

def account_to_dict(account: Account) -> Dict[str, Any]:
    """
    Convert Account to dictionary representation.

    Parameters
    ----------
    account : Account
        Account instance to serialize

    Returns
    -------
    dict
        Dictionary with account configuration
    """
    result = {
        "name": account.name,
        "annual_return": account.annual_params["return"],
        "annual_volatility": account.annual_params["volatility"],
        "initial_wealth": account.initial_wealth,
    }
    if account.display_name:
        result["display_name"] = account.display_name
    return result


def account_from_dict(data: Dict[str, Any]) -> Account:
    """
    Create Account from dictionary representation.

    Parameters
    ----------
    data : dict
        Dictionary with account configuration

    Returns
    -------
    Account
        Reconstructed account instance
    """
    from .portfolio import Account

    # Validate using Pydantic config
    config = AccountConfig.model_validate(data)

    return Account.from_annual(
        name=config.name,
        annual_return=config.annual_return,
        annual_volatility=config.annual_volatility,
        initial_wealth=config.initial_wealth,
        display_name=data.get("display_name"),
    )


# ---------------------------------------------------------------------------
# Income Serialization
# ---------------------------------------------------------------------------

def income_to_dict(income_model: IncomeModel) -> Dict[str, Any]:
    """
    Convert IncomeModel to dictionary representation.

    Parameters
    ----------
    income_model : IncomeModel
        Income model instance to serialize

    Returns
    -------
    dict
        Dictionary with income configuration
    """
    result: Dict[str, Any] = {}

    # Fixed income
    if income_model.fixed is not None:
        fixed_data = {
            "base": income_model.fixed.base,
            "annual_growth": income_model.fixed.annual_growth,
        }
        if income_model.fixed.salary_raises:
            # Convert date keys to strings for JSON serializability
            fixed_data["salary_raises"] = {
                d.isoformat() if isinstance(d, date) else str(d): v
                for d, v in income_model.fixed.salary_raises.items()
            }
        result["fixed"] = fixed_data

    # Variable income
    if income_model.variable is not None:
        var_data = {
            "base": income_model.variable.base,
            "sigma": income_model.variable.sigma,
            "annual_growth": income_model.variable.annual_growth,
        }
        if income_model.variable.seasonality is not None:
            s = income_model.variable.seasonality
            var_data["seasonality"] = list(s) if hasattr(s, '__iter__') else [s]
        if income_model.variable.floor is not None:
            var_data["floor"] = income_model.variable.floor
        if income_model.variable.cap is not None:
            var_data["cap"] = income_model.variable.cap
        if income_model.variable.seed is not None:
            var_data["seed"] = income_model.variable.seed
        result["variable"] = var_data

    # Contribution rates
    contrib = income_model.monthly_contribution or {}
    if isinstance(contrib, dict):
        fixed_rate = contrib.get("fixed", 0.3)
        var_rate = contrib.get("variable", 1.0)
        
        # Convert numpy arrays to lists for JSON serializability
        if isinstance(fixed_rate, np.ndarray):
            fixed_rate = fixed_rate.tolist()
        if isinstance(var_rate, np.ndarray):
            var_rate = var_rate.tolist()
            
        result["contribution_rate_fixed"] = fixed_rate
        result["contribution_rate_variable"] = var_rate
    else:
        result["contribution_rate_fixed"] = 0.3
        result["contribution_rate_variable"] = 1.0

    return result


def income_from_dict(data: Dict[str, Any]) -> IncomeModel:
    """
    Create IncomeModel from dictionary representation.

    Parameters
    ----------
    data : dict
        Dictionary with income configuration

    Returns
    -------
    IncomeModel
        Reconstructed income model instance
    """
    from .income import IncomeModel, FixedIncome, VariableIncome

    # Validate using Pydantic config
    config = IncomeConfig.model_validate(data)

    # Build fixed income
    fixed = None
    if config.fixed is not None:
        # Convert date strings back to date objects
        salary_raises = None
        if config.fixed.salary_raises:
            salary_raises = {
                date.fromisoformat(k): v
                for k, v in config.fixed.salary_raises.items()
            }

        fixed = FixedIncome(
            base=config.fixed.base,
            annual_growth=config.fixed.annual_growth,
            salary_raises=salary_raises,
        )

    # Build variable income
    variable = None
    if config.variable is not None:
        seasonality = None
        if config.variable.seasonality is not None:
            seasonality = np.array(config.variable.seasonality)

        variable = VariableIncome(
            base=config.variable.base,
            sigma=config.variable.sigma,
            annual_growth=config.variable.annual_growth,
            seasonality=seasonality,
            floor=config.variable.floor,
            cap=config.variable.cap,
            seed=config.variable.seed,
        )

    # Build income model
    # IncomeModel expects 12-element arrays for monthly contributions
    # Handle both scalar and per-month list inputs
    
    def _to_rate_array(rate):
        if isinstance(rate, list):
            return np.array(rate, dtype=float)
        return np.full(12, float(rate), dtype=float)

    monthly_contribution = {
        "fixed": _to_rate_array(config.contribution_rate_fixed),
        "variable": _to_rate_array(config.contribution_rate_variable),
    }

    return IncomeModel(
        fixed=fixed,
        variable=variable,
        monthly_contribution=monthly_contribution,
    )


# ---------------------------------------------------------------------------
# Withdrawal Serialization
# ---------------------------------------------------------------------------

def withdrawal_to_dict(withdrawal_model: WithdrawalModel) -> Dict[str, Any]:
    """
    Convert WithdrawalModel to dictionary representation.

    Parameters
    ----------
    withdrawal_model : WithdrawalModel
        Withdrawal model instance to serialize

    Returns
    -------
    dict
        Dictionary with withdrawal configuration containing:
        - scheduled: list of WithdrawalEvent dicts
        - stochastic: list of StochasticWithdrawal dicts

    Examples
    --------
    >>> from finopt.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent
    >>> from datetime import date
    >>> schedule = WithdrawalSchedule([WithdrawalEvent("Account1", 100000, date(2025, 6, 1))])
    >>> model = WithdrawalModel(scheduled=schedule)
    >>> withdrawal_to_dict(model)
    {'scheduled': [{'account': 'Account1', 'amount': 100000.0, 'date': '2025-06-01'}], 'stochastic': []}
    """
    result: Dict[str, Any] = {
        "scheduled": [],
        "stochastic": [],
    }

    # Serialize scheduled withdrawals
    if withdrawal_model.scheduled is not None:
        for event in withdrawal_model.scheduled.events:
            event_dict = {
                "account": event.account,
                "amount": event.amount,
                "date": event.date.isoformat(),
            }
            if event.description:
                event_dict["description"] = event.description
            result["scheduled"].append(event_dict)

    # Serialize stochastic withdrawals
    if withdrawal_model.stochastic:
        for sw in withdrawal_model.stochastic:
            sw_dict: Dict[str, Any] = {
                "account": sw.account,
                "base_amount": sw.base_amount,
                "sigma": sw.sigma,
                "floor": sw.floor,
            }
            # Include either month or date (mutually exclusive)
            if sw.month is not None:
                sw_dict["month"] = sw.month
            elif sw.date is not None:
                sw_dict["date"] = sw.date.isoformat()

            if sw.cap is not None:
                sw_dict["cap"] = sw.cap
            if sw.seed is not None:
                sw_dict["seed"] = sw.seed
            result["stochastic"].append(sw_dict)

    return result


def withdrawal_from_dict(data: Dict[str, Any]) -> WithdrawalModel:
    """
    Create WithdrawalModel from dictionary representation.

    Parameters
    ----------
    data : dict
        Dictionary with withdrawal configuration

    Returns
    -------
    WithdrawalModel
        Reconstructed withdrawal model instance

    Examples
    --------
    >>> data = {
    ...     "scheduled": [{"account": "Account1", "amount": 100000, "date": "2025-06-01"}],
    ...     "stochastic": []
    ... }
    >>> model = withdrawal_from_dict(data)
    >>> len(model.scheduled.events)
    1
    """
    from .withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent, StochasticWithdrawal

    # Validate using Pydantic config
    # Convert date strings to date objects for validation
    config_data: Dict[str, Any] = {"scheduled": [], "stochastic": []}

    for event_data in data.get("scheduled", []):
        config_data["scheduled"].append({
            "account": event_data["account"],
            "amount": event_data["amount"],
            "withdrawal_date": date.fromisoformat(event_data["date"]),
            "description": event_data.get("description", ""),
        })

    for sw_data in data.get("stochastic", []):
        sw_config: Dict[str, Any] = {
            "account": sw_data["account"],
            "base_amount": sw_data["base_amount"],
            "sigma": sw_data["sigma"],
            "floor": sw_data.get("floor", 0.0),
        }
        if "month" in sw_data:
            sw_config["month"] = sw_data["month"]
        elif "date" in sw_data:
            sw_config["withdrawal_date"] = date.fromisoformat(sw_data["date"])
        if "cap" in sw_data:
            sw_config["cap"] = sw_data["cap"]
        if "seed" in sw_data:
            sw_config["seed"] = sw_data["seed"]
        config_data["stochastic"].append(sw_config)

    # Validate with Pydantic
    config = WithdrawalConfig.model_validate(config_data)

    # Build WithdrawalSchedule
    events = []
    for event_config in config.scheduled:
        events.append(WithdrawalEvent(
            account=event_config.account,
            amount=event_config.amount,
            date=event_config.withdrawal_date,
            description=event_config.description,
        ))
    scheduled = WithdrawalSchedule(events=events) if events else None

    # Build StochasticWithdrawal list
    stochastic = []
    for sw_config in config.stochastic:
        stochastic.append(StochasticWithdrawal(
            account=sw_config.account,
            base_amount=sw_config.base_amount,
            sigma=sw_config.sigma,
            month=sw_config.month,
            date=sw_config.withdrawal_date,
            floor=sw_config.floor,
            cap=sw_config.cap,
            seed=sw_config.seed,
        ))

    return WithdrawalModel(
        scheduled=scheduled,
        stochastic=stochastic if stochastic else None,
    )


# ---------------------------------------------------------------------------
# Goal Serialization
# ---------------------------------------------------------------------------

def goals_to_dict(
    goals: List[Union[IntermediateGoal, TerminalGoal]]
) -> Dict[str, Any]:
    """
    Convert a list of goals to dictionary representation.

    Parameters
    ----------
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        List of goal objects to serialize

    Returns
    -------
    dict
        Dictionary with 'intermediate' and 'terminal' goal lists

    Examples
    --------
    >>> from datetime import date
    >>> from finopt.goals import IntermediateGoal, TerminalGoal
    >>> goals = [
    ...     IntermediateGoal(account="Savings", threshold=1_000_000, confidence=0.9,
    ...                      date=date(2025, 7, 1)),
    ...     TerminalGoal(account="Investment", threshold=5_000_000, confidence=0.8)
    ... ]
    >>> goals_to_dict(goals)
    {'intermediate': [...], 'terminal': [...]}
    """
    from .goals import IntermediateGoal, TerminalGoal

    result: Dict[str, Any] = {
        "intermediate": [],
        "terminal": [],
    }

    for goal in goals:
        if isinstance(goal, IntermediateGoal):
            goal_dict: Dict[str, Any] = {
                "account": goal.account,
                "threshold": goal.threshold,
                "confidence": goal.confidence,
                "date": goal.date.isoformat(),
            }
            result["intermediate"].append(goal_dict)

        elif isinstance(goal, TerminalGoal):
            result["terminal"].append({
                "account": goal.account,
                "threshold": goal.threshold,
                "confidence": goal.confidence,
            })

    return result


def goals_from_dict(
    data: Dict[str, Any],
    start_date: Optional[date] = None
) -> List[Union[IntermediateGoal, TerminalGoal]]:
    """
    Create a list of goals from dictionary representation.

    Parameters
    ----------
    data : dict
        Dictionary with 'intermediate' and 'terminal' goal lists
    start_date : date, optional
        Start date for backward compatibility with 'month' format.
        If data contains 'month' instead of 'date', this is used to
        convert month offset to calendar date. Defaults to today if None.

    Returns
    -------
    List[Union[IntermediateGoal, TerminalGoal]]
        Reconstructed goal objects

    Notes
    -----
    Backward compatibility: If an intermediate goal has 'month' instead of 'date',
    it will be converted to a date using start_date + month offset.

    Examples
    --------
    >>> data = {
    ...     "intermediate": [{"account": "Savings", "threshold": 1000000,
    ...                       "confidence": 0.9, "date": "2025-07-01"}],
    ...     "terminal": [{"account": "Investment", "threshold": 5000000, "confidence": 0.8}]
    ... }
    >>> goals = goals_from_dict(data)
    >>> len(goals)
    2
    """
    from .goals import IntermediateGoal, TerminalGoal

    goals: List[Union[IntermediateGoal, TerminalGoal]] = []

    # Process intermediate goals
    for goal_data in data.get("intermediate", []):
        # Handle backward compatibility: convert month to date
        if "date" in goal_data:
            goal_date = date.fromisoformat(goal_data["date"])
        elif "month" in goal_data:
            # Backward compatibility: convert month offset to date
            base_date = start_date if start_date else date.today()
            month_offset = goal_data["month"]
            # Calculate target date: start_date + month_offset months
            year = base_date.year + (base_date.month + month_offset - 1) // 12
            month = (base_date.month + month_offset - 1) % 12 + 1
            goal_date = date(year, month, 1)
            warnings.warn(
                f"IntermediateGoal with 'month={month_offset}' is deprecated. "
                f"Converted to date={goal_date.isoformat()}. "
                f"Please update your configuration to use 'date' format.",
                DeprecationWarning
            )
        else:
            raise ValueError(
                f"IntermediateGoal must have 'date' field, got: {goal_data.keys()}"
            )

        # Validate with Pydantic
        config_data = {
            "account": goal_data["account"],
            "threshold": goal_data["threshold"],
            "confidence": goal_data["confidence"],
            "goal_date": goal_date,
        }

        config = IntermediateGoalConfig.model_validate(config_data)

        # Build IntermediateGoal
        goals.append(IntermediateGoal(
            account=config.account,
            threshold=config.threshold,
            confidence=config.confidence,
            date=config.goal_date,
        ))

    # Process terminal goals
    for goal_data in data.get("terminal", []):
        config = TerminalGoalConfig.model_validate(goal_data)
        goals.append(TerminalGoal(
            account=config.account,
            threshold=config.threshold,
            confidence=config.confidence,
        ))

    return goals


# ---------------------------------------------------------------------------
# Scenario Serialization
# ---------------------------------------------------------------------------

def save_scenario(
    scenario_name: str,
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    path: Path,
    model: Optional[FinancialModel] = None,
    model_path: Optional[str] = None,
    withdrawals: Optional[WithdrawalModel] = None,
    start_date: Optional[date] = None,
    description: str = "",
    n_sims: int = 500,
    seed: Optional[int] = None,
    T_max: int = 240,
    solver: str = "ECOS",
    objective: str = "balanced",
) -> None:
    """
    Save a complete optimization scenario to JSON file.

    A scenario captures all parameters needed to reproduce an optimization:
    model configuration (or reference), goals, withdrawals, and parameters.

    Parameters
    ----------
    scenario_name : str
        Human-readable scenario name
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Financial goals for the scenario
    path : Path
        Output file path (should have .json extension)
    model : FinancialModel, optional
        If provided, embeds full model configuration in scenario.
        Mutually exclusive with model_path.
    model_path : str, optional
        Path to external model JSON file (for referencing without embedding).
        Mutually exclusive with model.
    withdrawals : WithdrawalModel, optional
        Scheduled and stochastic withdrawals
    start_date : date, optional
        Simulation start date. If None, uses today.
    description : str
        Optional scenario description
    n_sims : int
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility
    T_max : int
        Maximum optimization horizon (months)
    solver : str
        CVXPY solver backend
    objective : str
        Optimization objective function

    Examples
    --------
    >>> from pathlib import Path
    >>> from finopt.goals import TerminalGoal
    >>> goals = [TerminalGoal(account="Savings", threshold=10_000_000, confidence=0.8)]
    >>> save_scenario(
    ...     scenario_name="Retirement Plan",
    ...     goals=goals,
    ...     path=Path("scenarios/retirement.json"),
    ...     model_path="profiles/my_profile.json",
    ...     start_date=date(2025, 1, 1)
    ... )
    """
    from .goals import IntermediateGoal, TerminalGoal

    if model is not None and model_path is not None:
        raise ValueError("Specify either model or model_path, not both")

    if start_date is None:
        start_date = date.today()

    # Build scenario config
    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": scenario_name,
        "description": description,
        "start_date": start_date.isoformat(),
    }

    # Model: embed or reference
    if model is not None:
        config["model"] = {
            "income": income_to_dict(model.income),
            "accounts": [account_to_dict(acc) for acc in model.accounts],
        }
        if model.returns is not None:
            config["model"]["correlation"] = model.returns.default_correlation.tolist()
    elif model_path is not None:
        config["model_path"] = model_path

    # Goals
    goals_dict = goals_to_dict(goals)
    config["intermediate_goals"] = goals_dict["intermediate"]
    config["terminal_goals"] = goals_dict["terminal"]

    # Withdrawals
    if withdrawals is not None:
        config["withdrawals"] = withdrawal_to_dict(withdrawals)

    # Simulation parameters
    config["simulation"] = {
        "n_sims": n_sims,
        "seed": seed,
        "cache_enabled": True,
        "verbose": True,
    }

    # Optimization parameters
    config["optimization"] = {
        "T_max": T_max,
        "solver": solver,
        "objective": objective,
    }

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_scenario(
    path: Path,
    load_model_from_path: bool = True,
) -> Dict[str, Any]:
    """
    Load a scenario from JSON file.

    Returns a dictionary with all scenario components. If the scenario
    references an external model file and load_model_from_path=True,
    the model is loaded and included.

    Parameters
    ----------
    path : Path
        Input file path
    load_model_from_path : bool, default True
        If scenario has model_path, load the model from that file

    Returns
    -------
    dict
        Dictionary containing:
        - name: str
        - description: str
        - start_date: date
        - model: FinancialModel (if embedded or loaded from path)
        - model_path: str (if referenced)
        - goals: List[Union[IntermediateGoal, TerminalGoal]]
        - withdrawals: WithdrawalModel (if present)
        - simulation: SimulationConfig
        - optimization: OptimizationConfig

    Examples
    --------
    >>> from pathlib import Path
    >>> scenario = load_scenario(Path("scenarios/retirement.json"))
    >>> scenario["name"]
    'Retirement Plan'
    >>> len(scenario["goals"])
    2
    >>> scenario["model"]  # FinancialModel instance
    FinancialModel(M=2, ...)
    """
    with open(path, "r") as f:
        config = json.load(f)

    # Check schema version
    schema_version = config.get("schema_version", "0.0.0")
    if schema_version != SCHEMA_VERSION:
        warnings.warn(
            f"Scenario schema version {schema_version} differs from current "
            f"version {SCHEMA_VERSION}. May encounter compatibility issues.",
            UserWarning,
        )

    result: Dict[str, Any] = {
        "name": config["name"],
        "description": config.get("description", ""),
        "start_date": date.fromisoformat(config["start_date"]),
    }

    # Load model (embedded or from path)
    if "model" in config:
        # Embedded model configuration
        from .model import FinancialModel

        income = income_from_dict(config["model"]["income"])
        accounts = [account_from_dict(acc) for acc in config["model"]["accounts"]]
        correlation = None
        if "correlation" in config["model"]:
            correlation = np.array(config["model"]["correlation"])

        model = FinancialModel(income=income, accounts=accounts)
        if correlation is not None and model.returns is not None:
            model.returns.default_correlation = correlation

        result["model"] = model

    elif "model_path" in config:
        result["model_path"] = config["model_path"]
        if load_model_from_path:
            # Resolve path relative to scenario file
            model_file = path.parent / config["model_path"]
            if model_file.exists():
                result["model"] = load_model(model_file)
            else:
                # Try absolute path
                model_file = Path(config["model_path"])
                if model_file.exists():
                    result["model"] = load_model(model_file)
                else:
                    warnings.warn(
                        f"Model file not found: {config['model_path']}. "
                        f"Tried: {path.parent / config['model_path']} and {config['model_path']}"
                    )

    # Load goals
    goals_data = {
        "intermediate": config.get("intermediate_goals", []),
        "terminal": config.get("terminal_goals", []),
    }
    result["goals"] = goals_from_dict(goals_data)

    # Load withdrawals
    if "withdrawals" in config:
        result["withdrawals"] = withdrawal_from_dict(config["withdrawals"])
    else:
        result["withdrawals"] = None

    # Load simulation config
    sim_config = config.get("simulation", {})
    result["simulation"] = SimulationConfig(
        n_sims=sim_config.get("n_sims", 500),
        seed=sim_config.get("seed"),
        cache_enabled=sim_config.get("cache_enabled", True),
        verbose=sim_config.get("verbose", True),
    )

    # Load optimization config
    opt_config = config.get("optimization", {})
    result["optimization"] = OptimizationConfig(
        T_max=opt_config.get("T_max", 240),
        solver=opt_config.get("solver", "ECOS"),
        objective=opt_config.get("objective", "balanced"),
    )

    return result


# ---------------------------------------------------------------------------
# FinancialModel Serialization
# ---------------------------------------------------------------------------

def save_model(
    model: FinancialModel,
    path: Path,
    include_correlation: bool = True,
) -> None:
    """
    Save FinancialModel configuration to JSON file.

    Parameters
    ----------
    model : FinancialModel
        Model instance to save
    path : Path
        Output file path (should have .json extension)
    include_correlation : bool
        Whether to include return correlation matrix

    Examples
    --------
    >>> from finopt import FinancialModel
    >>> from pathlib import Path
    >>> save_model(model, Path("config.json"))
    """
    config = {
        "schema_version": SCHEMA_VERSION,
        "income": income_to_dict(model.income),
        "accounts": [account_to_dict(acc) for acc in model.accounts],
    }

    if include_correlation and model.returns is not None:
        config["correlation"] = model.returns.default_correlation.tolist()

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_model(path: Path) -> FinancialModel:
    """
    Load FinancialModel from JSON configuration file.

    Parameters
    ----------
    path : Path
        Input file path

    Returns
    -------
    FinancialModel
        Reconstructed model instance

    Examples
    --------
    >>> from pathlib import Path
    >>> model = load_model(Path("config.json"))
    """
    from .model import FinancialModel

    with open(path, "r") as f:
        config = json.load(f)

    # Check schema version
    schema_version = config.get("schema_version", "0.0.0")
    if schema_version != SCHEMA_VERSION:
        warnings.warn(
            f"Config schema version {schema_version} differs from current "
            f"version {SCHEMA_VERSION}. May encounter compatibility issues.",
            UserWarning,
        )

    # Reconstruct components
    income = income_from_dict(config["income"])
    accounts = [account_from_dict(acc_data) for acc_data in config["accounts"]]

    # Correlation matrix
    correlation = None
    if "correlation" in config:
        correlation = np.array(config["correlation"])

    # Build model
    model = FinancialModel(income=income, accounts=accounts)

    # Set correlation if provided
    if correlation is not None and model.returns is not None:
        model.returns.default_correlation = correlation

    return model


# ---------------------------------------------------------------------------
# OptimizationResult Serialization
# ---------------------------------------------------------------------------

def save_optimization_result(
    result: OptimizationResult,
    path: Path,
    include_policy: bool = True,
) -> None:
    """
    Save OptimizationResult to JSON file.

    Parameters
    ----------
    result : OptimizationResult
        Optimization result to save
    path : Path
        Output file path
    include_policy : bool
        Whether to include full allocation policy matrix X

    Examples
    --------
    >>> from pathlib import Path
    >>> save_optimization_result(result, Path("optimal_policy.json"))
    """
    config = {
        "schema_version": SCHEMA_VERSION,
        "T": result.T,
        "objective_value": result.objective_value,
        "feasible": result.feasible,
        "solve_time": result.solve_time,
        "n_iterations": result.n_iterations,
    }

    if include_policy:
        config["X"] = result.X.tolist()

    # Goal information
    config["goals"] = [
        {
            "type": "terminal" if not hasattr(g, "date") else "intermediate",
            "threshold": g.threshold,
            "confidence": g.confidence,
            "account": g.account if hasattr(g, "account") else None,
            "date": g.date.isoformat() if hasattr(g, "date") else None,
        }
        for g in result.goals
    ]

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_optimization_result(path: Path) -> Dict[str, Any]:
    """
    Load OptimizationResult from JSON file.

    Note: Returns dictionary instead of OptimizationResult because
    full reconstruction requires SimulationResult context.

    Parameters
    ----------
    path : Path
        Input file path

    Returns
    -------
    dict
        Dictionary with optimization result data

    Examples
    --------
    >>> from pathlib import Path
    >>> result_data = load_optimization_result(Path("optimal_policy.json"))
    >>> X = np.array(result_data["X"])
    >>> T = result_data["T"]
    """
    with open(path, "r") as f:
        config = json.load(f)

    # Check schema version
    schema_version = config.get("schema_version", "0.0.0")
    if schema_version != SCHEMA_VERSION:
        warnings.warn(
            f"Config schema version {schema_version} differs from current "
            f"version {SCHEMA_VERSION}. May encounter compatibility issues.",
            UserWarning,
        )

    # Convert X back to numpy array if present
    if "X" in config:
        config["X"] = np.array(config["X"])

    return config
