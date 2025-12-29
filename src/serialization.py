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
from typing import Dict, Any, List, Union, Optional, TYPE_CHECKING
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
)

if TYPE_CHECKING:
    from .model import FinancialModel
    from .portfolio import Account
    from .income import IncomeModel, FixedIncome, VariableIncome
    from .optimization import OptimizationResult

__all__ = [
    "save_model",
    "load_model",
    "save_optimization_result",
    "load_optimization_result",
    "account_to_dict",
    "account_from_dict",
    "income_to_dict",
    "income_from_dict",
]


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.1.0"


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
    return {
        "name": account.name,
        "annual_return": account.annual_params["return"],
        "annual_volatility": account.annual_params["volatility"],
        "initial_wealth": account.initial_wealth,
    }


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
        if income_model.fixed.raises:
            fixed_data["raises"] = income_model.fixed.raises
        result["fixed"] = fixed_data

    # Variable income
    if income_model.variable is not None:
        var_data = {
            "base": income_model.variable.base,
            "sigma": income_model.variable.sigma,
            "annual_growth": income_model.variable.annual_growth,
        }
        if income_model.variable.seasonality is not None:
            var_data["seasonality"] = income_model.variable.seasonality.tolist()
        if income_model.variable.floor is not None:
            var_data["floor"] = income_model.variable.floor
        if income_model.variable.cap is not None:
            var_data["cap"] = income_model.variable.cap
        if income_model.variable.seed is not None:
            var_data["seed"] = income_model.variable.seed
        result["variable"] = var_data

    # Contribution rates
    result["contribution_rate_fixed"] = income_model.monthly_contribution.get("fixed", 0.3)
    result["contribution_rate_variable"] = income_model.monthly_contribution.get("variable", 1.0)

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
        fixed = FixedIncome(
            base=config.fixed.base,
            annual_growth=config.fixed.annual_growth,
            raises=config.fixed.raises,
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
    monthly_contribution = {
        "fixed": config.contribution_rate_fixed,
        "variable": config.contribution_rate_variable,
    }

    return IncomeModel(
        fixed=fixed,
        variable=variable,
        monthly_contribution=monthly_contribution,
    )


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
        config["correlation"] = model.returns.correlation.tolist()

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
        model.returns.correlation = correlation

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
            "type": "terminal" if hasattr(g, "account") else "intermediate",
            "threshold": g.threshold,
            "confidence": g.confidence,
            "account": g.account if hasattr(g, "account") else None,
            "month": g.month if hasattr(g, "month") else None,
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
