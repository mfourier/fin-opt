"""
Model Reconstruction Service

Reconstructs FinOpt objects (FinancialModel, Goals, Withdrawals) from
Supabase JSON data using the existing serialization functions.

The JSON schema in Supabase matches the format from finopt.serialization,
so we can reuse those functions directly.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Union

import numpy as np

# Import from finopt package
from finopt import (
    FinancialModel,
    IntermediateGoal,
    TerminalGoal,
    WithdrawalModel,
    account_from_dict,
    income_from_dict,
    withdrawal_from_dict,
)
from finopt.serialization import goals_from_dict


def reconstruct_model(profile_data: dict[str, Any]) -> FinancialModel:
    """
    Reconstruct a FinancialModel from Supabase profile data.

    Parameters
    ----------
    profile_data : dict
        Profile record from Supabase with keys:
        - income_config: dict (matches income_to_dict output)
        - accounts_config: list[dict] (matches account_to_dict output)
        - correlation_matrix: list[list[float]] | None

    Returns
    -------
    FinancialModel
        Fully reconstructed model ready for simulation/optimization.

    Examples
    --------
    >>> profile = {
    ...     "income_config": {
    ...         "fixed": {"base": 1500000, "annual_growth": 0.03},
    ...         "contribution_rate_fixed": 0.3,
    ...         "contribution_rate_variable": 1.0
    ...     },
    ...     "accounts_config": [
    ...         {"name": "Emergency", "annual_return": 0.04,
    ...          "annual_volatility": 0.05, "initial_wealth": 0}
    ...     ],
    ...     "correlation_matrix": None
    ... }
    >>> model = reconstruct_model(profile)
    >>> model.M
    1
    """
    # Reconstruct income model
    income = income_from_dict(profile_data["income_config"])

    # Reconstruct accounts
    accounts = [
        account_from_dict(acc_data)
        for acc_data in profile_data["accounts_config"]
    ]

    # Create model
    model = FinancialModel(income=income, accounts=accounts)

    # Set correlation matrix if provided
    if profile_data.get("correlation_matrix") is not None:
        correlation = np.array(profile_data["correlation_matrix"])
        if model.returns is not None:
            model.returns.default_correlation = correlation

    return model


def reconstruct_goals(
    scenario_data: dict[str, Any],
    start_date: date | None = None,
) -> list[Union[IntermediateGoal, TerminalGoal]]:
    """
    Reconstruct goals from Supabase scenario data.

    Parameters
    ----------
    scenario_data : dict
        Scenario record from Supabase with keys:
        - intermediate_goals: list[dict]
        - terminal_goals: list[dict]
        - start_date: str (ISO format) - used if start_date param is None
    start_date : date, optional
        Start date for resolving goal months. If None, uses scenario's start_date.

    Returns
    -------
    list[Union[IntermediateGoal, TerminalGoal]]
        List of goal objects.

    Examples
    --------
    >>> scenario = {
    ...     "start_date": "2025-01-01",
    ...     "intermediate_goals": [
    ...         {"account": "Emergency", "threshold": 5500000,
    ...          "confidence": 0.9, "date": "2025-07-01"}
    ...     ],
    ...     "terminal_goals": [
    ...         {"account": "Retirement", "threshold": 20000000, "confidence": 0.8}
    ...     ]
    ... }
    >>> goals = reconstruct_goals(scenario)
    >>> len(goals)
    2
    """
    # Determine start date
    if start_date is None:
        start_date_str = scenario_data.get("start_date")
        if start_date_str:
            if isinstance(start_date_str, str):
                start_date = date.fromisoformat(start_date_str)
            elif isinstance(start_date_str, date):
                start_date = start_date_str

    # Build goals dict in the format expected by goals_from_dict
    goals_data = {
        "intermediate": scenario_data.get("intermediate_goals", []),
        "terminal": scenario_data.get("terminal_goals", []),
    }

    return goals_from_dict(goals_data, start_date=start_date)


def reconstruct_withdrawals(
    scenario_data: dict[str, Any]
) -> WithdrawalModel | None:
    """
    Reconstruct WithdrawalModel from Supabase scenario data.

    Parameters
    ----------
    scenario_data : dict
        Scenario record from Supabase with key:
        - withdrawals: dict | None (matches withdrawal_to_dict output)

    Returns
    -------
    WithdrawalModel | None
        Withdrawal model if withdrawals are defined, None otherwise.

    Examples
    --------
    >>> scenario = {
    ...     "withdrawals": {
    ...         "scheduled": [
    ...             {"account": "Emergency", "amount": 1000000, "date": "2025-06-01"}
    ...         ],
    ...         "stochastic": []
    ...     }
    ... }
    >>> withdrawals = reconstruct_withdrawals(scenario)
    >>> withdrawals is not None
    True
    """
    withdrawals_data = scenario_data.get("withdrawals")

    if withdrawals_data is None:
        return None

    # Check if withdrawals data is empty
    scheduled = withdrawals_data.get("scheduled", [])
    stochastic = withdrawals_data.get("stochastic", [])

    if not scheduled and not stochastic:
        return None

    return withdrawal_from_dict(withdrawals_data)


def reconstruct_from_scenario(
    scenario_data: dict[str, Any],
) -> tuple[FinancialModel, list[Union[IntermediateGoal, TerminalGoal]], WithdrawalModel | None]:
    """
    Reconstruct all components from a Supabase scenario with embedded profile.

    This is a convenience function that extracts and reconstructs all
    components needed for simulation/optimization from a single scenario
    record that includes the related profile.

    Parameters
    ----------
    scenario_data : dict
        Scenario record from Supabase with:
        - profiles: dict (nested profile data from JOIN)
        - intermediate_goals: list[dict]
        - terminal_goals: list[dict]
        - withdrawals: dict | None
        - start_date: str

    Returns
    -------
    tuple[FinancialModel, list[Goal], WithdrawalModel | None]
        Tuple of (model, goals, withdrawals).

    Examples
    --------
    >>> # Typically called with data from:
    >>> # supabase.table("scenarios").select("*, profiles(*)").eq("id", id).single()
    >>> scenario = fetch_scenario_with_profile(scenario_id)
    >>> model, goals, withdrawals = reconstruct_from_scenario(scenario)
    """
    # Extract profile data (nested from JOIN)
    profile_data = scenario_data.get("profiles")
    if profile_data is None:
        raise ValueError(
            "Scenario data must include 'profiles' from JOIN. "
            "Use: supabase.table('scenarios').select('*, profiles(*)').eq('id', id)"
        )

    # Reconstruct all components
    model = reconstruct_model(profile_data)
    goals = reconstruct_goals(scenario_data)
    withdrawals = reconstruct_withdrawals(scenario_data)

    return model, goals, withdrawals
