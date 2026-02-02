"""
Tests for API Reconstruction Service (api/services/reconstruction.py)

Tests cover:
- reconstruct_model() from profile data
- reconstruct_goals() from scenario data (intermediate + terminal)
- reconstruct_withdrawals() from scenario data
- reconstruct_from_scenario() end-to-end integration
- Error handling for missing/invalid data
"""

from datetime import date

import numpy as np
import pytest

from finopt import FinancialModel, IntermediateGoal, TerminalGoal, WithdrawalModel


def test_reconstruct_model_complete(sample_profile_data):
    """Test reconstruct_model creates FinancialModel from complete profile data."""
    from api.services.reconstruction import reconstruct_model
    
    model = reconstruct_model(sample_profile_data)
    
    # Check it's a FinancialModel
    assert isinstance(model, FinancialModel)
    
    # Check income model
    assert model.income is not None
    assert model.income.fixed.base == 1_500_000
    assert model.income.fixed.annual_growth == 0.03
    assert model.income.variable.base == 200_000
    
    # Check accounts
    assert model.M == 2
    assert len(model.accounts) == 2
    assert model.accounts[0].name == "Conservative"
    assert model.accounts[1].name == "Aggressive"
    
    # Check returns model created
    assert model.returns is not None


def test_reconstruct_model_with_correlation(sample_profile_data):
    """Test reconstruct_model handles correlation matrix."""
    from api.services.reconstruction import reconstruct_model
    
    # Add correlation matrix
    sample_profile_data["correlation_matrix"] = [
        [1.0, 0.3],
        [0.3, 1.0]
    ]
    
    model = reconstruct_model(sample_profile_data)
    
    # Check correlation was set
    assert model.returns is not None
    expected = np.array([[1.0, 0.3], [0.3, 1.0]])
    np.testing.assert_array_equal(model.returns.default_correlation, expected)


def test_reconstruct_goals_intermediate_only(sample_scenario_data):
    """Test reconstruct_goals with only intermediate goals."""
    from api.services.reconstruction import reconstruct_goals
    
    # Remove terminal goals
    sample_scenario_data["terminal_goals"] = []
    
    goals = reconstruct_goals(sample_scenario_data)
    
    assert len(goals) == 1
    assert isinstance(goals[0], IntermediateGoal)
    assert goals[0].account == "Conservative"
    assert goals[0].threshold == 5_000_000
    assert goals[0].confidence == 0.8
    assert goals[0].date == date(2025, 7, 1)


def test_reconstruct_goals_terminal_only(sample_scenario_data):
    """Test reconstruct_goals with only terminal goals."""
    from api.services.reconstruction import reconstruct_goals
    
    # Remove intermediate goals
    sample_scenario_data["intermediate_goals"] = []
    
    goals = reconstruct_goals(sample_scenario_data)
    
    assert len(goals) == 1
    assert isinstance(goals[0], TerminalGoal)
    assert goals[0].account == "Aggressive"
    assert goals[0].threshold == 30_000_000
    assert goals[0].confidence == 0.8


def test_reconstruct_goals_mixed(sample_scenario_data):
    """Test reconstruct_goals with both intermediate and terminal goals."""
    from api.services.reconstruction import reconstruct_goals
    
    goals = reconstruct_goals(sample_scenario_data)
    
    assert len(goals) == 2
    
    # First should be intermediate
    assert isinstance(goals[0], IntermediateGoal)
    assert goals[0].date == date(2025, 7, 1)
    
    # Second should be terminal
    assert isinstance(goals[1], TerminalGoal)


def test_reconstruct_goals_with_explicit_start_date(sample_scenario_data):
    """Test reconstruct_goals uses provided start_date parameter."""
    from api.services.reconstruction import reconstruct_goals
    
    custom_start = date(2026, 1, 1)
    goals = reconstruct_goals(sample_scenario_data, start_date=custom_start)
    
    # Goals should still be reconstructed
    assert len(goals) == 2


def test_reconstruct_withdrawals_none_when_empty(sample_scenario_data):
    """Test reconstruct_withdrawals returns None when withdrawals are null/empty."""
    from api.services.reconstruction import reconstruct_withdrawals
    
    # Null withdrawals
    sample_scenario_data["withdrawals"] = None
    result = reconstruct_withdrawals(sample_scenario_data)
    assert result is None
    
    # Empty withdrawals
    sample_scenario_data["withdrawals"] = {"scheduled": [], "stochastic": []}
    result = reconstruct_withdrawals(sample_scenario_data)
    assert result is None


def test_reconstruct_withdrawals_scheduled(sample_scenario_data):
    """Test reconstruct_withdrawals creates WithdrawalModel with scheduled withdrawals."""
    from api.services.reconstruction import reconstruct_withdrawals
    
    sample_scenario_data["withdrawals"] = {
        "scheduled": [
            {
                "account": "Conservative",
                "amount": 1_000_000,
                "date": "2025-06-01"
            }
        ],
        "stochastic": []
    }
    
    withdrawals = reconstruct_withdrawals(sample_scenario_data)
    
    assert isinstance(withdrawals, WithdrawalModel)
    assert len(withdrawals.scheduled.events) == 1
    assert withdrawals.scheduled.events[0].amount == 1_000_000


def test_reconstruct_from_scenario_integration(sample_scenario_data):
    """Test reconstruct_from_scenario reconstructs all components."""
    from api.services.reconstruction import reconstruct_from_scenario
    
    model, goals, withdrawals = reconstruct_from_scenario(sample_scenario_data)
    
    # Check model
    assert isinstance(model, FinancialModel)
    assert model.M == 2
    
    # Check goals
    assert len(goals) == 2
    assert isinstance(goals[0], IntermediateGoal)
    assert isinstance(goals[1], TerminalGoal)
    
    # Check withdrawals (None in sample data)
    assert withdrawals is None


def test_reconstruct_from_scenario_missing_profile_raises(sample_scenario_data):
    """Test reconstruct_from_scenario raises error when profile is missing."""
    from api.services.reconstruction import reconstruct_from_scenario
    
    # Remove profiles key
    del sample_scenario_data["profiles"]
    
    with pytest.raises(ValueError) as exc_info:
        reconstruct_from_scenario(sample_scenario_data)
    
    assert "profiles" in str(exc_info.value).lower()


def test_reconstruct_from_scenario_with_withdrawals(sample_scenario_data):
    """Test reconstruct_from_scenario handles scenarios with withdrawals."""
    from api.services.reconstruction import reconstruct_from_scenario
    
    # Add withdrawals to scenario
    sample_scenario_data["withdrawals"] = {
        "scheduled": [
            {
                "account": "Conservative",
                "amount": 500_000,
                "date": "2025-03-01"
            }
        ],
        "stochastic": []
    }
    
    model, goals, withdrawals = reconstruct_from_scenario(sample_scenario_data)
    
    # Withdrawals should be reconstructed
    assert withdrawals is not None
    assert isinstance(withdrawals, WithdrawalModel)
    assert len(withdrawals.scheduled) == 1


def test_reconstruct_goals_empty_returns_empty_list(sample_scenario_data):
    """Test reconstruct_goals returns empty list when no goals defined."""
    from api.services.reconstruction import reconstruct_goals
    
    sample_scenario_data["intermediate_goals"] = []
    sample_scenario_data["terminal_goals"] = []
    
    goals = reconstruct_goals(sample_scenario_data)
    
    assert goals == []
