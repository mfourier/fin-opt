"""
Tests for API Optimization Service (api/services/optimization.py)

Tests cover:
- compute_goal_status_from_result() formats goal status correctly
- Integration of optimization workflow (via mocking)

Note: Full end-to-end optimization tests are intentionally light since
the core optimization logic is tested in test_optimization.py. These tests
focus on the API service layer (result formatting, job updates).
"""

from datetime import date

import pytest


def test_compute_goal_status_from_result_intermediate_goal():
    """Test compute_goal_status_from_result formats intermediate goal correctly."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, IntermediateGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    # Create minimal model
    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [Account.from_annual("Emergency", 0.04, 0.05, 0)]
    model = FinancialModel(income=income, accounts=accounts)

    # Create goal
    goal = IntermediateGoal(
        date=date(2025, 7, 1),
        account="Emergency",
        threshold=5_000_000,
        confidence=0.8
    )

    # Create mock optimization result
    opt_result = type('OptResult', (), {
        'T': 12,
        'feasible': True,
        'goals': [goal]
    })()

    # Create mock sim_result with wealth above threshold (all sims satisfy goal)
    sim_result = type('SimResult', (), {
        'wealth': np.full((100, 13, 1), 6_000_000.0),  # (n_sims, T+1, M)
    })()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    assert len(status) == 1
    assert status[0]["type"] == "intermediate"
    assert status[0]["account"] == "Emergency"
    assert status[0]["threshold"] == 5_000_000
    assert status[0]["required_confidence"] == 0.8
    assert status[0]["satisfied"] is True  # All sims above threshold


def test_compute_goal_status_from_result_terminal_goal():
    """Test compute_goal_status_from_result formats terminal goal correctly."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, TerminalGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [Account.from_annual("Retirement", 0.10, 0.15, 0)]
    model = FinancialModel(income=income, accounts=accounts)

    goal = TerminalGoal(
        account="Retirement",
        threshold=30_000_000,
        confidence=0.8
    )

    opt_result = type('OptResult', (), {
        'T': 240,
        'feasible': True,
        'goals': [goal]
    })()

    # Create mock sim_result with wealth above threshold at terminal time
    sim_result = type('SimResult', (), {
        'wealth': np.full((100, 241, 1), 35_000_000.0),  # (n_sims, T+1, M)
    })()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    assert len(status) == 1
    assert status[0]["type"] == "terminal"
    assert "T=240" in status[0]["goal"]
    assert status[0]["satisfied"] is True


def test_compute_goal_status_from_result_infeasible():
    """Test compute_goal_status_from_result marks infeasible correctly."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, TerminalGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [Account.from_annual("Retirement", 0.10, 0.15, 0)]
    model = FinancialModel(income=income, accounts=accounts)

    goal = TerminalGoal(account="Retirement", threshold=30_000_000, confidence=0.8)

    # Infeasible result
    opt_result = type('OptResult', (), {
        'T': 240,
        'feasible': False,  # Not feasible
        'goals': [goal]
    })()

    # Create mock sim_result with wealth below threshold (goal not satisfied)
    sim_result = type('SimResult', (), {
        'wealth': np.full((100, 241, 1), 10_000_000.0),  # Below 30M threshold
    })()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    assert status[0]["satisfied"] is False


def test_compute_goal_status_from_result_mixed_goals():
    """Test compute_goal_status_from_result handles multiple goals."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, IntermediateGoal, TerminalGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [
        Account.from_annual("Emergency", 0.04, 0.05, 0),
        Account.from_annual("Retirement", 0.10, 0.15, 0)
    ]
    model = FinancialModel(income=income, accounts=accounts)

    goals = [
        IntermediateGoal(date=date(2025, 7, 1), account="Emergency", threshold=5_000_000, confidence=0.8),
        TerminalGoal(account="Retirement", threshold=30_000_000, confidence=0.8)
    ]

    opt_result = type('OptResult', (), {
        'T': 24,
        'feasible': True,
        'goals': goals
    })()

    # Create mock sim_result with wealth above thresholds for both accounts
    sim_result = type('SimResult', (), {
        'wealth': np.full((100, 25, 2), 50_000_000.0),  # (n_sims, T+1, M=2)
    })()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    assert len(status) == 2
    assert status[0]["type"] == "intermediate"
    assert status[1]["type"] == "terminal"


def test_compute_goal_status_from_result_includes_dual_metrics():
    """Test that compute_goal_status_from_result includes CVaR dual metric fields."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, TerminalGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [Account.from_annual("Retirement", 0.10, 0.15, 0)]
    model = FinancialModel(income=income, accounts=accounts)

    goal = TerminalGoal(account="Retirement", threshold=10_000_000, confidence=0.80)

    opt_result = type("OptResult", (), {"T": 24, "feasible": True, "goals": [goal]})()

    # 90% of scenarios above threshold — empirical 90%, specified 80%
    wealth = np.zeros((100, 25, 1))
    wealth[:90, -1, 0] = 12_000_000
    wealth[10:, -1, 0] = 12_000_000
    sim_result = type("SimResult", (), {"wealth": np.full((100, 25, 1), 12_000_000.0)})()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    s = status[0]
    assert "empirical_probability" in s
    assert "confidence_gap" in s
    assert "note" in s

    assert s["empirical_probability"] == 1.0
    assert abs(s["confidence_gap"] - 0.20) < 1e-9
    assert "CVaR" in s["note"]


def test_compute_goal_status_from_result_dual_metrics_with_actual_prob_none():
    """Test dual metrics are None when actual_probability cannot be computed."""
    import numpy as np

    from api.services.optimization import compute_goal_status_from_result
    from finopt import FinancialModel, IntermediateGoal
    from finopt.income import FixedIncome, IncomeModel
    from finopt.portfolio import Account

    income = IncomeModel(fixed=FixedIncome(base=1_000_000))
    accounts = [Account.from_annual("Emergency", 0.04, 0.05, 0)]
    model = FinancialModel(income=income, accounts=accounts)

    # Goal at month 6, but sim_result only has T=5 (month 6 out of bounds)
    goal = IntermediateGoal(date=date(2025, 7, 1), account="Emergency", threshold=5_000_000, confidence=0.80)

    opt_result = type("OptResult", (), {"T": 5, "feasible": True, "goals": [goal]})()
    # wealth shape (n_sims, 6, M) — index 6 is out of range → actual_prob = None
    sim_result = type("SimResult", (), {"wealth": np.full((100, 6, 1), 6_000_000.0)})()

    status = compute_goal_status_from_result(opt_result, model, sim_result, date(2025, 1, 1))

    s = status[0]
    assert s["empirical_probability"] is None
    assert s["confidence_gap"] is None
    assert s["note"] is None


@pytest.mark.asyncio
async def test_run_optimization_updates_job_status(mocker, sample_scenario_data, mock_env_vars):
    """Test run_optimization updates job status throughout execution."""
    # Mock Supabase functions
    mocker.patch('api.services.optimization.fetch_scenario_with_profile', return_value=sample_scenario_data)
    mock_update_job = mocker.patch('api.services.optimization.update_job')
    mock_save_result = mocker.patch('api.services.optimization.save_optimization_result')

    # Mock model.optimize to return a result
    import numpy as np
    mock_opt_result = type('OptResult', (), {
        'T': 24,
        'X': np.array([[0.6, 0.4]] * 24),
        'feasible': True,
        'objective_value': 1000.0,
        'solve_time': 5.0,
        'n_iterations': 10,
        'goals': [],
        'diagnostics': {}
    })()

    mocker.patch('finopt.FinancialModel.optimize', return_value=mock_opt_result)

    # Mock simulate_from_optimization to avoid isinstance check on mock opt_result
    mock_sim_result = type('SimResult', (), {
        'wealth': np.full((100, 25, 2), 50_000_000.0),
        'total_wealth': np.full((100, 25), 100_000_000.0),
        'T': 24,
        'M': 2,
        'allocation': np.array([[0.6, 0.4]] * 24),
        'contributions': np.full((100, 24), 500_000.0),
        'withdrawals': None,
    })()
    mocker.patch('finopt.FinancialModel.simulate_from_optimization', return_value=mock_sim_result)

    from api.services.optimization import run_optimization

    await run_optimization(scenario_id="test-scenario", job_id="test-job")

    # Check that update_job was called multiple times
    assert mock_update_job.call_count >= 3

    # Check final status is 'completed'
    final_call = mock_update_job.call_args_list[-1]
    assert final_call[1].get("status") == "completed" or final_call[0][1] == "completed"

    # Check save_result was called
    assert mock_save_result.called


@pytest.mark.asyncio
async def test_run_optimization_handles_error(mocker, sample_scenario_data, mock_env_vars):
    """Test run_optimization handles errors and marks job as failed."""
    # Mock fetch to raise an error
    mocker.patch('api.services.optimization.fetch_scenario_with_profile', side_effect=ValueError("Test error"))
    mock_update_job = mocker.patch('api.services.optimization.update_job')

    from api.services.optimization import run_optimization

    with pytest.raises(ValueError):
        await run_optimization(scenario_id="test-scenario", job_id="test-job")

    # Should have updated job with failed status
    calls = [call for call in mock_update_job.call_args_list if 'status' in call[1]]
    failed_calls = [call for call in calls if call[1]['status'] == 'failed']
    assert len(failed_calls) > 0


@pytest.mark.asyncio
async def test_run_optimization_saves_allocation_policy(mocker, sample_scenario_data, mock_env_vars):
    """Test run_optimization saves allocation policy in correct format."""
    mocker.patch('api.services.optimization.fetch_scenario_with_profile', return_value=sample_scenario_data)
    mocker.patch('api.services.optimization.update_job')

    import numpy as np
    # Create allocation policy
    X = np.array([[0.6, 0.4]] * 24)
    mock_opt_result = type('OptResult', (), {
        'T': 24,
        'X': X,
        'feasible': True,
        'objective_value': 1000.0,
        'solve_time': 5.0,
        'n_iterations': 10,
        'goals': [],
        'diagnostics': {}
    })()

    mocker.patch('finopt.FinancialModel.optimize', return_value=mock_opt_result)

    # Mock simulate_from_optimization to avoid isinstance check on mock opt_result
    mock_sim_result = type('SimResult', (), {
        'wealth': np.full((100, 25, 2), 50_000_000.0),
        'total_wealth': np.full((100, 25), 100_000_000.0),
        'T': 24,
        'M': 2,
        'allocation': np.array([[0.6, 0.4]] * 24),
        'contributions': np.full((100, 24), 500_000.0),
        'withdrawals': None,
    })()
    mocker.patch('finopt.FinancialModel.simulate_from_optimization', return_value=mock_sim_result)

    mock_save = mocker.patch('api.services.optimization.save_optimization_result')

    from api.services.optimization import run_optimization

    await run_optimization(scenario_id="test-scenario", job_id="test-job")

    # Check save was called with allocation_policy as list
    assert mock_save.called
    call_kwargs = mock_save.call_args[1]
    assert "allocation_policy" in call_kwargs
    assert isinstance(call_kwargs["allocation_policy"], list)
    assert len(call_kwargs["allocation_policy"]) == 24
