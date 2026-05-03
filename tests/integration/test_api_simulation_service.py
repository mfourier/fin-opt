"""
Tests for API Simulation Service (api/services/simulation.py)

Tests cover:
- compute_summary_stats() calculates statistics correctly
- compute_goal_status() evaluates goal satisfaction
- Integration of simulation workflow (via mocking)

Note: Full end-to-end simulation tests are intentionally light since
the core simulation logic is tested in test_model.py. These tests focus
on the API service layer (stats computation, goal evaluation, job updates).
"""

import numpy as np
import pytest
from datetime import date


def test_compute_summary_stats_structure():
    """Test compute_summary_stats returns correct structure."""
    from api.services.simulation import compute_summary_stats
    
    # Create sample wealth array (n_sims=10, T+1=25, M=2)
    wealth = np.random.uniform(1000000, 10000000, size=(10, 25, 2))
    account_names = ["Conservative", "Aggressive"]
    
    stats = compute_summary_stats(wealth, account_names)
    
    # Check structure
    assert "total_wealth" in stats
    assert "final_total_wealth" in stats
    assert "accounts" in stats
    
    # Check total wealth stats
    assert "mean" in stats["total_wealth"]
    assert "median" in stats["total_wealth"]
    assert "std" in stats["total_wealth"]
    assert "p10" in stats["total_wealth"]
    assert "p90" in stats["total_wealth"]
    
    # Check per-account stats
    assert len(stats["accounts"]) == 2
    assert stats["accounts"][0]["name"] == "Conservative"
    assert stats["accounts"][1]["name"] == "Aggressive"


def test_compute_summary_stats_calculations():
    """Test compute_summary_stats calculates correct values."""
    from api.services.simulation import compute_summary_stats
    
    # Create deterministic wealth array
    # 5 scenarios, 3 time periods, 2 accounts
    # Wealth: all scenarios have same trajectory
    wealth = np.array([
        [[1000, 2000], [1100, 2100], [1200, 2200]],
        [[1000, 2000], [1100, 2100], [1200, 2200]],
        [[1000, 2000], [1100, 2100], [1200, 2200]],
        [[1000, 2000], [1100, 2100], [1200, 2200]],
        [[1000, 2000], [1100, 2100], [1200, 2200]],
    ])
    
    stats = compute_summary_stats(wealth, ["A", "B"])
    
    # Final total wealth across scenarios (at T=2)
    # All scenarios: 1200 + 2200 = 3400
    assert stats["final_total_wealth"]["mean"] == 3400
    assert stats["final_total_wealth"]["median"] == 3400


def test_compute_goal_status_intermediate_satisfied(sample_scenario_data):
    """Test compute_goal_status for satisfied intermediate goal."""
    from api.services.simulation import compute_goal_status
    from finopt import IntermediateGoal
    
    # Create wealth array where Conservative account exceeds threshold at month 6
    # Shape: (n_sims=100, T+1=25, M=2)
    wealth = np.zeros((100, 25, 2))
    # Conservative account (index 0) has 6M at month 6 in all scenarios
    wealth[:, 6, 0] = 6_000_000  # All scenarios satisfy 5M threshold
    
    goals = [
        IntermediateGoal(
            date=date(2025, 7, 1),  # Month 6
            account=0,  # Conservative
            threshold=5_000_000,
            confidence=0.8
        )
    ]
    
    accounts = [
        type('Account', (), {'name': 'Conservative'})(),
        type('Account', (), {'name': 'Aggressive'})()
    ]
    
    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))
    
    assert len(status) == 1
    assert status[0]["satisfied"] is True  # 100% > 80% required
    assert status[0]["probability"] == 1.0


def test_compute_goal_status_intermediate_not_satisfied():
    """Test compute_goal_status for unsatisfied intermediate goal."""
    from api.services.simulation import compute_goal_status
    from finopt import IntermediateGoal
    
    # Create wealth array where only 50% of scenarios satisfy goal
    wealth = np.zeros((100, 25, 2))
    # Only first 50 scenarios reach threshold
    wealth[:50, 6, 0] = 6_000_000
    wealth[50:, 6, 0] = 3_000_000  # Below threshold
    
    goals = [
        IntermediateGoal(
            date=date(2025, 7, 1),
            account=0,
            threshold=5_000_000,
            confidence=0.8  # Requires 80%
        )
    ]
    
    accounts = [type('Account', (), {'name': 'Conservative'})()]
    
    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))
    
    assert status[0]["satisfied"] is False  # 50% < 80% required
    assert status[0]["probability"] == 0.5


def test_compute_goal_status_terminal_goal():
    """Test compute_goal_status evaluates terminal goal at final timestep."""
    from api.services.simulation import compute_goal_status
    from finopt import TerminalGoal
    
    # Final timestep wealth
    wealth = np.zeros((100, 25, 2))
    wealth[:, -1, 1] = 35_000_000  # All scenarios: 35M in Aggressive
    
    goals = [
        TerminalGoal(
            account=1,  # Aggressive
            threshold=30_000_000,
            confidence=0.8
        )
    ]
    
    accounts = [
        type('Account', (), {'name': 'Conservative'})(),
        type('Account', (), {'name': 'Aggressive'})()
    ]
    
    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))
    
    assert len(status) == 1
    assert status[0]["satisfied"] is True
    assert status[0]["probability"] == 1.0


def test_compute_goal_status_mixed_goals():
    """Test compute_goal_status handles mix of goal types."""
    from api.services.simulation import compute_goal_status
    from finopt import IntermediateGoal, TerminalGoal
    
    wealth = np.zeros((100, 25, 2))
    wealth[:, 6, 0] = 6_000_000  # Intermediate satisfied
    wealth[:, -1, 1] = 35_000_000  # Terminal satisfied
    
    goals = [
        IntermediateGoal(date=date(2025, 7, 1), account=0, threshold=5_000_000, confidence=0.8),
        TerminalGoal(account=1, threshold=30_000_000, confidence=0.8)
    ]
    
    accounts = [
        type('Account', (), {'name': 'Conservative'})(),
        type('Account', (), {'name': 'Aggressive'})()
    ]
    
    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))
    
    assert len(status) == 2
    assert all(s["satisfied"] for s in status)


def test_compute_goal_status_includes_dual_metrics():
    """Test that compute_goal_status includes CVaR dual metric fields."""
    from api.services.simulation import compute_goal_status
    from finopt import TerminalGoal
    import numpy as np

    wealth = np.zeros((100, 25, 1))
    # 90% of scenarios above threshold — empirical 90%, specified 80%
    wealth[:90, -1, 0] = 6_000_000
    wealth[90:, -1, 0] = 4_000_000

    goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=0.80)]
    accounts = [type("Account", (), {"name": "Conservative"})()]

    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))

    s = status[0]
    assert "empirical_probability" in s
    assert "confidence_gap" in s
    assert "note" in s

    assert abs(s["empirical_probability"] - 0.90) < 1e-9
    assert abs(s["confidence_gap"] - 0.10) < 1e-9
    assert "CVaR" in s["note"]


def test_compute_goal_status_dual_metrics_violation_note():
    """Test that violated goal produces warning note."""
    from api.services.simulation import compute_goal_status
    from finopt import TerminalGoal
    import numpy as np

    wealth = np.zeros((100, 25, 1))
    # 70% success, requires 85% → gap = -15%
    wealth[:70, -1, 0] = 6_000_000
    wealth[70:, -1, 0] = 4_000_000

    goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=0.85)]
    accounts = [type("Account", (), {"name": "Conservative"})()]

    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))

    s = status[0]
    assert s["satisfied"] is False
    assert s["confidence_gap"] < 0
    assert "Warning" in s["note"]


def test_compute_goal_status_dual_metrics_mild_conservatism():
    """Test that mild conservatism (gap < 1%) produces simpler note."""
    from api.services.simulation import compute_goal_status
    from finopt import TerminalGoal
    import numpy as np

    # 200 scenarios: 162 pass → 81%, specified 80.5% → gap = 0.5%
    wealth = np.zeros((200, 25, 1))
    wealth[:162, -1, 0] = 6_000_000
    wealth[162:, -1, 0] = 4_000_000

    goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=0.805)]
    accounts = [type("Account", (), {"name": "Conservative"})()]

    status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))

    s = status[0]
    assert s["confidence_gap"] < 0.01
    assert "CVaR constraint satisfied" in s["note"]


@pytest.mark.asyncio
async def test_run_simulation_updates_job_status(mocker, sample_scenario_data, mock_env_vars):
    """Test run_simulation updates job status throughout execution."""
    # Mock Supabase functions
    mocker.patch('api.services.simulation.fetch_scenario_with_profile', return_value=sample_scenario_data)
    mock_update_job = mocker.patch('api.services.simulation.update_job')
    mock_save_result = mocker.patch('api.services.simulation.save_simulation_result')
    
    from api.services.simulation import run_simulation
    
    await run_simulation(scenario_id="test-scenario", job_id="test-job")
    
    # Check that update_job was called multiple times with progress updates
    assert mock_update_job.call_count >= 3
    
    # Check final status is 'completed'
    final_call = mock_update_job.call_args_list[-1]
    assert final_call[1].get("status") == "completed" or final_call[0][1] == "completed"


@pytest.mark.asyncio
async def test_run_simulation_handles_error(mocker, sample_scenario_data, mock_env_vars):
    """Test run_simulation handles errors and marks job as failed."""
    # Mock fetch to raise an error
    mocker.patch('api.services.simulation.fetch_scenario_with_profile', side_effect=ValueError("Test error"))
    mock_update_job = mocker.patch('api.services.simulation.update_job')
    
    from api.services.simulation import run_simulation
    
    with pytest.raises(ValueError):
        await run_simulation(scenario_id="test-scenario", job_id="test-job")
    
    # Should have updated job with failed status
    calls = [call for call in mock_update_job.call_args_list if 'status' in call[1]]
    failed_calls = [call for call in calls if call[1]['status'] == 'failed']
    assert len(failed_calls) > 0
