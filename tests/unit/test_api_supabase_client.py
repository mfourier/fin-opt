"""
Tests for API Supabase Client (api/supabase_client.py)

Tests cover:
- Supabase client creation with service role key
- update_job() with various status transitions
- save_optimization_result() data structure
- save_simulation_result() data structure
- fetch_scenario_with_profile() query and error handling

All tests mock Supabase calls to avoid requiring a real database.
"""

from unittest.mock import MagicMock

import pytest


def test_get_supabase_client_uses_service_key(mocker, mock_env_vars):
    """Test get_supabase_client creates client with service role key."""
    mock_create_client = mocker.patch('api.supabase_client.create_client')

    from api.config import get_settings
    from api.supabase_client import get_supabase_client

    # Clear cache and get client
    get_supabase_client.cache_clear()
    client = get_supabase_client()

    # Should have called create_client with service key
    settings = get_settings()
    mock_create_client.assert_called_once_with(
        settings.supabase_url,
        settings.supabase_service_key
    )

    get_supabase_client.cache_clear()


def test_get_supabase_client_cached(mocker, mock_env_vars):
    """Test get_supabase_client returns cached instance."""
    mock_create_client = mocker.patch('api.supabase_client.create_client')

    from api.supabase_client import get_supabase_client

    get_supabase_client.cache_clear()

    client1 = get_supabase_client()
    client2 = get_supabase_client()

    # Should only create client once
    assert mock_create_client.call_count == 1

    get_supabase_client.cache_clear()


def test_update_job_sets_status(mocker, mock_env_vars):
    """Test update_job updates job status."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import update_job

    update_job(job_id="test-job-id", status="running")

    # Should call update with status
    mock_client.table.assert_called_once_with("jobs")
    update_call = mock_client.table().update
    update_call.assert_called_once()

    # Check update data includes status
    update_data = update_call.call_args[0][0]
    assert update_data["status"] == "running"


def test_update_job_running_sets_started_at(mocker, mock_env_vars):
    """Test update_job sets started_at timestamp when status is 'running'."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import update_job

    update_job(job_id="test-job-id", status="running")

    update_data = mock_client.table().update.call_args[0][0]

    assert "status" in update_data
    assert update_data["status"] == "running"
    assert "started_at" in update_data


def test_update_job_completed_sets_completed_at(mocker, mock_env_vars):
    """Test update_job sets completed_at when status is 'completed'."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import update_job

    update_job(job_id="test-job-id", status="completed")

    update_data = mock_client.table().update.call_args[0][0]

    assert update_data["status"] == "completed"
    assert "completed_at" in update_data


def test_update_job_progress_clamped(mocker, mock_env_vars):
    """Test update_job clamps progress to 0-100 range."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import update_job

    # Test progress > 100
    update_job(job_id="test-job", progress=150)
    update_data = mock_client.table().update.call_args[0][0]
    assert update_data["progress"] == 100

    # Test progress < 0
    update_job(job_id="test-job", progress=-10)
    update_data = mock_client.table().update.call_args[0][0]
    assert update_data["progress"] == 0


def test_update_job_with_step_and_error(mocker, mock_env_vars):
    """Test update_job handles step and error_message."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import update_job

    update_job(
        job_id="test-job",
        status="failed",
        step="Loading model",
        error_message="Model not found"
    )

    update_data = mock_client.table().update.call_args[0][0]

    assert update_data["status"] == "failed"
    assert update_data["current_step"] == "Loading model"
    assert update_data["error_message"] == "Model not found"


def test_save_optimization_result_structure(mocker, mock_env_vars):
    """Test save_optimization_result inserts correct data structure."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import save_optimization_result

    allocation = [[0.6, 0.4], [0.5, 0.5]]
    goal_status = [{"goal": "Test", "satisfied": True}]
    diagnostics = {"solver": "ECOS", "iterations": 10}

    save_optimization_result(
        job_id="test-job",
        allocation_policy=allocation,
        optimal_horizon=24,
        objective_value=1000.0,
        feasible=True,
        solve_time=5.5,
        diagnostics=diagnostics,
        goal_status=goal_status
    )

    # Check insert was called with results table
    mock_client.table.assert_called_once_with("results")
    insert_data = mock_client.table().insert.call_args[0][0]

    assert insert_data["job_id"] == "test-job"
    assert insert_data["result_type"] == "optimization"
    assert insert_data["allocation_policy"] == allocation
    assert insert_data["optimal_horizon"] == 24
    assert insert_data["objective_value"] == 1000.0
    assert insert_data["feasible"] is True
    assert insert_data["solve_time"] == 5.5
    assert insert_data["diagnostics"] == diagnostics
    assert insert_data["goal_status"] == goal_status


def test_save_simulation_result_structure(mocker, mock_env_vars):
    """Test save_simulation_result inserts correct data structure."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import save_simulation_result

    summary_stats = {
        "total_wealth": {"mean": 10000000, "median": 9500000},
        "accounts": [{"mean": 6000000}, {"mean": 4000000}]
    }
    goal_status = [{"goal": "Emergency", "probability": 0.85}]

    save_simulation_result(
        job_id="sim-job",
        summary_stats=summary_stats,
        goal_status=goal_status
    )

    insert_data = mock_client.table().insert.call_args[0][0]

    assert insert_data["job_id"] == "sim-job"
    assert insert_data["result_type"] == "simulation"
    assert insert_data["summary_stats"] == summary_stats
    assert insert_data["goal_status"] == goal_status


def test_fetch_scenario_with_profile_success(mocker, mock_env_vars):
    """Test fetch_scenario_with_profile returns scenario data."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = {"id": "scenario-123", "profiles": {"income_config": {}}}

    # Setup chain: table().select().eq().single().execute()
    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_response

    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import fetch_scenario_with_profile

    result = fetch_scenario_with_profile("scenario-123")

    assert result == {"id": "scenario-123", "profiles": {"income_config": {}}}

    # Verify query structure
    mock_client.table.assert_called_once_with("scenarios")
    mock_client.table().select.assert_called_once_with("*, profiles(*)")


def test_fetch_scenario_with_profile_not_found_raises(mocker, mock_env_vars):
    """Test fetch_scenario_with_profile raises ValueError when scenario not found."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = None  # No data found

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = mock_response

    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import fetch_scenario_with_profile

    with pytest.raises(ValueError) as exc_info:
        fetch_scenario_with_profile("nonexistent-id")

    assert "not found" in str(exc_info.value).lower()
    assert "nonexistent-id" in str(exc_info.value)


def test_authorize_job_for_user_success(mocker, mock_env_vars):
    """Test authorize_job_for_user accepts a matching pending job."""
    mock_client = MagicMock()
    job_response = MagicMock()
    scenario_response = MagicMock()
    job_response.data = {
        "id": "job-123",
        "scenario_id": "scenario-123",
        "job_type": "optimization",
        "status": "pending",
    }
    scenario_response.data = {
        "id": "scenario-123",
        "is_demo": False,
        "profiles": {"user_id": "user-123"},
    }

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.side_effect = [
        job_response,
        scenario_response,
    ]
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import authorize_job_for_user

    authorize_job_for_user("job-123", "scenario-123", "user-123", "optimization")


def test_authorize_job_for_user_rejects_wrong_owner(mocker, mock_env_vars):
    """Test authorize_job_for_user rejects access to another user's scenario."""
    mock_client = MagicMock()
    job_response = MagicMock()
    scenario_response = MagicMock()
    job_response.data = {
        "id": "job-123",
        "scenario_id": "scenario-123",
        "job_type": "optimization",
        "status": "pending",
    }
    scenario_response.data = {
        "id": "scenario-123",
        "is_demo": False,
        "profiles": {"user_id": "someone-else"},
    }

    mock_client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.side_effect = [
        job_response,
        scenario_response,
    ]
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    from api.supabase_client import authorize_job_for_user

    with pytest.raises(PermissionError, match="do not have access"):
        authorize_job_for_user("job-123", "scenario-123", "user-123", "optimization")


def test_reap_orphaned_jobs_marks_running_and_pending_failed(mocker, mock_env_vars):
    """reap_orphaned_jobs marks pending/running jobs failed and returns the count."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    resp = MagicMock()
    resp.data = [{"id": "j1"}, {"id": "j2"}]
    table = mock_client.table.return_value
    table.update.return_value.in_.return_value.execute.return_value = resp

    from api.supabase_client import reap_orphaned_jobs

    n = reap_orphaned_jobs()

    assert n == 2
    mock_client.table.assert_called_with("jobs")

    update_data = table.update.call_args[0][0]
    assert update_data["status"] == "failed"
    assert "error_message" in update_data

    in_args = table.update.return_value.in_.call_args[0]
    assert in_args[0] == "status"
    assert set(in_args[1]) == {"pending", "running"}

    # default older_than_seconds=0 -> no created_at age filter
    table.update.return_value.in_.return_value.lt.assert_not_called()


def test_reap_orphaned_jobs_age_filter(mocker, mock_env_vars):
    """older_than_seconds adds a created_at cutoff to the query."""
    mock_client = MagicMock()
    mocker.patch('api.supabase_client.get_supabase_client', return_value=mock_client)

    resp = MagicMock()
    resp.data = []
    table = mock_client.table.return_value
    table.update.return_value.in_.return_value.lt.return_value.execute.return_value = resp

    from api.supabase_client import reap_orphaned_jobs

    n = reap_orphaned_jobs(older_than_seconds=600)

    assert n == 0
    lt_args = table.update.return_value.in_.return_value.lt.call_args[0]
    assert lt_args[0] == "created_at"
