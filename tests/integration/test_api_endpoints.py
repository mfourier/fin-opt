"""
Tests for API Endpoints (api/main.py)

Tests cover:
- /health endpoint
- /simulate endpoint (valid and invalid requests)
- /optimize endpoint (valid and invalid requests)
- CORS headers
- Error handlers
- Request/response schemas

Uses FastAPI TestClient which doesn't require a running server.
"""



def test_health_endpoint(fastapi_test_client):
    """Test /health endpoint returns correct structure."""
    response = fastapi_test_client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert data["environment"] == "development"


def test_health_endpoint_cors_headers(fastapi_test_client):
    """Test /health endpoint includes CORS headers."""
    # CORS headers only appear when Origin header is present in the request
    response = fastapi_test_client.get("/health", headers={"Origin": "http://localhost:3000"})

    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers


def test_simulate_valid_request(fastapi_test_client, mocker):
    """Test /simulate endpoint accepts valid request and queues job."""
    # Mock background task execution
    mock_run_simulation = mocker.patch('api.main.run_simulation')

    request_data = {
        "scenario_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_id": "660e8400-e29b-41d4-a716-446655440000"
    }

    response = fastapi_test_client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"
    assert data["job_id"] == request_data["job_id"]
    assert "message" in data


def test_simulate_invalid_request_missing_fields(fastapi_test_client):
    """Test /simulate endpoint rejects request with missing fields."""
    request_data = {
        "scenario_id": "550e8400-e29b-41d4-a716-446655440000"
        # Missing job_id
    }

    response = fastapi_test_client.post("/simulate", json=request_data)

    # Should return 422 Unprocessable Entity (Pydantic validation error)
    assert response.status_code == 422


def test_simulate_invalid_request_wrong_types(fastapi_test_client):
    """Test /simulate endpoint rejects request with wrong field types."""
    request_data = {
        "scenario_id": 12345,  # Should be string
        "job_id": "660e8400-e29b-41d4-a716-446655440000"
    }

    response = fastapi_test_client.post("/simulate", json=request_data)

    assert response.status_code == 422


def test_optimize_valid_request(fastapi_test_client, mocker):
    """Test /optimize endpoint accepts valid request and queues job."""
    # Mock background task execution
    mock_run_optimization = mocker.patch('api.main.run_optimization')

    request_data = {
        "scenario_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_id": "660e8400-e29b-41d4-a716-446655440000"
    }

    response = fastapi_test_client.post("/optimize", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "queued"
    assert data["job_id"] == request_data["job_id"]


def test_optimize_invalid_request(fastapi_test_client):
    """Test /optimize endpoint rejects invalid request."""
    request_data = {}  # Empty request

    response = fastapi_test_client.post("/optimize", json=request_data)

    assert response.status_code == 422


def test_cors_preflight_request(fastapi_test_client):
    """Test CORS preflight (OPTIONS) request is handled."""
    # Preflight requires Origin header to trigger CORS
    response = fastapi_test_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        }
    )

    # Should allow OPTIONS
    assert response.status_code in [200, 204]


def test_cors_headers_present(fastapi_test_client):
    """Test CORS headers are present in responses."""
    # CORS headers only appear when Origin header is present
    response = fastapi_test_client.get("/health", headers={"Origin": "http://localhost:3000"})

    headers = response.headers

    # Check for CORS headers
    assert "access-control-allow-origin" in headers


def test_endpoint_not_found_returns_404(fastapi_test_client):
    """Test requesting non-existent endpoint returns 404."""
    response = fastapi_test_client.get("/nonexistent")

    assert response.status_code == 404


def test_method_not_allowed_returns_405(fastapi_test_client):
    """Test using wrong HTTP method returns 405."""
    # /health only accepts GET
    response = fastapi_test_client.post("/health")

    assert response.status_code == 405


def test_simulate_response_schema(fastapi_test_client, mocker):
    """Test /simulate response matches JobResponse schema."""
    mocker.patch('api.main.run_simulation')

    request_data = {
        "scenario_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_id": "660e8400-e29b-41d4-a716-446655440000"
    }

    response = fastapi_test_client.post("/simulate", json=request_data)
    data = response.json()

    # Verify response schema
    assert "status" in data
    assert "job_id" in data
    assert "message" in data
    assert isinstance(data["status"], str)
    assert isinstance(data["job_id"], str)
    assert isinstance(data["message"], str)


def test_optimize_response_schema(fastapi_test_client, mocker):
    """Test /optimize response matches JobResponse schema."""
    mocker.patch('api.main.run_optimization')

    request_data = {
        "scenario_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_id": "660e8400-e29b-41d4-a716-446655440000"
    }

    response = fastapi_test_client.post("/optimize", json=request_data)
    data = response.json()

    # Verify response schema
    assert "status" in data
    assert "job_id" in data
    assert "message" in data
