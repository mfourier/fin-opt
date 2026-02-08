"""
Tests for API Configuration (api/config.py)

Tests cover:
- Settings validation with complete environment variables
- Default values when optional vars are not set
- CORS origins parser (comma-separated string to list)
- Environment property helpers (is_development, is_production)
- Error handling for missing required variables
- Settings caching behavior
"""

import pytest
from pydantic import ValidationError


def test_settings_from_complete_env(mock_env_vars):
    """Test Settings loads correctly with all environment variables set."""
    from api.config import get_settings
    
    settings = get_settings()
    
    assert settings.supabase_url == "https://test.supabase.co"
    assert settings.supabase_anon_key == "test-anon-key"
    assert settings.supabase_service_key == "test-service-key"
    assert settings.environment == "development"
    assert settings.cors_origins == ["http://localhost:3000", "http://localhost:5173"]


def test_settings_defaults(mock_env_vars):
    """Test Settings uses default values for optional fields."""
    from api.config import get_settings
    
    settings = get_settings()
    
    # Default values
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.log_level == "INFO"
    assert settings.debug is False


def test_cors_origins_parser_string(monkeypatch):
    """Test CORS origins parser converts comma-separated string to list."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "key1")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key2")
    monkeypatch.setenv("CORS_ORIGINS_STR", "http://example.com,https://app.example.com, http://localhost:3000")
    
    from api.config import get_settings, Settings
    get_settings.cache_clear()
    
    settings = get_settings()
    
    # Should parse and trim whitespace
    assert settings.cors_origins == [
        "http://example.com",
        "https://app.example.com",
        "http://localhost:3000"
    ]
    
    get_settings.cache_clear()


def test_cors_origins_parser_list(monkeypatch):
    """Test CORS origins property returns list from string field."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "key1")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key2")
    monkeypatch.setenv("CORS_ORIGINS_STR", "http://example.com,https://app.example.com")
    
    from api.config import Settings, get_settings
    get_settings.cache_clear()
    
    settings = get_settings()
    
    assert isinstance(settings.cors_origins, list)
    assert settings.cors_origins == ["http://example.com", "https://app.example.com"]
    
    get_settings.cache_clear()



def test_environment_property_development(mock_env_vars):
    """Test is_development property returns True in development mode."""
    from api.config import get_settings
    
    settings = get_settings()
    
    assert settings.environment == "development"
    assert settings.is_development is True
    assert settings.is_production is False


def test_environment_property_production(monkeypatch):
    """Test is_production property returns True in production mode."""
    monkeypatch.setenv("SUPABASE_URL", "https://prod.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "prod-anon")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "prod-service")
    monkeypatch.setenv("ENVIRONMENT", "production")
    
    from api.config import get_settings
    get_settings.cache_clear()
    
    settings = get_settings()
    
    assert settings.environment == "production"
    assert settings.is_production is True
    assert settings.is_development is False
    
    get_settings.cache_clear()


def test_missing_required_var_raises_error(monkeypatch):
    """Test Settings raises ValidationError when required vars are missing."""
    # Only set some required vars, missing others
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    from api.config import Settings, get_settings
    get_settings.cache_clear()

    # Disable .env file loading so only actual env vars are used
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    # Check error mentions missing fields
    error_str = str(exc_info.value)
    assert "supabase_anon_key" in error_str.lower() or "Field required" in error_str

    get_settings.cache_clear()


def test_get_settings_cached(mock_env_vars):
    """Test get_settings() returns the same instance (cached)."""
    from api.config import get_settings
    
    # Clear cache first
    get_settings.cache_clear()
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Should be the exact same object (cached)
    assert settings1 is settings2
    
    get_settings.cache_clear()


def test_api_port_validation(monkeypatch):
    """Test api_port validates range (1-65535)."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "key1")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key2")
    monkeypatch.setenv("API_PORT", "99999")  # Invalid port
    
    from api.config import Settings, get_settings
    get_settings.cache_clear()
    
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    
    error_str = str(exc_info.value)
    assert "api_port" in error_str.lower()
    
    get_settings.cache_clear()
