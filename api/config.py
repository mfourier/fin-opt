"""
API Configuration

Uses pydantic-settings for type-safe environment variable management.
All settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or via a .env file.
    The .env file is loaded automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Disable JSON parsing for environment variables
        # This allows our custom validator to handle list fields
        env_parse_none_str="null",
    )

    # -------------------------------------------------------------------------
    # Supabase Configuration
    # -------------------------------------------------------------------------
    supabase_url: str = Field(
        ...,
        description="Supabase project URL (e.g., https://xxx.supabase.co)"
    )
    supabase_anon_key: str = Field(
        ...,
        description="Supabase anon/public key (for RLS-protected operations)"
    )
    supabase_service_key: str = Field(
        ...,
        description="Supabase service role key (bypasses RLS, keep secret)"
    )

    # -------------------------------------------------------------------------
    # API Server Configuration
    # -------------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind the API server"
    )

    # -------------------------------------------------------------------------
    # CORS Configuration
    # -------------------------------------------------------------------------
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Allowed CORS origins (comma-separated string)"
    )

    @property
    def cors_origins(self) -> list[str]:
        """Get CORS origins as a list."""
        if isinstance(self.cors_origins_str, list):
            return self.cors_origins_str
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose errors)"
    )

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    In tests, you can clear the cache with get_settings.cache_clear().

    Returns
    -------
    Settings
        Application settings loaded from environment.

    Raises
    ------
    ValidationError
        If required environment variables are missing or invalid.
    """
    return Settings()
