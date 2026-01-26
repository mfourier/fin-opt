"""
Unit tests for config.py Pydantic models.

Tests validation, defaults, and serialization of configuration classes.
"""

import pytest
from pathlib import Path

from src.config import (
    SimulationConfig,
    OptimizationConfig,
    FixedIncomeConfig,
    VariableIncomeConfig,
    IncomeConfig,
    AccountConfig,
    AppSettings,
)


class TestSimulationConfig:
    """Tests for SimulationConfig validation."""

    def test_defaults(self):
        """Test default values."""
        config = SimulationConfig()

        assert config.n_sims == 500
        assert config.seed is None
        assert config.cache_enabled is True
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom parameter values."""
        config = SimulationConfig(n_sims=1000, seed=42, cache_enabled=False)

        assert config.n_sims == 1000
        assert config.seed == 42
        assert config.cache_enabled is False

    def test_n_sims_validation_min(self):
        """Test n_sims minimum validation."""
        with pytest.raises(ValueError):
            SimulationConfig(n_sims=50)  # Less than 100

    def test_n_sims_validation_max(self):
        """Test n_sims maximum validation."""
        with pytest.raises(ValueError):
            SimulationConfig(n_sims=20_000)  # Greater than 10,000

    def test_immutable(self):
        """Test that config is frozen (immutable)."""
        config = SimulationConfig()

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            config.n_sims = 1000

    def test_serialization(self):
        """Test JSON serialization."""
        config = SimulationConfig(n_sims=1000, seed=42)

        # To dict
        data = config.model_dump()
        assert data["n_sims"] == 1000
        assert data["seed"] == 42

        # From dict
        restored = SimulationConfig.model_validate(data)
        assert restored.n_sims == 1000
        assert restored.seed == 42


class TestOptimizationConfig:
    """Tests for OptimizationConfig validation."""

    def test_defaults(self):
        """Test default values."""
        config = OptimizationConfig()

        assert config.T_max == 240
        assert config.T_min == 12
        assert config.solver == "ECOS"
        assert config.objective == "balanced"
        assert config.search_strategy == "binary"

    def test_T_min_validation(self):
        """Test T_min must be <= T_max."""
        # Valid
        config = OptimizationConfig(T_min=10, T_max=100)
        assert config.T_min == 10

        # Invalid
        with pytest.raises(ValueError, match="T_min.*must be.*T_max"):
            OptimizationConfig(T_min=300, T_max=240)

    def test_solver_validation(self):
        """Test solver must be from allowed list."""
        # Valid
        for solver in ["ECOS", "SCS", "CLARABEL", "MOSEK"]:
            config = OptimizationConfig(solver=solver)
            assert config.solver == solver

        # Invalid
        with pytest.raises(ValueError):
            OptimizationConfig(solver="INVALID")

    def test_objective_validation(self):
        """Test objective must be from allowed list."""
        # Valid
        for obj in ["risky", "balanced", "conservative", "risky_turnover"]:
            config = OptimizationConfig(objective=obj)
            assert config.objective == obj

        # Invalid
        with pytest.raises(ValueError):
            OptimizationConfig(objective="invalid")


class TestIncomeConfig:
    """Tests for income configuration models."""

    def test_fixed_income_config(self):
        """Test FixedIncomeConfig validation."""
        config = FixedIncomeConfig(base=1_500_000, annual_growth=0.03)

        assert config.base == 1_500_000
        assert config.annual_growth == 0.03
        assert config.salary_raises is None

    def test_fixed_income_negative_base(self):
        """Test negative base raises validation error."""
        with pytest.raises(ValueError):
            FixedIncomeConfig(base=-1000)

    def test_variable_income_config(self):
        """Test VariableIncomeConfig validation."""
        config = VariableIncomeConfig(
            base=200_000,
            sigma=0.15,
            annual_growth=0.02
        )

        assert config.base == 200_000
        assert config.sigma == 0.15
        assert config.annual_growth == 0.02

    def test_variable_income_seasonality_validation(self):
        """Test seasonality must have 12 factors."""
        # Valid - 12 factors
        seasonality = [1.0] * 12
        config = VariableIncomeConfig(base=200_000, seasonality=seasonality)
        assert len(config.seasonality) == 12

        # Invalid - wrong length
        with pytest.raises(ValueError, match="12 factors"):
            VariableIncomeConfig(base=200_000, seasonality=[1.0, 1.0, 1.0])

    def test_variable_income_seasonality_values(self):
        """Test seasonality factors must be non-negative (no sum constraint)."""
        # Valid - any non-negative values work (seasonality is a multiplier)
        seasonality = [0.5] * 12  # Sums to 6, but that's fine
        config = VariableIncomeConfig(base=200_000, seasonality=seasonality)
        assert len(config.seasonality) == 12
        assert all(s >= 0 for s in config.seasonality)

        # Valid - zeros are allowed
        seasonality_with_zeros = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
        config2 = VariableIncomeConfig(base=200_000, seasonality=seasonality_with_zeros)
        assert len(config2.seasonality) == 12

        # Invalid - negative values should raise error
        with pytest.raises(ValueError, match="non-negative|negative|>= 0"):
            VariableIncomeConfig(base=200_000, seasonality=[-1.0] + [1.0] * 11)

    def test_income_config_combined(self):
        """Test combined IncomeConfig."""
        fixed = FixedIncomeConfig(base=1_500_000)
        variable = VariableIncomeConfig(base=200_000, sigma=0.1)

        config = IncomeConfig(fixed=fixed, variable=variable)

        assert config.fixed.base == 1_500_000
        assert config.variable.base == 200_000
        assert config.contribution_rate_fixed == 0.3  # Default
        assert config.contribution_rate_variable == 1.0  # Default


class TestAccountConfig:
    """Tests for AccountConfig validation."""

    def test_basic_account(self):
        """Test basic account configuration."""
        config = AccountConfig(
            name="Conservative",
            annual_return=0.04,
            annual_volatility=0.05,
            initial_wealth=1_000_000
        )

        assert config.name == "Conservative"
        assert config.annual_return == 0.04
        assert config.annual_volatility == 0.05
        assert config.initial_wealth == 1_000_000

    def test_name_validation(self):
        """Test name must be non-empty."""
        with pytest.raises(ValueError):
            AccountConfig(
                name="",
                annual_return=0.04,
                annual_volatility=0.05
            )

    def test_return_validation(self):
        """Test return must be in valid range."""
        # Valid
        AccountConfig(name="Test", annual_return=-0.1, annual_volatility=0.05)
        AccountConfig(name="Test", annual_return=0.5, annual_volatility=0.05)

        # Invalid - too low
        with pytest.raises(ValueError):
            AccountConfig(name="Test", annual_return=-0.6, annual_volatility=0.05)

        # Invalid - too high
        with pytest.raises(ValueError):
            AccountConfig(name="Test", annual_return=1.5, annual_volatility=0.05)

    def test_volatility_validation(self):
        """Test volatility must be non-negative."""
        # Valid
        AccountConfig(name="Test", annual_return=0.08, annual_volatility=0.0)

        # Invalid
        with pytest.raises(ValueError):
            AccountConfig(name="Test", annual_return=0.08, annual_volatility=-0.1)


class TestAppSettings:
    """Tests for AppSettings with environment variables."""

    def test_defaults(self):
        """Test default application settings."""
        settings = AppSettings()

        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.max_cache_size_mb == 500
        assert settings.enable_telemetry is False

    def test_custom_settings(self):
        """Test custom settings."""
        settings = AppSettings(
            debug=True,
            log_level="DEBUG",
            max_cache_size_mb=1000
        )

        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.max_cache_size_mb == 1000

    def test_cache_dir_default(self):
        """Test cache directory default."""
        settings = AppSettings()

        assert isinstance(settings.cache_dir, Path)
        assert "finopt" in str(settings.cache_dir)


class TestConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_simulation_config_json(self):
        """Test SimulationConfig JSON roundtrip."""
        config = SimulationConfig(n_sims=1000, seed=42)

        # To JSON
        json_str = config.model_dump_json()
        assert "1000" in json_str
        assert "42" in json_str

        # From JSON
        restored = SimulationConfig.model_validate_json(json_str)
        assert restored.n_sims == 1000
        assert restored.seed == 42

    def test_optimization_config_dict(self):
        """Test OptimizationConfig dict roundtrip."""
        config = OptimizationConfig(
            T_max=120,
            solver="SCS",
            objective="conservative"
        )

        # To dict
        data = config.model_dump()
        assert data["T_max"] == 120
        assert data["solver"] == "SCS"

        # From dict
        restored = OptimizationConfig.model_validate(data)
        assert restored.T_max == 120
        assert restored.solver == "SCS"

    def test_nested_config_serialization(self):
        """Test nested config serialization."""
        fixed = FixedIncomeConfig(base=1_500_000, annual_growth=0.03)
        variable = VariableIncomeConfig(base=200_000, sigma=0.1)
        config = IncomeConfig(fixed=fixed, variable=variable)

        # To dict
        data = config.model_dump()
        assert data["fixed"]["base"] == 1_500_000
        assert data["variable"]["sigma"] == 0.1

        # From dict
        restored = IncomeConfig.model_validate(data)
        assert restored.fixed.base == 1_500_000
        assert restored.variable.sigma == 0.1
