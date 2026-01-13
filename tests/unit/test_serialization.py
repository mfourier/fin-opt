"""
Unit tests for serialization.py module.

Tests model serialization and deserialization.
"""

import pytest
import numpy as np
import json
from pathlib import Path
from datetime import date
import tempfile

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.model import FinancialModel
from src.serialization import (
    SCHEMA_VERSION,
    account_to_dict,
    account_from_dict,
    income_to_dict,
    income_from_dict,
    save_model,
    load_model,
)


# ============================================================================
# ACCOUNT SERIALIZATION TESTS
# ============================================================================

class TestAccountSerialization:
    """Test account serialization."""

    def test_account_to_dict(self):
        """Test account_to_dict returns correct structure."""
        acc = Account.from_annual("Emergency", annual_return=0.04,
                                   annual_volatility=0.05, initial_wealth=1_000_000)

        result = account_to_dict(acc)

        assert "name" in result
        assert "annual_return" in result
        assert "annual_volatility" in result
        assert "initial_wealth" in result
        assert result["name"] == "Emergency"
        assert result["initial_wealth"] == 1_000_000

    def test_account_from_dict(self):
        """Test account_from_dict reconstructs Account."""
        data = {
            "name": "Emergency",
            "annual_return": 0.04,
            "annual_volatility": 0.05,
            "initial_wealth": 1_000_000,
        }

        acc = account_from_dict(data)

        assert acc.name == "Emergency"
        assert acc.initial_wealth == 1_000_000
        assert acc.annual_params["return"] == pytest.approx(0.04, rel=0.01)

    def test_account_roundtrip(self):
        """Test account serialization roundtrip."""
        original = Account.from_annual("Test", 0.08, 0.12, 500_000)

        serialized = account_to_dict(original)
        recovered = account_from_dict(serialized)

        assert recovered.name == original.name
        assert recovered.initial_wealth == original.initial_wealth
        assert recovered.annual_params["return"] == pytest.approx(
            original.annual_params["return"], rel=0.01
        )


# ============================================================================
# INCOME SERIALIZATION TESTS
# ============================================================================

class TestIncomeSerialization:
    """Test income serialization."""

    def test_income_to_dict_fixed_only(self):
        """Test income_to_dict with fixed income only."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000, annual_growth=0.05),
            variable=None
        )

        result = income_to_dict(income)

        assert "fixed" in result
        assert result["fixed"]["base"] == 1_000_000
        assert result["fixed"]["annual_growth"] == 0.05

    def test_income_to_dict_with_variable(self):
        """Test income_to_dict with variable income."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
        )

        result = income_to_dict(income)

        assert "fixed" in result
        assert "variable" in result
        assert result["variable"]["base"] == 200_000
        assert result["variable"]["sigma"] == 0.1

    def test_income_from_dict_fixed_only(self):
        """Test income_from_dict with fixed only."""
        data = {
            "fixed": {"base": 1_000_000, "annual_growth": 0.05},
            "contribution_rate_fixed": 0.3,
            "contribution_rate_variable": 1.0,
        }

        income = income_from_dict(data)

        assert income.fixed is not None
        assert income.fixed.base == 1_000_000

    def test_income_from_dict_with_variable(self):
        """Test income_from_dict with variable."""
        data = {
            "fixed": {"base": 1_000_000, "annual_growth": 0.05},
            "variable": {"base": 200_000, "sigma": 0.1, "annual_growth": 0.03},
            "contribution_rate_fixed": 0.3,
            "contribution_rate_variable": 1.0,
        }

        income = income_from_dict(data)

        assert income.fixed is not None
        assert income.variable is not None
        assert income.variable.sigma == 0.1

    def test_income_roundtrip(self):
        """Test income serialization roundtrip."""
        original = IncomeModel(
            fixed=FixedIncome(base=1_500_000, annual_growth=0.04),
            variable=VariableIncome(base=200_000, sigma=0.15)
        )

        serialized = income_to_dict(original)
        recovered = income_from_dict(serialized)

        assert recovered.fixed.base == original.fixed.base
        assert recovered.variable.sigma == original.variable.sigma


# ============================================================================
# FINANCIALMODEL SERIALIZATION TESTS
# ============================================================================

class TestModelSerialization:
    """Test FinancialModel serialization."""

    @pytest.fixture
    def sample_model(self):
        """Create sample FinancialModel."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
            variable=VariableIncome(base=200_000, sigma=0.1)
        )
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return FinancialModel(income=income, accounts=accounts)

    def test_save_model(self, sample_model, tmp_path):
        """Test save_model creates JSON file."""
        path = tmp_path / "config.json"

        save_model(sample_model, path)

        assert path.exists()
        content = json.loads(path.read_text())
        assert "schema_version" in content
        assert "income" in content
        assert "accounts" in content

    def test_save_model_schema_version(self, sample_model, tmp_path):
        """Test save_model includes correct schema version."""
        path = tmp_path / "config.json"

        save_model(sample_model, path)

        content = json.loads(path.read_text())
        assert content["schema_version"] == SCHEMA_VERSION

    def test_load_model(self, sample_model, tmp_path):
        """Test load_model reconstructs FinancialModel."""
        path = tmp_path / "config.json"
        save_model(sample_model, path)

        loaded = load_model(path)

        assert isinstance(loaded, FinancialModel)
        assert len(loaded.accounts) == 2
        assert loaded.income.fixed is not None

    def test_model_roundtrip(self, sample_model, tmp_path):
        """Test model serialization roundtrip."""
        path = tmp_path / "config.json"

        save_model(sample_model, path)
        loaded = load_model(path)

        # Verify accounts
        assert len(loaded.accounts) == len(sample_model.accounts)
        for orig, load in zip(sample_model.accounts, loaded.accounts):
            assert load.name == orig.name

        # Verify income
        assert loaded.income.fixed.base == sample_model.income.fixed.base

    def test_save_model_creates_directory(self, sample_model, tmp_path):
        """Test save_model creates parent directory."""
        path = tmp_path / "subdir" / "config.json"

        save_model(sample_model, path)

        assert path.exists()

    def test_save_model_with_correlation(self, sample_model, tmp_path):
        """Test save_model includes correlation matrix."""
        path = tmp_path / "config.json"

        save_model(sample_model, path, include_correlation=True)

        content = json.loads(path.read_text())
        assert "correlation" in content


# ============================================================================
# SCHEMA VERSION TESTS
# ============================================================================

class TestSchemaVersion:
    """Test schema versioning."""

    def test_schema_version_format(self):
        """Test schema version is valid semver."""
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_load_model_warns_on_version_mismatch(self, tmp_path):
        """Test load_model warns when schema version differs."""
        # Create config with different version
        config = {
            "schema_version": "0.0.1",
            "income": {
                "fixed": {"base": 1_000_000, "annual_growth": 0.0},
                "contribution_rate_fixed": 0.3,
                "contribution_rate_variable": 1.0,
            },
            "accounts": [
                {
                    "name": "Test",
                    "annual_return": 0.08,
                    "annual_volatility": 0.10,
                    "initial_wealth": 0,
                }
            ],
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        with pytest.warns(UserWarning, match="schema version"):
            load_model(path)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_income_with_seasonality(self):
        """Test serialization preserves seasonality."""
        seasonality = [1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.0, 0.8, 0.9, 1.0, 1.05, 1.15]
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, seasonality=seasonality)
        )

        serialized = income_to_dict(income)

        assert "seasonality" in serialized["variable"]
        assert len(serialized["variable"]["seasonality"]) == 12

    def test_income_with_floor_cap(self):
        """Test serialization preserves floor and cap."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, floor=100_000, cap=300_000)
        )

        serialized = income_to_dict(income)

        assert serialized["variable"]["floor"] == 100_000
        assert serialized["variable"]["cap"] == 300_000

    def test_account_zero_initial_wealth(self):
        """Test serialization handles zero initial wealth."""
        acc = Account.from_annual("Test", 0.08, 0.10, initial_wealth=0)

        serialized = account_to_dict(acc)
        recovered = account_from_dict(serialized)

        assert recovered.initial_wealth == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
