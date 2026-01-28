"""
Unit tests for serialization.py module.

Tests model serialization and deserialization.
"""

import pytest
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import date
import tempfile

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.model import FinancialModel
from src.goals import IntermediateGoal, TerminalGoal
from src.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent, StochasticWithdrawal
from src.serialization import (
    SCHEMA_VERSION,
    account_to_dict,
    account_from_dict,
    income_to_dict,
    income_from_dict,
    save_model,
    load_model,
    withdrawal_to_dict,
    withdrawal_from_dict,
    goals_to_dict,
    goals_from_dict,
    save_scenario,
    load_scenario,
    save_optimization_result,
    load_optimization_result,
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

    def test_income_with_salary_raises(self):
        """Test serialization preserves salary raises."""
        raises = {date(2025, 6, 1): 0.10, date(2026, 1, 1): 0.05}
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000, salary_raises=raises),
            variable=None
        )

        serialized = income_to_dict(income)

        assert "salary_raises" in serialized["fixed"]
        # Dates should be serialized as ISO strings
        assert "2025-06-01" in serialized["fixed"]["salary_raises"]
        assert serialized["fixed"]["salary_raises"]["2025-06-01"] == 0.10

    def test_income_with_numpy_array_rates(self):
        """Test serialization handles numpy array contribution rates."""
        fixed_rates = np.array([0.3] * 12)
        var_rates = np.array([1.0] * 12)
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, sigma=0.1),
            monthly_contribution={"fixed": fixed_rates, "variable": var_rates}
        )

        serialized = income_to_dict(income)

        # Should be converted to lists
        assert isinstance(serialized["contribution_rate_fixed"], list)
        assert len(serialized["contribution_rate_fixed"]) == 12


# ============================================================================
# WITHDRAWAL SERIALIZATION TESTS
# ============================================================================

class TestWithdrawalSerialization:
    """Test withdrawal serialization."""

    def test_scheduled_withdrawal_to_dict(self):
        """Test withdrawal_to_dict with scheduled events."""
        schedule = WithdrawalSchedule([
            WithdrawalEvent("Conservative", 100_000, date(2025, 6, 1)),
            WithdrawalEvent("Aggressive", 50_000, date(2025, 12, 1), description="Holiday"),
        ])
        model = WithdrawalModel(scheduled=schedule)

        result = withdrawal_to_dict(model)

        assert "scheduled" in result
        assert len(result["scheduled"]) == 2
        assert result["scheduled"][0]["account"] == "Conservative"
        assert result["scheduled"][0]["amount"] == 100_000
        assert result["scheduled"][0]["date"] == "2025-06-01"
        assert result["scheduled"][1]["description"] == "Holiday"

    def test_scheduled_withdrawal_from_dict(self):
        """Test withdrawal_from_dict with scheduled events."""
        data = {
            "scheduled": [
                {"account": "Conservative", "amount": 100000, "date": "2025-06-01"},
            ],
            "stochastic": []
        }

        model = withdrawal_from_dict(data)

        assert model.scheduled is not None
        assert len(model.scheduled.events) == 1
        assert model.scheduled.events[0].account == "Conservative"
        assert model.scheduled.events[0].date == date(2025, 6, 1)

    def test_scheduled_withdrawal_roundtrip(self):
        """Test scheduled withdrawal roundtrip serialization."""
        original = WithdrawalModel(
            scheduled=WithdrawalSchedule([
                WithdrawalEvent("Test", 200_000, date(2025, 7, 1), description="Test event")
            ])
        )

        serialized = withdrawal_to_dict(original)
        recovered = withdrawal_from_dict(serialized)

        assert recovered.scheduled is not None
        assert len(recovered.scheduled.events) == 1
        assert recovered.scheduled.events[0].amount == 200_000
        assert recovered.scheduled.events[0].description == "Test event"

    def test_stochastic_withdrawal_with_month(self):
        """Test stochastic withdrawal with month field."""
        stochastic = StochasticWithdrawal(
            account="Conservative",
            base_amount=50_000,
            sigma=0.2,
            month=6,
            floor=10_000
        )
        model = WithdrawalModel(stochastic=[stochastic])

        serialized = withdrawal_to_dict(model)

        assert len(serialized["stochastic"]) == 1
        assert serialized["stochastic"][0]["month"] == 6
        assert "date" not in serialized["stochastic"][0]

    def test_stochastic_withdrawal_with_date(self):
        """Test stochastic withdrawal with date field."""
        stochastic = StochasticWithdrawal(
            account="Aggressive",
            base_amount=100_000,
            sigma=0.15,
            date=date(2025, 8, 1),
            floor=20_000
        )
        model = WithdrawalModel(stochastic=[stochastic])

        serialized = withdrawal_to_dict(model)

        assert len(serialized["stochastic"]) == 1
        assert serialized["stochastic"][0]["date"] == "2025-08-01"
        assert "month" not in serialized["stochastic"][0]

    def test_stochastic_withdrawal_with_optional_fields(self):
        """Test stochastic withdrawal with cap and seed."""
        stochastic = StochasticWithdrawal(
            account="Test",
            base_amount=75_000,
            sigma=0.1,
            month=3,
            floor=10_000,
            cap=150_000,
            seed=42
        )
        model = WithdrawalModel(stochastic=[stochastic])

        serialized = withdrawal_to_dict(model)
        recovered = withdrawal_from_dict(serialized)

        assert recovered.stochastic[0].cap == 150_000
        assert recovered.stochastic[0].seed == 42

    def test_withdrawal_model_roundtrip(self):
        """Test complete withdrawal model roundtrip."""
        original = WithdrawalModel(
            scheduled=WithdrawalSchedule([
                WithdrawalEvent("Acc1", 100_000, date(2025, 6, 1))
            ]),
            stochastic=[
                StochasticWithdrawal("Acc2", 50_000, 0.1, date=date(2025, 9, 1), floor=10_000)
            ]
        )

        serialized = withdrawal_to_dict(original)
        recovered = withdrawal_from_dict(serialized)

        assert recovered.scheduled is not None
        assert len(recovered.scheduled.events) == 1
        assert len(recovered.stochastic) == 1

    def test_empty_withdrawal_model(self):
        """Test empty withdrawal model serialization."""
        model = WithdrawalModel(scheduled=None, stochastic=None)

        serialized = withdrawal_to_dict(model)
        recovered = withdrawal_from_dict(serialized)

        assert serialized["scheduled"] == []
        assert serialized["stochastic"] == []
        assert recovered.scheduled is None
        assert recovered.stochastic is None


# ============================================================================
# GOAL SERIALIZATION TESTS
# ============================================================================

class TestGoalSerialization:
    """Test goal serialization."""

    def test_intermediate_goal_to_dict(self):
        """Test goals_to_dict with intermediate goal."""
        goals = [
            IntermediateGoal(
                account="Savings",
                threshold=3_000_000,
                confidence=0.9,
                date=date(2025, 7, 1)
            )
        ]

        result = goals_to_dict(goals)

        assert len(result["intermediate"]) == 1
        assert result["intermediate"][0]["account"] == "Savings"
        assert result["intermediate"][0]["threshold"] == 3_000_000
        assert result["intermediate"][0]["date"] == "2025-07-01"

    def test_intermediate_goal_from_dict(self):
        """Test goals_from_dict with intermediate goal."""
        data = {
            "intermediate": [
                {"account": "Savings", "threshold": 3000000, "confidence": 0.9,
                 "date": "2025-07-01"}
            ],
            "terminal": []
        }

        goals = goals_from_dict(data)

        assert len(goals) == 1
        assert isinstance(goals[0], IntermediateGoal)
        assert goals[0].date == date(2025, 7, 1)

    def test_intermediate_goal_roundtrip(self):
        """Test intermediate goal roundtrip."""
        original = [
            IntermediateGoal("Test", 5_000_000, 0.85, date=date(2025, 12, 1))
        ]

        serialized = goals_to_dict(original)
        recovered = goals_from_dict(serialized)

        assert len(recovered) == 1
        assert recovered[0].threshold == 5_000_000
        assert recovered[0].confidence == 0.85

    def test_terminal_goal_to_dict(self):
        """Test goals_to_dict with terminal goal."""
        goals = [
            TerminalGoal(account="Investment", threshold=50_000_000, confidence=0.8)
        ]

        result = goals_to_dict(goals)

        assert len(result["terminal"]) == 1
        assert result["terminal"][0]["account"] == "Investment"
        assert "date" not in result["terminal"][0]

    def test_terminal_goal_from_dict(self):
        """Test goals_from_dict with terminal goal."""
        data = {
            "intermediate": [],
            "terminal": [
                {"account": "Retirement", "threshold": 100000000, "confidence": 0.75}
            ]
        }

        goals = goals_from_dict(data)

        assert len(goals) == 1
        assert isinstance(goals[0], TerminalGoal)
        assert goals[0].threshold == 100_000_000

    def test_terminal_goal_roundtrip(self):
        """Test terminal goal roundtrip."""
        original = [TerminalGoal("Acc", 25_000_000, 0.9)]

        serialized = goals_to_dict(original)
        recovered = goals_from_dict(serialized)

        assert len(recovered) == 1
        assert recovered[0].threshold == 25_000_000

    def test_mixed_goals_roundtrip(self):
        """Test mixed intermediate and terminal goals roundtrip."""
        original = [
            IntermediateGoal("Emergency", 5_000_000, 0.95, date=date(2025, 6, 1)),
            IntermediateGoal("House", 20_000_000, 0.80, date=date(2027, 1, 1)),
            TerminalGoal("Retirement", 100_000_000, 0.75),
        ]

        serialized = goals_to_dict(original)
        recovered = goals_from_dict(serialized)

        assert len(recovered) == 3
        # Order: intermediate first, then terminal
        intermediate = [g for g in recovered if isinstance(g, IntermediateGoal)]
        terminal = [g for g in recovered if isinstance(g, TerminalGoal)]
        assert len(intermediate) == 2
        assert len(terminal) == 1

    def test_goal_from_dict_backward_compat_month(self):
        """Test backward compatibility for old month format."""
        data = {
            "intermediate": [
                {"account": "Old", "threshold": 1000000, "confidence": 0.8, "month": 6}
            ],
            "terminal": []
        }

        with pytest.warns(DeprecationWarning, match="month=6"):
            goals = goals_from_dict(data, start_date=date(2025, 1, 1))

        assert len(goals) == 1
        # month=6 from Jan 2025 should be July 2025 (month 7)
        assert goals[0].date.month == 7

    def test_goal_missing_date_raises_error(self):
        """Test error when intermediate goal has no date or month."""
        data = {
            "intermediate": [
                {"account": "Bad", "threshold": 1000000, "confidence": 0.8}
            ],
            "terminal": []
        }

        with pytest.raises(ValueError, match="must have 'date' field"):
            goals_from_dict(data)


# ============================================================================
# SCENARIO SERIALIZATION TESTS
# ============================================================================

class TestScenarioSerialization:
    """Test scenario serialization."""

    @pytest.fixture
    def sample_model(self):
        """Create sample model for scenario tests."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
            variable=VariableIncome(base=200_000, sigma=0.1)
        )
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return FinancialModel(income=income, accounts=accounts)

    @pytest.fixture
    def sample_goals(self):
        """Create sample goals for scenario tests."""
        return [
            IntermediateGoal("Conservative", 5_000_000, 0.9, date=date(2025, 6, 1)),
            TerminalGoal("Aggressive", 50_000_000, 0.8),
        ]

    def test_save_scenario_with_embedded_model(self, sample_model, sample_goals, tmp_path):
        """Test save_scenario with embedded model."""
        path = tmp_path / "scenario.json"

        save_scenario(
            scenario_name="Test Scenario",
            goals=sample_goals,
            path=path,
            model=sample_model,
            start_date=date(2025, 1, 1)
        )

        assert path.exists()
        content = json.loads(path.read_text())
        assert content["name"] == "Test Scenario"
        assert "model" in content
        assert "model_path" not in content

    def test_save_scenario_with_model_path(self, sample_goals, tmp_path):
        """Test save_scenario with model path reference."""
        path = tmp_path / "scenario.json"

        save_scenario(
            scenario_name="Reference Scenario",
            goals=sample_goals,
            path=path,
            model_path="profiles/my_model.json",
            start_date=date(2025, 1, 1)
        )

        content = json.loads(path.read_text())
        assert content["model_path"] == "profiles/my_model.json"
        assert "model" not in content

    def test_load_scenario_embedded_model(self, sample_model, sample_goals, tmp_path):
        """Test load_scenario with embedded model."""
        path = tmp_path / "scenario.json"
        save_scenario("Test", sample_goals, path, model=sample_model, start_date=date(2025, 1, 1))

        scenario = load_scenario(path)

        assert scenario["name"] == "Test"
        assert "model" in scenario
        assert isinstance(scenario["model"], FinancialModel)
        assert len(scenario["goals"]) == 2

    def test_load_scenario_external_model(self, sample_model, sample_goals, tmp_path):
        """Test load_scenario with external model file."""
        # Save model to separate file
        model_path = tmp_path / "model.json"
        save_model(sample_model, model_path)

        # Save scenario referencing the model
        scenario_path = tmp_path / "scenario.json"
        save_scenario(
            "External",
            sample_goals,
            scenario_path,
            model_path="model.json",  # Relative path
            start_date=date(2025, 1, 1)
        )

        scenario = load_scenario(scenario_path)

        assert "model" in scenario
        assert isinstance(scenario["model"], FinancialModel)

    def test_scenario_with_goals(self, sample_model, tmp_path):
        """Test scenario preserves goals correctly."""
        goals = [
            IntermediateGoal("Acc1", 1_000_000, 0.9, date=date(2025, 3, 1)),
            IntermediateGoal("Acc2", 2_000_000, 0.85, date=date(2025, 6, 1)),
            TerminalGoal("Acc2", 10_000_000, 0.8),
        ]
        path = tmp_path / "goals_scenario.json"

        save_scenario("Goals Test", goals, path, model=sample_model, start_date=date(2025, 1, 1))
        scenario = load_scenario(path)

        assert len(scenario["goals"]) == 3
        intermediate = [g for g in scenario["goals"] if isinstance(g, IntermediateGoal)]
        terminal = [g for g in scenario["goals"] if isinstance(g, TerminalGoal)]
        assert len(intermediate) == 2
        assert len(terminal) == 1

    def test_scenario_with_withdrawals(self, sample_model, sample_goals, tmp_path):
        """Test scenario with withdrawals."""
        withdrawals = WithdrawalModel(
            scheduled=WithdrawalSchedule([
                WithdrawalEvent("Conservative", 100_000, date(2025, 6, 1))
            ])
        )
        path = tmp_path / "withdrawal_scenario.json"

        save_scenario(
            "Withdrawal Test",
            sample_goals,
            path,
            model=sample_model,
            withdrawals=withdrawals,
            start_date=date(2025, 1, 1)
        )
        scenario = load_scenario(path)

        assert scenario["withdrawals"] is not None
        assert len(scenario["withdrawals"].scheduled.events) == 1

    def test_scenario_with_simulation_config(self, sample_model, sample_goals, tmp_path):
        """Test scenario preserves simulation config."""
        path = tmp_path / "sim_config.json"

        save_scenario(
            "Sim Config Test",
            sample_goals,
            path,
            model=sample_model,
            n_sims=1000,
            seed=42,
            start_date=date(2025, 1, 1)
        )
        scenario = load_scenario(path)

        assert scenario["simulation"].n_sims == 1000
        assert scenario["simulation"].seed == 42

    def test_scenario_with_optimization_config(self, sample_model, sample_goals, tmp_path):
        """Test scenario preserves optimization config."""
        path = tmp_path / "opt_config.json"

        save_scenario(
            "Opt Config Test",
            sample_goals,
            path,
            model=sample_model,
            T_max=120,
            solver="SCS",
            objective="conservative",
            start_date=date(2025, 1, 1)
        )
        scenario = load_scenario(path)

        assert scenario["optimization"].T_max == 120
        assert scenario["optimization"].solver == "SCS"
        assert scenario["optimization"].objective == "conservative"

    def test_save_scenario_error_both_model_and_path(self, sample_model, sample_goals, tmp_path):
        """Test error when both model and model_path provided."""
        path = tmp_path / "error.json"

        with pytest.raises(ValueError, match="either model or model_path"):
            save_scenario(
                "Error Test",
                sample_goals,
                path,
                model=sample_model,
                model_path="some/path.json"
            )

    def test_load_scenario_missing_model_file(self, sample_goals, tmp_path):
        """Test warning when model file not found."""
        path = tmp_path / "missing_model.json"
        save_scenario(
            "Missing Model",
            sample_goals,
            path,
            model_path="nonexistent/model.json",
            start_date=date(2025, 1, 1)
        )

        with pytest.warns(UserWarning, match="Model file not found"):
            scenario = load_scenario(path)

        assert "model" not in scenario or scenario.get("model") is None


# ============================================================================
# OPTIMIZATION RESULT SERIALIZATION TESTS
# ============================================================================

class TestOptimizationResultSerialization:
    """Test optimization result serialization."""

    @pytest.fixture
    def mock_optimization_result(self):
        """Create a mock optimization result."""
        from src.optimization import OptimizationResult
        from src.goals import GoalSet

        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

        goals = [
            TerminalGoal("Aggressive", 50_000_000, 0.8)
        ]

        goal_set = GoalSet(goals, accounts, date(2025, 1, 1))

        return OptimizationResult(
            T=24,
            X=np.tile([0.4, 0.6], (24, 1)),
            objective_value=1.5e7,
            feasible=True,
            goals=goals,
            goal_set=goal_set,
            solve_time=0.5,
            n_iterations=10,
        )

    def test_save_optimization_result(self, mock_optimization_result, tmp_path):
        """Test save_optimization_result creates file."""
        path = tmp_path / "result.json"

        save_optimization_result(mock_optimization_result, path)

        assert path.exists()
        content = json.loads(path.read_text())
        assert content["T"] == 24
        assert content["feasible"] is True

    def test_save_optimization_result_with_policy(self, mock_optimization_result, tmp_path):
        """Test save with include_policy=True."""
        path = tmp_path / "with_policy.json"

        save_optimization_result(mock_optimization_result, path, include_policy=True)

        content = json.loads(path.read_text())
        assert "X" in content
        assert len(content["X"]) == 24
        assert len(content["X"][0]) == 2

    def test_save_optimization_result_without_policy(self, mock_optimization_result, tmp_path):
        """Test save with include_policy=False."""
        path = tmp_path / "no_policy.json"

        save_optimization_result(mock_optimization_result, path, include_policy=False)

        content = json.loads(path.read_text())
        assert "X" not in content

    def test_load_optimization_result(self, mock_optimization_result, tmp_path):
        """Test load_optimization_result returns dict."""
        path = tmp_path / "load_test.json"
        save_optimization_result(mock_optimization_result, path)

        result = load_optimization_result(path)

        assert result["T"] == 24
        assert result["feasible"] is True
        assert isinstance(result["X"], np.ndarray)
        assert result["X"].shape == (24, 2)

    def test_optimization_result_roundtrip(self, mock_optimization_result, tmp_path):
        """Test optimization result roundtrip."""
        path = tmp_path / "roundtrip.json"

        save_optimization_result(mock_optimization_result, path)
        loaded = load_optimization_result(path)

        np.testing.assert_array_almost_equal(
            loaded["X"],
            mock_optimization_result.X
        )
        assert loaded["objective_value"] == pytest.approx(mock_optimization_result.objective_value)
        assert loaded["solve_time"] == mock_optimization_result.solve_time


# ============================================================================
# ADDITIONAL EDGE CASES
# ============================================================================

class TestAdditionalEdgeCases:
    """Additional edge cases for improved coverage."""

    def test_model_with_correlation_roundtrip(self, tmp_path):
        """Test model with correlation matrix roundtrip."""
        income = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=None
        )
        accounts = [
            Account.from_annual("Acc1", 0.08, 0.10),
            Account.from_annual("Acc2", 0.12, 0.15),
        ]
        model = FinancialModel(income=income, accounts=accounts)

        # Set custom correlation
        custom_corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        model.returns.default_correlation = custom_corr

        path = tmp_path / "corr_model.json"
        save_model(model, path, include_correlation=True)
        loaded = load_model(path)

        np.testing.assert_array_almost_equal(
            loaded.returns.default_correlation,
            custom_corr
        )

    def test_scenario_schema_version_mismatch(self, tmp_path):
        """Test scenario warns on schema version mismatch."""
        # Create scenario with different version
        config = {
            "schema_version": "0.0.1",
            "name": "Old Scenario",
            "start_date": "2025-01-01",
            "model": {
                "income": {
                    "fixed": {"base": 1_000_000, "annual_growth": 0.0},
                    "contribution_rate_fixed": 0.3,
                    "contribution_rate_variable": 1.0,
                },
                "accounts": [
                    {"name": "Test", "annual_return": 0.08, "annual_volatility": 0.10,
                     "initial_wealth": 0}
                ]
            },
            "intermediate_goals": [],
            "terminal_goals": [],
        }
        path = tmp_path / "old_scenario.json"
        path.write_text(json.dumps(config))

        with pytest.warns(UserWarning, match="schema version"):
            load_scenario(path)

    def test_income_roundtrip_with_salary_raises(self):
        """Test income roundtrip preserves salary raises."""
        raises = {date(2025, 6, 1): 0.10}
        original = IncomeModel(
            fixed=FixedIncome(base=1_000_000, salary_raises=raises),
            variable=None
        )

        serialized = income_to_dict(original)
        recovered = income_from_dict(serialized)

        assert recovered.fixed.salary_raises is not None
        assert date(2025, 6, 1) in recovered.fixed.salary_raises

    def test_income_roundtrip_with_seasonality(self):
        """Test income roundtrip preserves seasonality."""
        seasonality = np.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.0, 0.8, 0.9, 1.0, 1.05, 1.15])
        original = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, seasonality=seasonality)
        )

        serialized = income_to_dict(original)
        recovered = income_from_dict(serialized)

        np.testing.assert_array_almost_equal(
            recovered.variable.seasonality,
            seasonality
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
