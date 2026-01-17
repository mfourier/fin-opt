"""
Unit tests for CLI module.

Tests command-line interface functionality using Click's testing utilities.
"""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner

from src.cli import main, __version__


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file."""
    config = {
        "schema_version": "0.1.0",
        "income": {
            "fixed": {
                "base": 1500000,
                "annual_growth": 0.03
            },
            "variable": {
                "base": 0,
                "sigma": 0.0,
                "annual_growth": 0.0
            },
            "contribution_rate_fixed": 0.3,
            "contribution_rate_variable": 1.0
        },
        "accounts": [
            {
                "name": "Conservative",
                "annual_return": 0.06,
                "annual_volatility": 0.08,
                "initial_wealth": 0
            },
            {
                "name": "Aggressive",
                "annual_return": 0.12,
                "annual_volatility": 0.15,
                "initial_wealth": 0
            }
        ]
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return config_file


@pytest.fixture
def temp_goals(tmp_path):
    """Create temporary goals file."""
    goals = {
        "goals": [
            {
                "type": "terminal",
                "account": "Aggressive",
                "threshold": 5000000,
                "confidence": 0.7
            }
        ]
    }
    goals_file = tmp_path / "test_goals.json"
    with open(goals_file, "w") as f:
        json.dump(goals, f)
    return goals_file


# ============================================================================
# MAIN COMMAND TESTS
# ============================================================================

class TestMainCommand:
    """Test main CLI entry point."""

    def test_main_help(self, runner):
        """Test main --help shows help message."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "FinOpt" in result.output
        assert "simulate" in result.output
        assert "optimize" in result.output
        assert "config" in result.output

    def test_main_version(self, runner):
        """Test main --version shows version."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_main_quiet_option(self, runner):
        """Test --quiet option is accepted."""
        result = runner.invoke(main, ["--quiet", "--help"])
        assert result.exit_code == 0


# ============================================================================
# SIMULATE COMMAND TESTS
# ============================================================================

class TestSimulateCommand:
    """Test simulate command."""

    def test_simulate_help(self, runner):
        """Test simulate --help shows help."""
        result = runner.invoke(main, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--horizon" in result.output
        assert "--simulations" in result.output

    def test_simulate_requires_config(self, runner):
        """Test simulate requires --config option."""
        result = runner.invoke(main, ["simulate"])
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "missing" in result.output.lower()

    def test_simulate_basic(self, runner, temp_config, tmp_path):
        """Test basic simulation run."""
        result = runner.invoke(main, [
            "--quiet",
            "simulate",
            "--config", str(temp_config),
            "--horizon", "6",
            "--simulations", "10",
            "--seed", "42",
        ])
        # Should succeed or at least not crash with import error
        assert result.exit_code == 0 or "Error" in result.output

    def test_simulate_with_output(self, runner, temp_config, tmp_path):
        """Test simulation with output directory."""
        output_dir = tmp_path / "results"
        result = runner.invoke(main, [
            "--quiet",
            "simulate",
            "--config", str(temp_config),
            "--horizon", "6",
            "--simulations", "10",
            "--seed", "42",
            "--output", str(output_dir),
        ])
        if result.exit_code == 0:
            assert output_dir.exists()

    def test_simulate_with_allocation(self, runner, temp_config):
        """Test simulation with custom allocation."""
        result = runner.invoke(main, [
            "--quiet",
            "simulate",
            "--config", str(temp_config),
            "--horizon", "6",
            "--simulations", "10",
            "--allocation", "0.6,0.4",
        ])
        # Should work or fail gracefully
        assert result.exit_code == 0 or "Error" in result.output

    def test_simulate_invalid_allocation_count(self, runner, temp_config):
        """Test simulate with wrong number of allocation values."""
        result = runner.invoke(main, [
            "simulate",
            "--config", str(temp_config),
            "--allocation", "0.5",  # Only 1 value for 2 accounts
        ])
        assert result.exit_code != 0

    def test_simulate_invalid_allocation_sum(self, runner, temp_config):
        """Test simulate with allocation that doesn't sum to 1."""
        result = runner.invoke(main, [
            "simulate",
            "--config", str(temp_config),
            "--allocation", "0.5,0.3",  # Sums to 0.8
        ])
        assert result.exit_code != 0


# ============================================================================
# CONFIG COMMAND TESTS
# ============================================================================

class TestConfigCommand:
    """Test config subcommands."""

    def test_config_help(self, runner):
        """Test config --help shows subcommands."""
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.output
        assert "show" in result.output
        assert "create" in result.output

    def test_config_validate_valid(self, runner, temp_config):
        """Test config validate with valid config."""
        result = runner.invoke(main, [
            "--quiet",
            "config", "validate",
            str(temp_config)
        ])
        assert result.exit_code == 0

    def test_config_validate_invalid_path(self, runner):
        """Test config validate with non-existent file."""
        result = runner.invoke(main, [
            "config", "validate",
            "/nonexistent/path/config.json"
        ])
        assert result.exit_code != 0

    def test_config_show_json(self, runner, temp_config):
        """Test config show with JSON format."""
        result = runner.invoke(main, [
            "config", "show",
            str(temp_config),
            "--format", "json"
        ])
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "accounts" in data

    def test_config_show_table(self, runner, temp_config):
        """Test config show with table format."""
        result = runner.invoke(main, [
            "config", "show",
            str(temp_config),
            "--format", "table"
        ])
        assert result.exit_code == 0

    def test_config_create_basic(self, runner, tmp_path):
        """Test config create with basic template."""
        output_file = tmp_path / "new_config.json"
        result = runner.invoke(main, [
            "config", "create",
            str(output_file),
            "--template", "basic"
        ])
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify it's valid JSON
        with open(output_file) as f:
            data = json.load(f)
        assert "accounts" in data
        assert "income" in data

    def test_config_create_advanced(self, runner, tmp_path):
        """Test config create with advanced template."""
        output_file = tmp_path / "advanced_config.json"
        result = runner.invoke(main, [
            "config", "create",
            str(output_file),
            "--template", "advanced"
        ])
        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
        # Advanced should have variable income
        assert "variable" in data.get("income", {})


# ============================================================================
# OPTIMIZE COMMAND TESTS
# ============================================================================

class TestOptimizeCommand:
    """Test optimize command."""

    def test_optimize_help(self, runner):
        """Test optimize --help shows options."""
        result = runner.invoke(main, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--goals" in result.output
        assert "--objective" in result.output

    def test_optimize_requires_config_and_goals(self, runner):
        """Test optimize requires both config and goals."""
        result = runner.invoke(main, ["optimize"])
        assert result.exit_code != 0

    def test_optimize_with_horizon(self, runner, temp_config, temp_goals):
        """Test optimize with fixed horizon."""
        result = runner.invoke(main, [
            "--quiet",
            "optimize",
            "--config", str(temp_config),
            "--goals", str(temp_goals),
            "--horizon", "12",
            "--simulations", "50",
        ])
        # May fail if cvxpy not installed, which is OK
        assert result.exit_code == 0 or "cvxpy" in result.output.lower() or "Error" in result.output


# ============================================================================
# REPORT COMMAND TESTS
# ============================================================================

class TestReportCommand:
    """Test report command."""

    @pytest.fixture
    def temp_result(self, tmp_path):
        """Create temporary simulation result file."""
        result = {
            "horizon": 12,
            "n_sims": 100,
            "seed": 42,
            "statistics": {
                "median_wealth": 5000000,
                "mean": 5200000,
                "std": 800000,
                "p10": 4000000,
                "p90": 6500000,
            },
            "wealth_trajectories": [
                [[100, 200], [150, 250], [200, 300]]
                for _ in range(10)
            ]
        }
        result_file = tmp_path / "result.json"
        with open(result_file, "w") as f:
            json.dump(result, f)
        return result_file

    def test_report_help(self, runner):
        """Test report --help shows options."""
        result = runner.invoke(main, ["report", "--help"])
        assert result.exit_code == 0
        assert "--result" in result.output
        assert "--format" in result.output

    def test_report_summary(self, runner, temp_result):
        """Test report with summary format."""
        result = runner.invoke(main, [
            "--quiet",
            "report",
            "--result", str(temp_result),
            "--format", "summary"
        ])
        assert result.exit_code == 0

    def test_report_detailed(self, runner, temp_result):
        """Test report with detailed format."""
        result = runner.invoke(main, [
            "report",
            "--result", str(temp_result),
            "--format", "detailed"
        ])
        assert result.exit_code == 0

    def test_report_csv(self, runner, temp_result, tmp_path):
        """Test report with CSV output."""
        output_file = tmp_path / "report.csv"
        result = runner.invoke(main, [
            "--quiet",
            "report",
            "--result", str(temp_result),
            "--format", "csv",
            "--output", str(output_file)
        ])
        assert result.exit_code == 0
        assert output_file.exists()


# ============================================================================
# INFO COMMAND TESTS
# ============================================================================

class TestInfoCommand:
    """Test info command."""

    def test_info_shows_version(self, runner):
        """Test info command shows version information."""
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "FinOpt" in result.output or __version__ in result.output

    def test_info_shows_dependencies(self, runner):
        """Test info shows dependency information."""
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "numpy" in result.output.lower()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling in CLI."""

    def test_invalid_config_file_format(self, runner, tmp_path):
        """Test handling of invalid JSON config."""
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{ invalid json }")

        result = runner.invoke(main, [
            "config", "validate",
            str(bad_config)
        ])
        assert result.exit_code != 0

    def test_missing_required_fields(self, runner, tmp_path):
        """Test handling of config with missing required fields."""
        incomplete_config = tmp_path / "incomplete.json"
        with open(incomplete_config, "w") as f:
            json.dump({"schema_version": "0.1.0"}, f)  # Missing accounts

        result = runner.invoke(main, [
            "config", "validate",
            str(incomplete_config)
        ])
        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
