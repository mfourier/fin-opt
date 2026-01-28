"""
Unit tests for plotting.py module.

Tests ModelPlottingMixin visualization methods including:
- plot() unified interface and mode dispatch
- _plot_allocation() 4-panel analysis
- _plot_comparison() strategy comparison
- Delegate methods to component plotters
"""

import pytest
import numpy as np
from datetime import date
from unittest.mock import Mock, patch, MagicMock

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.model import FinancialModel, SimulationResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def income():
    """Create test income model."""
    return IncomeModel(
        fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
        variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
    )


@pytest.fixture
def accounts():
    """Create test accounts."""
    return [
        Account.from_annual("Conservative", 0.04, 0.05),
        Account.from_annual("Aggressive", 0.14, 0.15),
    ]


@pytest.fixture
def model(income, accounts):
    """Create test FinancialModel with plotting mixin."""
    return FinancialModel(income=income, accounts=accounts)


@pytest.fixture
def allocation_policy():
    """Standard 60-40 allocation policy for T=12."""
    return np.tile([0.6, 0.4], (12, 1))


@pytest.fixture
def simulation_result(model, allocation_policy):
    """Pre-computed simulation result."""
    return model.simulate(
        T=12,
        n_sims=50,
        X=allocation_policy,
        seed=42,
        start=date(2025, 1, 1)
    )


# ============================================================================
# PLOT DISPATCH TESTS
# ============================================================================

class TestPlotDispatch:
    """Test plot() unified interface and mode dispatch."""

    def test_invalid_mode_raises_error(self, model):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            model.plot("invalid_mode")

    def test_invalid_mode_suggests_valid_options(self, model):
        """Test error message contains valid mode options."""
        with pytest.raises(ValueError) as exc_info:
            model.plot("foo")

        error_msg = str(exc_info.value)
        assert "income" in error_msg
        assert "wealth" in error_msg
        assert "allocation" in error_msg

    def test_dispatch_returns_dispatch_dict(self, model):
        """Test _dispatch_plot has all expected modes."""
        # Access internal dispatch dictionary
        dispatch = {
            "income": model._plot_income,
            "contributions": model._plot_contributions,
            "returns": model._plot_returns,
            "returns_cumulative": model._plot_returns_cumulative,
            "returns_horizon": model._plot_returns_horizon,
            "wealth": model._plot_wealth,
            "allocation": model._plot_allocation,
            "comparison": model._plot_comparison,
        }

        for mode in dispatch:
            assert hasattr(model, f"_plot_{mode.replace('_', '_')}")


# ============================================================================
# WEALTH MODE TESTS
# ============================================================================

class TestPlotWealth:
    """Test wealth mode plotting."""

    def test_wealth_requires_T_parameter(self, model, allocation_policy):
        """Test wealth mode requires T when no result provided."""
        with pytest.raises(ValueError, match="requires T and X"):
            model.plot("wealth", X=allocation_policy)

    def test_wealth_requires_X_parameter(self, model):
        """Test wealth mode requires X when no result provided."""
        with pytest.raises(ValueError, match="requires T and X"):
            model.plot("wealth", T=12)

    def test_wealth_with_result_bypasses_simulation(self, model, simulation_result):
        """Test wealth mode uses provided result without re-simulating."""
        fig, ax = model.plot(
            "wealth",
            result=simulation_result,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_wealth_auto_simulates(self, model, allocation_policy):
        """Test wealth mode auto-simulates when result not provided."""
        fig, ax = model.plot(
            "wealth",
            T=12,
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_wealth_extracts_start_from_result(self, model, simulation_result):
        """Test wealth mode extracts start date from SimulationResult."""
        # This should work without explicitly passing start
        fig, ax = model.plot(
            "wealth",
            result=simulation_result,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_wealth_invalid_result_type(self, model):
        """Test wealth mode rejects invalid result type."""
        with pytest.raises(TypeError, match="requires result parameter"):
            model._plot_wealth(result="not_a_result")


# ============================================================================
# ALLOCATION MODE TESTS
# ============================================================================

class TestPlotAllocation:
    """Test allocation mode 4-panel visualization."""

    def test_allocation_requires_X_parameter(self, model):
        """Test allocation mode requires X parameter."""
        with pytest.raises(ValueError, match="requires X"):
            model.plot("allocation")

    def test_allocation_returns_figure_and_axes(self, model, allocation_policy):
        """Test allocation mode returns figure and axes dict."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        assert isinstance(axes, dict)
        assert "fractions" in axes
        assert "absolute" in axes
        assert "cumulative" in axes
        assert "decomposition" in axes
        plt.close(fig)

    def test_allocation_with_result(self, model, allocation_policy, simulation_result):
        """Test allocation mode with pre-computed result."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            result=simulation_result,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_allocation_validates_X_shape(self, model):
        """Test allocation validates X has correct number of accounts."""
        X_wrong = np.tile([0.3, 0.3, 0.4], (12, 1))  # 3 accounts

        with pytest.raises(ValueError, match="accounts"):
            model.plot("allocation", X=X_wrong, n_sims=30, seed=42)

    def test_allocation_panel_fractions_stacked(self, model, allocation_policy):
        """Test panel 1 shows allocation fractions correctly."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        ax_fractions = axes["fractions"]
        # Should have bars (patches)
        assert len(ax_fractions.patches) > 0
        # Y-axis should be 0-1 for fractions
        assert ax_fractions.get_ylim()[1] <= 1.1
        plt.close(fig)

    def test_allocation_with_start_date(self, model, allocation_policy):
        """Test allocation with calendar start date."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            start=date(2025, 1, 1),
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_allocation_with_custom_colors(self, model, allocation_policy):
        """Test allocation with custom account colors."""
        colors = {"Conservative": "blue", "Aggressive": "red"}

        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            colors=colors,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_allocation_with_trajectories(self, model, allocation_policy):
        """Test allocation with trajectory display enabled."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            show_trajectories=True,
            trajectory_alpha=0.1,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_allocation_decomposition_panel(self, model, allocation_policy):
        """Test panel 4 decomposition has stacked bars."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        ax_decomp = axes["decomposition"]
        # Should have stacked bars (capital + gains)
        assert len(ax_decomp.patches) >= 2  # At least 2 accounts
        plt.close(fig)

    def test_allocation_custom_figsize(self, model, allocation_policy):
        """Test allocation with custom figure size."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            figsize=(20, 12),
            return_fig_ax=True
        )

        assert fig.get_figwidth() == 20
        assert fig.get_figheight() == 12
        plt.close(fig)


# ============================================================================
# COMPARISON MODE TESTS
# ============================================================================

class TestPlotComparison:
    """Test comparison mode for multi-strategy analysis."""

    def test_comparison_requires_results_dict(self, model):
        """Test comparison mode requires results parameter."""
        with pytest.raises(ValueError, match="requires results"):
            model.plot("comparison")

    def test_comparison_with_valid_results(self, model, allocation_policy):
        """Test comparison with valid results dictionary."""
        # Create two strategies
        X_conservative = np.tile([0.8, 0.2], (12, 1))
        X_aggressive = np.tile([0.2, 0.8], (12, 1))

        result1 = model.simulate(T=12, n_sims=30, X=X_conservative, seed=42)
        result2 = model.simulate(T=12, n_sims=30, X=X_aggressive, seed=43)

        fig, axes = model.plot(
            "comparison",
            results={"Conservative": result1, "Aggressive": result2},
            return_fig_ax=True
        )

        assert fig is not None
        assert len(axes) == 2  # Two panels
        plt.close(fig)

    def test_comparison_invalid_results_type(self, model):
        """Test comparison rejects non-dict results."""
        with pytest.raises(TypeError, match="requires results dict"):
            model._plot_comparison(results="not_a_dict")

    def test_comparison_metric_total_wealth(self, model, allocation_policy):
        """Test comparison with total_wealth metric."""
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        fig, axes = model.plot(
            "comparison",
            results={"Strategy": result},
            metric="total_wealth",
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_comparison_metric_wealth(self, model, allocation_policy):
        """Test comparison with wealth metric."""
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        fig, axes = model.plot(
            "comparison",
            results={"Strategy": result},
            metric="wealth",
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_comparison_invalid_metric(self, model, allocation_policy):
        """Test comparison rejects invalid metric."""
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        with pytest.raises(ValueError, match="Unsupported metric"):
            model.plot(
                "comparison",
                results={"Strategy": result},
                metric="invalid_metric"
            )

    def test_comparison_custom_title(self, model, allocation_policy):
        """Test comparison with custom title."""
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        fig, axes = model.plot(
            "comparison",
            results={"Strategy": result},
            title="My Custom Comparison",
            return_fig_ax=True
        )

        assert "My Custom Comparison" in fig._suptitle.get_text()
        plt.close(fig)


# ============================================================================
# DELEGATE METHOD TESTS
# ============================================================================

class TestPlotDelegates:
    """Test delegate methods that forward to component plotters."""

    def test_plot_income_delegates(self, model):
        """Test _plot_income delegates to income.plot()."""
        with patch.object(model.income, 'plot') as mock_plot:
            mock_plot.return_value = (Mock(), Mock())
            model._plot_income(months=12)
            mock_plot.assert_called_once()

    def test_plot_contributions_delegates(self, model):
        """Test _plot_contributions delegates to income.plot()."""
        with patch.object(model.income, 'plot') as mock_plot:
            mock_plot.return_value = (Mock(), Mock())
            model._plot_contributions(months=12)
            mock_plot.assert_called_once()

    def test_plot_returns_delegates(self, model):
        """Test _plot_returns delegates to returns.plot()."""
        with patch.object(model.returns, 'plot') as mock_plot:
            mock_plot.return_value = (Mock(), Mock())
            model._plot_returns(T=12, n_sims=50, seed=42)
            mock_plot.assert_called_once()

    def test_plot_returns_cumulative_delegates(self, model):
        """Test _plot_returns_cumulative delegates to returns.plot_cumulative()."""
        with patch.object(model.returns, 'plot_cumulative') as mock_plot:
            mock_plot.return_value = (Mock(), Mock())
            model._plot_returns_cumulative(T=12, n_sims=50, seed=42)
            mock_plot.assert_called_once()

    def test_plot_returns_horizon_delegates(self, model):
        """Test _plot_returns_horizon delegates to returns.plot_horizon_analysis()."""
        with patch.object(model.returns, 'plot_horizon_analysis') as mock_plot:
            mock_plot.return_value = (Mock(), Mock())
            model._plot_returns_horizon()
            mock_plot.assert_called_once()


# ============================================================================
# INCOME/CONTRIBUTIONS MODE TESTS
# ============================================================================

class TestPlotIncome:
    """Test income and contributions modes."""

    def test_income_mode_works(self, model):
        """Test income mode generates plot."""
        fig, ax = model.plot(
            "income",
            months=12,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_contributions_mode_works(self, model):
        """Test contributions mode generates plot."""
        fig, ax = model.plot(
            "contributions",
            months=12,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_income_with_start_date(self, model):
        """Test income with calendar start date."""
        fig, ax = model.plot(
            "income",
            months=12,
            start=date(2025, 1, 1),
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)


# ============================================================================
# RETURNS MODE TESTS
# ============================================================================

class TestPlotReturns:
    """Test returns visualization modes."""

    def test_returns_mode_works(self, model):
        """Test returns mode generates plot."""
        fig, ax = model.plot(
            "returns",
            T=12,
            n_sims=50,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_returns_cumulative_works(self, model):
        """Test returns_cumulative mode generates plot."""
        # plot_cumulative returns (fig, axes, ax_hists) for M>1
        result = model.plot(
            "returns_cumulative",
            T=24,
            n_sims=50,
            seed=42,
            return_fig_ax=True
        )

        assert result is not None
        assert result[0] is not None  # fig
        plt.close(result[0])

    def test_returns_horizon_works(self, model):
        """Test returns_horizon mode generates plot."""
        fig, axes = model.plot(
            "returns_horizon",
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)


# ============================================================================
# COMMON PARAMETERS TESTS
# ============================================================================

class TestPlotCommonParameters:
    """Test common plotting parameters across modes."""

    def test_return_fig_ax_false_returns_none(self, model, allocation_policy):
        """Test return_fig_ax=False returns None."""
        result = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=False
        )

        assert result is None
        plt.close('all')

    def test_custom_title(self, model, allocation_policy):
        """Test custom title is applied."""
        fig, axes = model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            title="Custom Title",
            return_fig_ax=True
        )

        # Title should be in suptitle
        suptitle = fig._suptitle
        assert suptitle is not None
        assert "Custom Title" in suptitle.get_text()
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["income", "contributions"])
    def test_T_parameter_maps_to_months(self, model, mode):
        """Test T parameter maps to months for income modes."""
        fig, ax = model.plot(
            mode,
            T=18,  # Using T instead of months
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)


# ============================================================================
# SAVE PATH TESTS
# ============================================================================

class TestPlotSavePath:
    """Test save_path parameter for saving figures."""

    def test_save_path_allocation(self, model, allocation_policy, tmp_path):
        """Test saving allocation plot to file."""
        save_file = tmp_path / "allocation.png"

        model.plot(
            "allocation",
            X=allocation_policy,
            n_sims=30,
            seed=42,
            save_path=str(save_file)
        )

        assert save_file.exists()
        plt.close('all')

    def test_save_path_comparison(self, model, allocation_policy, tmp_path):
        """Test saving comparison plot to file."""
        save_file = tmp_path / "comparison.png"
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        model.plot(
            "comparison",
            results={"Strategy": result},
            save_path=str(save_file)
        )

        assert save_file.exists()
        plt.close('all')


# ============================================================================
# EDGE CASES
# ============================================================================

class TestPlotEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_account_allocation(self, income):
        """Test allocation with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        model = FinancialModel(income=income, accounts=accounts)
        X = np.ones((12, 1))  # 100% to single account

        fig, axes = model.plot(
            "allocation",
            X=X,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_short_horizon(self, model):
        """Test with very short horizon T=3."""
        X = np.tile([0.6, 0.4], (3, 1))

        fig, axes = model.plot(
            "allocation",
            X=X,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_long_horizon(self, model):
        """Test with long horizon T=120."""
        X = np.tile([0.6, 0.4], (120, 1))

        fig, axes = model.plot(
            "allocation",
            X=X,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_comparison_single_strategy(self, model, allocation_policy):
        """Test comparison with single strategy."""
        result = model.simulate(T=12, n_sims=30, X=allocation_policy, seed=42)

        fig, axes = model.plot(
            "comparison",
            results={"Only Strategy": result},
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_comparison_many_strategies(self, model):
        """Test comparison with many strategies."""
        results = {}
        for i in range(5):
            X = np.tile([0.2 * i, 1 - 0.2 * i], (12, 1))
            X = np.clip(X, 0, 1)
            X = X / X.sum(axis=1, keepdims=True)  # Normalize
            results[f"Strategy_{i}"] = model.simulate(
                T=12, n_sims=20, X=X, seed=42 + i
            )

        fig, axes = model.plot(
            "comparison",
            results=results,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_deterministic_contributions(self, income, accounts):
        """Test allocation plot with deterministic (1D) contributions."""
        # Use income model with no variable component
        income_det = IncomeModel(
            fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
            variable=None
        )
        model = FinancialModel(income=income_det, accounts=accounts)
        X = np.tile([0.6, 0.4], (12, 1))

        fig, axes = model.plot(
            "allocation",
            X=X,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)


# ============================================================================
# KWARGS FORWARDING TESTS
# ============================================================================

class TestKwargsForwarding:
    """Test that kwargs are properly forwarded through dispatch."""

    def test_income_kwargs_forwarded(self, model):
        """Test income mode forwards kwargs."""
        # These kwargs should be forwarded to income.plot()
        fig, ax = model.plot(
            "income",
            months=12,
            n_simulations=30,
            show_trajectories=True,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)

    def test_returns_kwargs_forwarded(self, model):
        """Test returns mode forwards kwargs."""
        fig, ax = model.plot(
            "returns",
            T=12,
            n_sims=30,
            seed=42,
            show_trajectories=True,
            return_fig_ax=True
        )

        assert fig is not None
        plt.close(fig)


# ============================================================================
# CACHE CONTROL TESTS
# ============================================================================

class TestCacheControl:
    """Test cache control parameter behavior."""

    def test_use_cache_true_default(self, model, allocation_policy):
        """Test use_cache=True is default behavior."""
        # First call should simulate
        result1 = model.plot(
            "wealth",
            T=12,
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        # Second call with same params should use cache
        result2 = model.plot(
            "wealth",
            T=12,
            X=allocation_policy,
            n_sims=30,
            seed=42,
            return_fig_ax=True
        )

        assert result1[0] is not None
        assert result2[0] is not None
        plt.close('all')

    def test_use_cache_false_forces_resimulation(self, model, allocation_policy):
        """Test use_cache=False forces re-simulation."""
        result = model.plot(
            "wealth",
            T=12,
            X=allocation_policy,
            n_sims=30,
            seed=42,
            use_cache=False,
            return_fig_ax=True
        )

        assert result[0] is not None
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
