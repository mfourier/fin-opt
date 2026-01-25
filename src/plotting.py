"""
Plotting utilities for FinOpt financial models.

Purpose
-------
This module provides the ModelPlottingMixin class which adds visualization
capabilities to FinancialModel through multiple inheritance. By separating
plotting logic from business logic, we maintain cleaner, more modular code.

Design Pattern: Mixin
--------------------
ModelPlottingMixin is designed to be inherited by FinancialModel, providing:
- Unified plot() interface with mode dispatch
- Individual _plot_* methods for each visualization type
- Delegation to component-level plotting (income, returns, portfolio)

The mixin expects these attributes/methods from the host class:
- self.income: IncomeModel
- self.returns: ReturnModel
- self.portfolio: Portfolio
- self.accounts: List[Account]
- self.M: int (number of accounts)
- self.simulate(): simulation method
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from .model import SimulationResult

__all__ = ["ModelPlottingMixin"]


class ModelPlottingMixin:
    """
    Mixin providing unified plotting interface for FinancialModel.
    
    This mixin adds the plot() method and all private _plot_* methods
    to FinancialModel via multiple inheritance. The API remains unchanged:
    
        model.plot("wealth", T=24, X=X, n_sims=500)
    
    Available Modes
    ---------------
    Pre-simulation (no simulation needed):
    - "income": Income streams (fixed, variable, total)
    - "contributions": Monthly contribution schedule
    - "returns": Return distributions and trajectories
    - "returns_cumulative": Cumulative return evolution
    - "returns_horizon": Risk-return by investment horizon
    
    Simulation-based (auto-simulates if result not provided):
    - "wealth": Portfolio dynamics (4 panels)
    - "allocation": Allocation analysis with investment gains (4 panels)
    - "comparison": Compare multiple strategies
    
    Note
    ----
    This class should only be used as a mixin with FinancialModel.
    Do not instantiate directly.
    """
    
    # -----------------------------------------------------------------------
    # Unified plotting interface
    # -----------------------------------------------------------------------

    def plot(
        self,
        mode: str,
        *,
        # Simulation parameters (for modes that need to simulate)
        T: Optional[int] = None,
        X: Optional[np.ndarray] = None,
        n_sims: int = 500,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        
        # Bypass: pre-computed result (optional)
        result: Optional[SimulationResult] = None,
        
        # Common plotting parameters
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        
        # Cache control
        use_cache: bool = True,
        
        # Mode-specific kwargs
        **kwargs
    ):
        """
        Unified plotting interface with automatic simulation.
        
        Dispatches to specialized plotting methods based on mode. For modes
        requiring simulation ("wealth", "comparison"), automatically executes
        simulate() internally unless a pre-computed result is provided.
        Simulations are cached by default to avoid redundant computation.
        
        Parameters
        ----------
        mode : str
            Visualization type. Available modes:
            
            **Pre-simulation (no simulation needed):**
            - "income": Income streams (fixed, variable, total)
            - "contributions": Monthly contribution schedule
            - "returns": Return distributions and trajectories
            - "returns_cumulative": Cumulative return evolution
            - "returns_horizon": Risk-return by investment horizon
            
            **Simulation-based (auto-simulates if result not provided):**
            - "wealth": Portfolio dynamics (4 panels)
            - "allocation": Allocation analysis with investment gains (4 panels)
            - "comparison": Compare multiple strategies
        
        T : int, optional
            Time horizon for simulation-based modes.
            Required if mode in {"wealth"} and result not provided.
        X : np.ndarray, shape (T, M), optional
            Allocation policy for simulation-based modes.
            Required if mode in {"wealth"} and result not provided.
        n_sims : int, default 500
            Number of Monte Carlo simulations.
        start : date, optional
            Calendar start date. Used for:
            - Simulation: aligns income seasonality and calendar index
            - Plotting: enables calendar-based x-axis (YYYY-MM format)
            If None, numeric month indices (0, 1, 2, ...) are used.
        seed : int, optional
            Random seed for reproducibility.
        result : SimulationResult, optional
            Pre-computed simulation result. If provided, bypasses simulate()
            and uses this result directly. Useful when result already exists
            or for multiple plots from same simulation.
        figsize : tuple, optional
            Figure size (width, height). Defaults vary by mode.
        title : str, optional
            Main figure title. Auto-generated if None.
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        return_fig_ax : bool, default False
            If True, returns (fig, ax) or (fig, axes) for customization.
        use_cache : bool, default True
            If True, checks cache before simulating. If False, forces
            re-simulation even if cached result exists.
        **kwargs
            Mode-specific parameters forwarded to underlying methods.
            
            **Income/Contributions modes:**
            - months : int, projection horizon
            - show_trajectories : bool
            - n_simulations : int
            - dual_axis : "auto"|True|False (income only)
            
            **Returns modes:**
            - correlation : np.ndarray
            - show_trajectories : bool
            - trajectory_alpha : float
            - show_percentiles : bool (cumulative only)
            - percentiles : tuple (cumulative only)
            - horizons : np.ndarray (horizon only)
            
            **Wealth mode:**
            - show_trajectories : bool
            - trajectory_alpha : float
            - colors : dict
            - hist_bins : int
            - hist_color : str
            
            **Comparison mode:**
            - results : dict[str, SimulationResult], required
            - metric : str
        
        Returns
        -------
        None or (fig, ax)
            If return_fig_ax=True, returns figure and axes.
            Otherwise displays plot and returns None.
        
        Raises
        ------
        ValueError
            If mode is invalid or required parameters are missing.
        
        Examples
        --------
        >>> from datetime import date
        >>> model = FinancialModel(income, accounts)
        >>> X = np.tile([0.6, 0.4], (24, 1))
        >>> 
        >>> # Pre-simulation: income streams (calendar axis)
        >>> model.plot("income", months=24, start=date(2025,1,1))
        >>> 
        >>> # Pre-simulation: return analysis (calendar axis)
        >>> model.plot("returns", T=24, n_sims=300, seed=42, start=date(2025,1,1))
        >>> model.plot("returns_cumulative", T=120, n_sims=500, start=date(2025,1,1))
        >>> 
        >>> # Simulation-based: auto-simulates + caches (calendar axis propagated)
        >>> model.plot("wealth", T=24, X=X, n_sims=500, seed=42, start=date(2025,1,1))
        >>> # Second call: instant (cached)
        >>> model.plot("wealth", T=24, X=X, n_sims=500, seed=42, title="Alt View")
        >>> 
        >>> # Bypass simulation if result exists (start extracted from result)
        >>> result = model.simulate(T=24, X=X, n_sims=500, start=date(2025,1,1))
        >>> model.plot("wealth", result=result, show_trajectories=False)
        >>> 
        >>> # Strategy comparison
        >>> result1 = model.simulate(T=24, X=X_conservative, n_sims=500)
        >>> result2 = model.simulate(T=24, X=X_aggressive, n_sims=500)
        >>> model.plot("comparison", results={"60-40": result1, "80-20": result2})
        """
        # Import here to avoid circular dependency at module level
        from .model import SimulationResult
        
        # Determine if mode needs simulation
        simulation_modes = {"wealth", "comparison"}
        needs_simulation = mode in simulation_modes

        if needs_simulation:
            # Option 1: result already provided (bypass)
            if result is not None:
                sim_result = result
            
            # Option 2: simulate with given parameters
            else:
                if mode == "wealth":
                    # Validate required parameters
                    if T is None or X is None:
                        raise ValueError(
                            f"mode='wealth' requires T and X parameters "
                            f"(or provide result=...)"
                        )
                    sim_result = self.simulate(
                        T=T, X=X, n_sims=n_sims, 
                        start=start, seed=seed, use_cache=use_cache
                    )
                
                elif mode == "comparison":
                    # This mode requires dict of results (special case)
                    if "results" not in kwargs:
                        raise ValueError(
                            f"mode='comparison' requires results parameter "
                            f"(dict of SimulationResult objects)"
                        )
                    sim_result = None  # no single result
        
        # Build dispatch kwargs - start with user-provided kwargs
        dispatch_kwargs = {
            "figsize": figsize,
            "title": title,
            "save_path": save_path,
            "return_fig_ax": return_fig_ax,
            **kwargs  # User kwargs first (will not be overridden)
        }
        
        # Add mode-specific parameters (only if not already in kwargs)
        if mode in ["returns", "returns_cumulative"]:
            # These modes expect T, n_sims, seed, start, correlation (optional)
            if T is not None:
                dispatch_kwargs.setdefault("T", T)
            dispatch_kwargs.setdefault("n_sims", n_sims)
            if seed is not None:
                dispatch_kwargs.setdefault("seed", seed)
            if start is not None:
                dispatch_kwargs.setdefault("start", start)
            # correlation comes from kwargs if provided
        
        elif mode == "returns_horizon":
            # This mode uses horizons array from kwargs, no T/n_sims/seed/start
            pass
        
        elif mode == "allocation":
            # Allocation analysis requires X and optionally result
            if X is None:
                raise ValueError("mode='allocation' requires X parameter")
            
            dispatch_kwargs["X"] = X
            if result is not None:
                dispatch_kwargs["result"] = result
            else:
                # Will auto-simulate, need these params
                dispatch_kwargs.setdefault("n_sims", n_sims)
                if seed is not None:
                    dispatch_kwargs["seed"] = seed
                if start is not None:
                    dispatch_kwargs["start"] = start
            
        elif mode in ["income", "contributions"]:
            dispatch_kwargs.setdefault("months", T)
            if start is not None:
                dispatch_kwargs.setdefault("start", start)
        
        elif mode == "wealth":
            # Inject simulation result and allocation policy
            dispatch_kwargs["result"] = sim_result
            if X is not None:
                dispatch_kwargs["X"] = X
            # start will be extracted from sim_result in _plot_wealth
        
        elif mode == "comparison":
            # results already in kwargs, no additional parameters needed
            pass
        
        # Remove None values (let underlying methods use their defaults)
        dispatch_kwargs = {k: v for k, v in dispatch_kwargs.items() 
                        if v is not None}
        
        # Dispatch
        return self._dispatch_plot(mode, **dispatch_kwargs)

    def _dispatch_plot(self, mode: str, **kwargs):
        """Internal dispatcher to plotting methods."""
        _dispatch = {
            # Pre-simulation
            "income": self._plot_income,
            "contributions": self._plot_contributions,
            "returns": self._plot_returns,
            "returns_cumulative": self._plot_returns_cumulative,
            "returns_horizon": self._plot_returns_horizon,
            
            # Simulation-based
            "wealth": self._plot_wealth,
            "allocation": self._plot_allocation,
            "comparison": self._plot_comparison,
        }
        
        if mode not in _dispatch:
            valid = ", ".join(f"'{m}'" for m in _dispatch.keys())
            raise ValueError(
                f"Invalid mode '{mode}'. Valid modes: {valid}"
            )
        
        return _dispatch[mode](**kwargs)

    # -----------------------------------------------------------------------
    # Plot delegates (forward to component methods)
    # -----------------------------------------------------------------------

    def _plot_income(self, **kwargs):
        """Delegate to income.plot_income()."""
        return self.income.plot(mode="income", **kwargs)

    def _plot_contributions(self, **kwargs):
        """Delegate to income.plot_contributions()."""
        return self.income.plot(mode="contributions", **kwargs)

    def _plot_returns(self, **kwargs):
        """Delegate to returns.plot()."""
        return self.returns.plot(**kwargs)

    def _plot_returns_cumulative(self, **kwargs):
        """Delegate to returns.plot_cumulative()."""
        return self.returns.plot_cumulative(**kwargs)

    def _plot_returns_horizon(self, **kwargs):
        """Delegate to returns.plot_horizon_analysis()."""
        return self.returns.plot_horizon_analysis(**kwargs)

    def _plot_wealth(
        self, 
        result: SimulationResult, 
        X: Optional[np.ndarray] = None, 
        **kwargs
    ):
        """
        Delegate to portfolio.plot() with automatic start extraction.
        
        Extracts start date from SimulationResult if not explicitly provided
        in kwargs, enabling seamless calendar-based visualization.
        """
        from .model import SimulationResult
        
        if not isinstance(result, SimulationResult):
            raise TypeError("mode='wealth' requires result parameter")
        
        X_plot = X if X is not None else result.allocation
        portfolio_result = {
            "wealth": result.wealth,
            "total_wealth": result.total_wealth
        }
        
        # Extract start from SimulationResult if not explicitly provided
        # User-provided start in kwargs takes precedence
        if "start" not in kwargs:
            kwargs["start"] = result.start
        
        return self.portfolio.plot(portfolio_result, X_plot, **kwargs)

    def _plot_allocation(
        self,
        X: np.ndarray,
        result: Optional[SimulationResult] = None,
        n_sims: int = 500,
        seed: Optional[int] = None,
        start: Optional[date] = None,
        figsize: tuple = (15, 10),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        colors: Optional[dict] = None,
        show_trajectories: bool = False,
        trajectory_alpha: float = 0.05,
    ) -> Optional[tuple]:
        """
        Analyze allocation policy with investment return decomposition.
        
        Creates 4-panel visualization showing policy execution and investment gains:
        1. Allocation fractions (stacked bar): x_t^m policy over time
        2. Monthly allocations (stacked area): A_t * x_t^m absolute contributions
        3. Cumulative capital vs wealth (lines): investment growth by account
        4. Final decomposition (stacked bar): capital + gains = wealth
        
        Parameters
        ----------
        X : np.ndarray, shape (T, M)
            Allocation policy matrix. Time horizon T is inferred from X.shape[0].
        result : SimulationResult, optional
            Pre-computed simulation result. If None, executes simulate() internally
            using n_sims and seed parameters.
        n_sims : int, default 500
            Number of MC simulations (used only if result is None).
        seed : int, optional
            Random seed for reproducibility (used only if result is None).
        start : date, optional
            Start date for calendar alignment. If None and result is None, uses 
            numeric months. If result is provided, extracts start from result.start.
        figsize : tuple, default (15, 10)
            Figure size (width, height).
        title : str, optional
            Main figure title.
        save_path : str, optional
            Path to save figure.
        return_fig_ax : bool, default False
            If True, returns (fig, axes_dict).
        colors : dict, optional
            Custom colors for accounts. Keys are account names.
        show_trajectories : bool, default False
            Whether to show individual simulation paths in Panel 3.
        trajectory_alpha : float, default 0.05
            Transparency for trajectory lines.
        
        Returns
        -------
        None or (fig, axes_dict)
            If return_fig_ax=True, returns figure and axes dictionary.
        
        Notes
        -----
        **Panel 3 shows investment effect**: The gap between cumulative capital
        (W_0 + contributions) and wealth represents compound interest gains.
        Larger gaps indicate better investment performance.
        
        **Panel 4 stacked decomposition**: For each account m:
        - Blue bar (base): Total capital invested W_0^m + Σ(A_t * x_t^m)
        - Orange bar (stacked): Net gain W_T^m - W_0^m - Σ(A_t * x_t^m)
        - Total height = Final wealth W_T^m
        
        Stacked bar design makes the relationship Capital + Gain = Wealth
        visually obvious and eliminates comparison ambiguity.
        
        Examples
        --------
        >>> # With pre-computed simulation
        >>> X = np.tile([0.6, 0.4], (24, 1))
        >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
        >>> model.plot("allocation", X=X, result=result)
        >>> 
        >>> # Auto-simulate
        >>> model.plot("allocation", X=X, n_sims=500, seed=42, 
        ...           start=date(2025,1,1))
        """
        from matplotlib.ticker import FuncFormatter
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from .utils import millions_formatter, month_index
        from .model import SimulationResult
        
        # Infer horizon from X
        T, M = X.shape
        if M != self.M:
            raise ValueError(f"X has {M} accounts but model has {self.M}")
        
        # Execute simulation if not provided
        if result is None:
            result = self.simulate(T=T, X=X, n_sims=n_sims, seed=seed, start=start)
        
        # Extract data from SimulationResult
        W = result.wealth  # (n_sims, T+1, M)
        A = result.contributions  # (n_sims, T) or (T,)
        n_sims_actual = W.shape[0]
        T_plus_1 = W.shape[1]
        
        # Validate dimensions
        if W.shape != (n_sims_actual, T + 1, M):
            raise ValueError(f"Wealth shape {W.shape} inconsistent with X shape {X.shape}")
        
        # Handle stochastic vs deterministic contributions
        if A.ndim == 1:  # Deterministic: (T,)
            A_mean = A
        else:  # Stochastic: (n_sims, T)
            if A.shape[0] != n_sims_actual or A.shape[1] != T:
                raise ValueError(f"Contributions shape {A.shape} inconsistent with expected ({n_sims_actual}, {T})")
            A_mean = A.mean(axis=0)
        
        # Compute absolute monthly allocations: A_t * x_t^m
        A_abs = A_mean[:, None] * X  # (T, M)
        
        # Mean wealth by account: (n_sims, T+1, M) -> (T+1, M)
        W_mean = W.mean(axis=0)
        
        # Extract initial wealth per account
        W0 = W_mean[0, :]  # (M,) - initial wealth at t=0
        
        # Compute cumulative contributions by account
        cum_contrib = np.cumsum(A_abs, axis=0)  # (T, M)
        # Total capital invested = W_0 + cumulative contributions
        cum_capital = cum_contrib + W0[None, :]  # (T, M)
        cum_capital_with_init = np.vstack([W0, cum_capital])  # (T+1, M) to match W
        
        # Final state decomposition
        total_contrib_final = cum_contrib[-1, :]  # (M,) - sum of monthly contributions only
        total_capital_invested = W0 + total_contrib_final  # (M,) - W_0 + contributions
        total_wealth_final = W_mean[-1, :]  # (M,) - final wealth
        # Net gains = W_T - W_0 - Σ(A_t * x_t^m)
        gains_final = total_wealth_final - total_capital_invested  # (M,)
        
        # Setup colors
        if colors is None:
            colors = {}
        default_colors = plt.cm.Dark2(np.linspace(0, 1, M))
        account_colors = []
        for i, acc in enumerate(self.accounts):
            if acc.name in colors:
                account_colors.append(colors[acc.name])
            elif acc.label in colors:
                account_colors.append(colors[acc.label])
            elif i in colors:
                account_colors.append(colors[i])
            else:
                account_colors.append(default_colors[i])
        
        # Time axis (T+1 points for wealth, T points for contributions/policy)
        start_date = result.start if hasattr(result, 'start') and result.start else start
        if start_date is not None:
            time_axis = month_index(start_date, T_plus_1)  # T+1 points
            time_axis_policy = time_axis[:-1]  # T points for X
            xlabel = "Date"
        else:
            time_axis = np.arange(T + 1)
            time_axis_policy = np.arange(T)
            xlabel = "Month"
        
        # Create figure with 4 panels
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        ax_fractions = fig.add_subplot(gs[0, 0])
        ax_absolute = fig.add_subplot(gs[0, 1])
        ax_cumulative = fig.add_subplot(gs[1, 0])
        ax_decomposition = fig.add_subplot(gs[1, 1])
        
        # ========== Panel 1: Allocation Fractions (stacked bar) ==========
        bar_width = np.diff(time_axis_policy).min() if start_date is not None else 0.9
        bottom = np.zeros(T)
        for m in range(M):
            ax_fractions.bar(
                time_axis_policy,
                X[:, m],
                bottom=bottom,
                width=bar_width,
                color=account_colors[m],
                label=self.accounts[m].label,
                edgecolor='white',
                linewidth=0.3,
                alpha=0.85
            )
            bottom += X[:, m]
        
        ax_fractions.set_xlabel(xlabel, fontsize=10)
        ax_fractions.set_ylabel("Allocation Fraction", fontsize=10)
        ax_fractions.set_title(r"Allocation Policy $(X = (x_t^m))$", fontsize=11, fontweight='bold')
        ax_fractions.set_ylim(0, 1)
        ax_fractions.grid(True, alpha=0.3)
        ax_fractions.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax_fractions.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        if start_date is not None:
            margin = (time_axis_policy[-1] - time_axis_policy[0]) * 0.02
            ax_fractions.set_xlim(time_axis_policy[0] - margin, time_axis_policy[-1] + margin)
        else:
            ax_fractions.set_xlim(-0.5, T - 0.5)
        
        # ========== Panel 2: Monthly Allocations (stacked area) ==========
        ax_absolute.stackplot(
            time_axis_policy,
            *[A_abs[:, m] for m in range(M)],
            labels=[acc.label for acc in self.accounts],
            colors=account_colors,
            alpha=0.8
        )
        ax_absolute.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_absolute.set_xlabel(xlabel, fontsize=10)
        ax_absolute.set_ylabel("Monthly Allocation (CLP)", fontsize=10)
        ax_absolute.set_title(r"Monthly Allocations by Account $(A_t \cdot x_t^m)$", fontsize=11, fontweight='bold')
        ax_absolute.grid(True, alpha=0.3)
        ax_absolute.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        # ========== Panel 3: Cumulative Capital vs Wealth  ==========
        # Plot trajectories if requested
        if show_trajectories:
            for i in range(n_sims_actual):
                for m in range(M):
                    ax_cumulative.plot(
                        time_axis,
                        W[i, :, m],
                        color=account_colors[m],
                        alpha=trajectory_alpha,
                        linewidth=0.6,
                        label='_nolegend_'
                    )

        # Plot wealth (solid) and capital (dashed) with same color per account
        for m in range(M):
            # Wealth (solid line)
            ax_cumulative.plot(
                time_axis,
                W_mean[:, m],
                color=account_colors[m],
                linewidth=2.5,
                linestyle='-',
                label='_nolegend_'
            )
            # Capital invested (dashed, same color with transparency)
            ax_cumulative.plot(
                time_axis,
                cum_capital_with_init[:, m],
                color=account_colors[m],
                linewidth=2,
                linestyle='--',
                alpha=0.5,
                label='_nolegend_'
            )

        # Clear any existing legend before creating new one
        if ax_cumulative.get_legend():
            ax_cumulative.get_legend().remove()

        # Create clean legend with generic entries
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, linestyle='-', label='Wealth'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', alpha=0.5, label='Capital Invested')
        ]
        ax_cumulative.legend(
            handles=legend_elements, 
            loc='upper left', 
            fontsize=9, 
            framealpha=1.0,
            edgecolor='black',
            fancybox=False
        )

        ax_cumulative.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_cumulative.set_xlabel(xlabel, fontsize=10)
        ax_cumulative.set_ylabel("Amount (CLP)", fontsize=10)
        ax_cumulative.set_title(r"Capital Invested vs Wealth $(W_0^m + \sum A_s x_s^m \text{ vs } W_t^m)$", 
                            fontsize=11, fontweight='bold')
        ax_cumulative.grid(True, alpha=0.3)
        
        # ========== Panel 4: Stacked Bar Decomposition ==========
        x_pos = np.arange(M)
        bar_width_decomp = 0.6  # Wider for clarity
        
        # Stacked bars: Capital (base) + Gain (top)
        bars_capital = ax_decomposition.bar(
            x_pos,
            total_capital_invested,
            bar_width_decomp,
            label='Capital Invested',
            color='steelblue',
            alpha=0.85,
            edgecolor='white',
            linewidth=1.5
        )
        
        bars_gain = ax_decomposition.bar(
            x_pos,
            gains_final,
            bar_width_decomp,
            bottom=total_capital_invested,
            label='Net Gain',
            color='darkorange',
            alpha=0.85,
            edgecolor='white',
            linewidth=1.5
        )
        
        ax_decomposition.set_xticks(x_pos)
        ax_decomposition.set_xticklabels(
            [acc.label for acc in self.accounts],
            rotation=25,
            ha='right',
            fontsize=9
        )
        ax_decomposition.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_decomposition.set_ylabel("Amount (CLP)", fontsize=10)
        ax_decomposition.set_title("Final State Decomposition", fontsize=11, fontweight='bold')
        ax_decomposition.legend(loc='upper left', fontsize=9, framealpha=0.95)
        ax_decomposition.grid(True, alpha=0.3, axis='y')
        
        # Enhanced annotations on orange bars only
        for m in range(M):
            if total_capital_invested[m] > 0 and gains_final[m] > 0:
                pct_gain = (gains_final[m] / total_capital_invested[m]) * 100
                gain_value_M = gains_final[m] / 1e6  # In millions
                
                # Position at center of orange bar
                y_position = total_capital_invested[m] + gains_final[m] / 2
                
                ax_decomposition.text(
                    x_pos[m],
                    y_position,
                    f'${gain_value_M:.1f}M\n(+{pct_gain:.1f}%)',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='darkorange', 
                            edgecolor='white', alpha=0.95, linewidth=1.5)
                )
        
        # ========== Date formatting ==========
        if start_date is not None:
            for ax in [ax_fractions, ax_absolute, ax_cumulative]:
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                tick_interval = max(3, T // 8) if T > 24 else max(1, T // 12)
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=tick_interval))
                
                for label in ax.get_xticklabels():
                    label.set_ha('right')
                    label.set_fontsize(8)
        
        # Main title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Summary annotation (2 lines with proper spacing)
        total_capital_invested_all = total_capital_invested.sum()
        total_wealth_all = total_wealth_final.sum()
        total_gains_all = gains_final.sum()
        overall_return_pct = (total_gains_all / total_capital_invested_all * 100) if total_capital_invested_all > 0 else 0
        
        param_text_line1 = (f"Horizon: {T} months  |  Simulations: {n_sims_actual}  |  "
                        f"Capital Invested: ${total_capital_invested_all:,.0f}".replace(",", "."))
        param_text_line2 = (f"Total Gains: ${total_gains_all:,.0f} (+{overall_return_pct:.1f}%)".replace(",", "."))
        
        # Adjust layout with proper bottom margin
        fig.tight_layout(rect=[0, 0.05, 1, 0.97 if title else 0.99])
        
        # Two-line annotation with proper spacing
        fig.text(0.01, 0.028, param_text_line1, ha='left', va='bottom', fontsize=8, alpha=0.85)
        fig.text(0.01, 0.008, param_text_line2, ha='left', va='bottom', fontsize=8, 
                alpha=0.95, fontweight='bold', color='darkorange')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if return_fig_ax:
            return fig, {
                'fractions': ax_fractions,
                'absolute': ax_absolute,
                'cumulative': ax_cumulative,
                'decomposition': ax_decomposition
            }

    def _plot_comparison(
        self,
        results: dict[str, SimulationResult],
        metric: str = "total_wealth",
        **kwargs
    ) -> Optional[tuple]:
        """
        Compare multiple strategies side-by-side.
        
        Creates 2-panel comparison:
        - Left: Mean trajectory over time
        - Right: Final distribution boxplots
        
        Parameters
        ----------
        results : dict[str, SimulationResult]
            Dictionary mapping strategy labels to simulation results.
        metric : str, default "total_wealth"
            Metric to compare. Options: "total_wealth", "wealth".
        **kwargs
            Plotting parameters (figsize, title, save_path, return_fig_ax).
        """
        if not isinstance(results, dict):
            raise TypeError("mode='comparison' requires results dict")
        
        import matplotlib.pyplot as plt
        
        figsize = kwargs.get("figsize", (14, 6))
        title = kwargs.get("title", f"Strategy Comparison: {metric}")
        save_path = kwargs.get("save_path", None)
        return_fig_ax = kwargs.get("return_fig_ax", False)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Mean trajectories
        for label, result in results.items():
            if metric == "total_wealth":
                data = result.total_wealth.mean(axis=0)
            elif metric == "wealth":
                # Sum across all accounts
                data = result.wealth.sum(axis=2).mean(axis=0)
            else:
                raise ValueError(
                    f"Unsupported metric '{metric}'. "
                    f"Valid: 'total_wealth', 'wealth'"
                )
            
            axes[0].plot(data, label=label, linewidth=2.5)
        
        axes[0].set_xlabel("Month", fontsize=11)
        axes[0].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        axes[0].set_title("Mean Trajectories", fontsize=12, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Final distribution boxplots
        final_data = [r.total_wealth[:, -1] for r in results.values()]
        bp = axes[1].boxplot(final_data, labels=results.keys(), patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1].set_ylabel("Final Wealth", fontsize=11)
        axes[1].set_title("Distribution at T", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if return_fig_ax:
            return fig, axes
