"""
Stochastic return generation for FinOpt portfolios.

Mathematical Model
------------------
Gross returns follow a correlated lognormal distribution:
    1 + R_t^m ~ LogNormal(μ_log^m, Σ)

where Σ = D @ ρ @ D is the covariance matrix constructed from:
    - D = diag(σ_log): vector of log-volatilities
    - ρ: correlation matrix (user-specified or default)

Parameter conversion from arithmetic to log-space:
    σ_log = sqrt(log(1 + σ_arith² / (1 + μ_arith)²))
    μ_log = log(1 + μ_arith) - σ_log² / 2

Design principles
-----------------
- Dual temporal representation: seamless access to monthly/annual parameters
- No Portfolio dependency: loose coupling via Account interface
- Correlation override: sensitivity analysis per generate() call
- Lognormal guarantee: R_t > -1 (realistic constraint)
"""

from __future__ import annotations
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .portfolio import Account
from .utils import monthly_to_annual

__all__ = ["ReturnModel"]


class ReturnModel:
    """
    Correlated lognormal return generator for multiple portfolios.
    
    Generates vectorized return samples (n_sims, T, M) where:
    - n_sims: number of Monte Carlo trajectories
    - T: time horizon (months)
    - M: number of portfolios (len(accounts))
    
    Parameters
    ----------
    accounts : List[Account]
        Portfolio specifications with return_strategy parameters.
    default_correlation : np.ndarray, shape (M, M), optional
        Default cross-sectional correlation matrix (symmetric, PSD).
        If None, uses identity (uncorrelated accounts).
        Can be overridden per generate() call.
    
    Properties
    ----------
    monthly_params : List[Dict[str, float]]
        Monthly return parameters for all accounts.
    annual_params : List[Dict[str, float]]
        Annualized return parameters for all accounts.
    
    Methods
    -------
    generate(T, n_sims, correlation, seed) -> np.ndarray
        Returns correlated lognormal samples: (n_sims, T, M).
    params_table() -> pd.DataFrame
        Comparison table of monthly vs annual parameters.
    plot(T, n_sims, correlation, seed, **kwargs)
        Visualize return distributions and statistics.
    plot_cumulative(T, n_sims, **kwargs)
        Visualize cumulative return trajectories.
    plot_horizon_analysis(horizons, **kwargs)
        Analyze risk-return profile across investment horizons.
    
    Notes
    -----
    - Returns are IID across time (no GARCH/AR structure)
    - Correlation can vary per generate() call (sensitivity analysis)
    - Lognormal ensures R_t > -1 (realistic constraint)
    
    Examples
    --------
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Growth", 0.12, 0.20)
    ... ]
    >>> returns = ReturnModel(accounts, default_correlation=np.eye(2))
    >>> 
    >>> # Introspection
    >>> print(returns.params_table())
    >>> print(returns)  # Human-readable summary
    >>> 
    >>> # Generation
    >>> R = returns.generate(T=24, n_sims=500, seed=42)
    >>> R.shape
    (500, 24, 2)
    """
    
    def __init__(
        self, 
        accounts: List[Account],
        default_correlation: Optional[np.ndarray] = None
    ):
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        M = len(accounts)
        
        # Default: uncorrelated accounts
        if default_correlation is None:
            default_correlation = np.eye(M)
        
        if default_correlation.shape != (M, M):
            raise ValueError(
                f"default_correlation shape {default_correlation.shape} "
                f"does not match number of accounts {M}"
            )
        
        # Validate correlation matrix
        self._validate_correlation(default_correlation)
        
        self.accounts = accounts
        self.default_correlation = default_correlation
        self.M = M
        
        # Convert arithmetic parameters to log-normal parameters (precompute)
        self._mu_log = np.zeros(M)
        self._sigma_log = np.zeros(M)
        
        for i, acc in enumerate(accounts):
            mu_arith = acc.return_strategy["mu"]
            sigma_arith = acc.return_strategy["sigma"]
            
            # Lognormal parameter conversion
            sigma_log = np.sqrt(np.log(1 + sigma_arith**2 / (1 + mu_arith)**2))
            mu_log = np.log(1 + mu_arith) - 0.5 * sigma_log**2
            
            self._mu_log[i] = mu_log
            self._sigma_log[i] = sigma_log
    
    @staticmethod
    def _validate_correlation(rho: np.ndarray):
        """Validate correlation matrix properties."""
        if not np.allclose(rho, rho.T):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(rho), 1.0):
            raise ValueError("correlation matrix diagonal must be 1.0")
        
        eigvals = np.linalg.eigvalsh(rho)
        if np.any(eigvals < -1e-10):
            raise ValueError(
                f"correlation matrix must be positive semi-definite "
                f"(min eigenvalue: {eigvals.min():.6f})"
            )
    
    def _build_covariance(self, correlation: np.ndarray) -> np.ndarray:
        """Build log-space covariance: Σ = D @ ρ @ D."""
        D = np.diag(self._sigma_log)
        return D @ correlation @ D
    
    # ========== Dual temporal representation (NEW) ==========
    
    @property
    def monthly_params(self) -> List[dict[str, float]]:
        """
        Monthly return parameters for all accounts.
        
        Returns
        -------
        List[Dict[str, float]]
            List of dicts with keys {"mu", "sigma"} in monthly space.
        
        Examples
        --------
        >>> returns.monthly_params
        [{'mu': 0.00327, 'sigma': 0.01443}, {'mu': 0.00948, 'sigma': 0.05773}]
        """
        return [acc.monthly_params for acc in self.accounts]
    
    @property
    def annual_params(self) -> List[dict[str, float]]:
        """
        Annualized return parameters for all accounts.
        
        Converts internal monthly parameters to annualized equivalents.
        
        Returns
        -------
        List[Dict[str, float]]
            List of dicts with keys {"return", "volatility"} in annual space.
        
        Examples
        --------
        >>> returns.annual_params
        [{'return': 0.04, 'volatility': 0.05}, {'return': 0.12, 'volatility': 0.20}]
        """
        return [acc.annual_params for acc in self.accounts]
    
    def params_table(self) -> pd.DataFrame:
        """
        Comparison table of monthly vs annual parameters.
        
        Returns
        -------
        pd.DataFrame
            Table with columns: Account, μ (monthly), μ (annual), 
            σ (monthly), σ (annual).
        
        Examples
        --------
        >>> print(returns.params_table())
                     μ (monthly)  μ (annual)  σ (monthly)  σ (annual)
        Emergency        0.0033       4.00%       0.0144       5.00%
        Growth           0.0095      12.00%       0.0577      20.00%
        """
        rows = []
        for acc in self.accounts:
            mp, ap = acc.monthly_params, acc.annual_params
            rows.append({
                "Account": acc.name,
                "μ (monthly)": f"{mp['mu']:.4f}",
                "μ (annual)": f"{ap['return']:.2%}",
                "σ (monthly)": f"{mp['sigma']:.4f}",
                "σ (annual)": f"{ap['volatility']:.2%}"
            })
        return pd.DataFrame(rows).set_index("Account")
    
    def __repr__(self) -> str:
        """
        String representation showing annualized parameters (user-friendly).
        
        Examples
        --------
        >>> print(returns)
        ReturnModel(M=2, ρ=eye, accounts=['Emergency': 4.0%/year, 'Growth': 12.0%/year])
        """
        corr_desc = "eye" if np.allclose(self.default_correlation, np.eye(self.M)) else "custom"
        acc_summary = [f"'{acc.name}': {acc.annual_params['return']:.1%}/year" 
                       for acc in self.accounts]
        return (f"ReturnModel(M={self.M}, ρ={corr_desc}, "
                f"accounts=[{', '.join(acc_summary)}])")
    
    # ========== Core generation method ==========
    
    def generate(
        self,
        T: int,
        n_sims: int = 1,
        correlation: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate correlated lognormal return samples.
        
        Parameters
        ----------
        T : int
            Time horizon (number of periods).
        n_sims : int, default 1
            Number of Monte Carlo trajectories.
        correlation : np.ndarray, shape (M, M), optional
            Correlation matrix override. If None, uses default_correlation.
        seed : Optional[int]
            Random seed for reproducibility.
        
        Returns
        -------
        R : np.ndarray, shape (n_sims, T, M)
            Arithmetic returns for each simulation, period, and portfolio.
            Guaranteed: R[i, t, m] > -1 for all (i, t, m).
        
        Algorithm
        ---------
        1. Use correlation override or default
        2. Build Σ = D @ ρ @ D
        3. Sample Z ~ N(μ_log, Σ) of shape (n_sims, T, M)
        4. Transform: R = exp(Z) - 1
        """
        if T <= 0:
            return np.zeros((n_sims, 0, self.M), dtype=float)
        if n_sims <= 0:
            raise ValueError(f"n_sims must be positive, got {n_sims}")
        
        # Use override or default correlation
        rho = correlation if correlation is not None else self.default_correlation
        self._validate_correlation(rho)
        
        # Build covariance matrix
        cov = self._build_covariance(rho)
        
        # Generate returns
        rng = np.random.default_rng(seed)
        Z = rng.multivariate_normal(
            mean=self._mu_log,
            cov=cov,
            size=(n_sims, T)
        )
        
        R = np.exp(Z) - 1.0
        return R
    
    # ========== Visualization methods ==========
    
    def plot(
        self,
        T: int = 32,
        n_sims: int = 300,
        correlation: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        start: Optional[date] = None,
        figsize: tuple = (16, 8),
        title: Optional[str] = r'Monthly Return Distributions $(R_t^m)$',
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        show_trajectories: bool = True,
        trajectory_alpha: float = 0.05,
    ):
        """
        Visualize return distributions with 3 panels.
        
        Panel layout:
        - Top-left: Return trajectories (individual simulations)
        - Top-right: Marginal histograms (monthly return distributions)
        - Bottom: Summary statistics table (monthly + annualized metrics)
        
        Parameters
        ----------
        T : int, default 32
            Time horizon (months).
        n_sims : int, default 300
            Number of simulations to generate.
        correlation : np.ndarray, optional
            Correlation matrix override.
        seed : Optional[int]
            Random seed for reproducibility.
        start : Optional[date], default None
            Start date for temporal axis. If None, uses numeric month indices (0, 1, 2, ...).
            If provided, x-axis shows calendar dates (first-of-month).
            Aligns with income.py and portfolio.py temporal representation.
        figsize : tuple, default (16, 8)
            Figure size (width, height).
        title : str, optional
            Main title for the figure.
        save_path : str, optional
            Path to save figure.
        return_fig_ax : bool, default False
            If True, returns (fig, axes_dict).
        show_trajectories : bool, default True
            Whether to show individual paths in trajectories panel.
        trajectory_alpha : float, default 0.05
            Transparency for trajectory lines.
        
        Returns
        -------
        None or (fig, axes_dict)
            If return_fig_ax=True, returns figure and dict of axes.
            
        Examples
        --------
        >>> from datetime import date
        >>> 
        >>> # Numeric time axis (default)
        >>> returns.plot(T=24, n_sims=300, seed=42)
        >>> 
        >>> # Calendar time axis
        >>> returns.plot(T=24, n_sims=300, seed=42, start=date(2025, 1, 1))
        """
        # Generate returns
        R = self.generate(T, n_sims, correlation, seed)  # (n_sims, T, M)
        
        if start is not None:
            from .utils import month_index
            time_axis = month_index(start, T)
            xlabel = "Date"
        else:
            time_axis = np.arange(T)
            xlabel = "Month"
        
        # Setup figure: 2 rows, 2 columns (bottom row merged)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[1, 0.8])
        
        ax_traj = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, :])
        
        # Account names and colors
        names = [acc.name for acc in self.accounts]
        colors = plt.cm.Dark2(np.linspace(0, 1, self.M))
        
        # ========== Panel 1: Trajectories ==========
        if show_trajectories:
            for m in range(self.M):
                for i in range(n_sims):
                    ax_traj.plot(
                        time_axis, 
                        R[i, :, m] * 100,
                        color=colors[m], 
                        alpha=trajectory_alpha,
                        linewidth=0.8,
                        label=names[m] if i == 0 else '_nolegend_'
                    )
        
        for m in range(self.M):
            mean_path = R[:, :, m].mean(axis=0) * 100
            ax_traj.plot(time_axis, mean_path, color=colors[m], linewidth=2.5,
                        label=f"{names[m]} (mean)")
        
        ax_traj.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_traj.set_xlabel(xlabel)
        ax_traj.set_ylabel("Monthly Return (%)")
        ax_traj.set_title("Return Trajectories")
        ax_traj.legend(loc='best', fontsize=8)
        ax_traj.grid(True, alpha=0.3)
        
        # ========== Panel 2: Histograms ==========
        R_flat = R.reshape(-1, self.M)
        for m in range(self.M):
            ax_hist.hist(R_flat[:, m] * 100, bins=50, alpha=0.6, 
                        color=colors[m], label=names[m], density=True)
        
        ax_hist.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_hist.set_xlabel("Monthly Return (%)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Return Distributions")
        ax_hist.legend(loc='best')
        ax_hist.grid(True, alpha=0.3)
        
        # ========== Panel 3: Statistics ==========
        ax_stats.axis('off')
        
        stats_data = []
        for m in range(self.M):
            R_m = R[:, :, m].flatten() * 100
            
            mean_m = R_m.mean()
            std_m = R_m.std()
            median_m = np.percentile(R_m, 50)
            q25_m = np.percentile(R_m, 25)
            q75_m = np.percentile(R_m, 75)
            
            mean_a = ((1 + mean_m/100)**12 - 1) * 100
            std_a = std_m * np.sqrt(12)
            median_a = ((1 + median_m/100)**12 - 1) * 100
            q25_a = ((1 + q25_m/100)**12 - 1) * 100
            q75_a = ((1 + q75_m/100)**12 - 1) * 100
            
            stats_data.append([
                f"{names[m]}",
                f"{mean_m:.2f}%",
                f"{std_m:.2f}%",
                f"{median_m:.2f}%",
                f"[{q25_m:.2f}%, {q75_m:.2f}%]",
                f"{R_m.min():.2f}%",
                f"{R_m.max():.2f}%"
            ])
            
            stats_data.append([
                f"  ↳ Annualized",
                f"{mean_a:.2f}%",
                f"{std_a:.2f}%",
                f"{median_a:.2f}%",
                f"[{q25_a:.2f}%, {q75_a:.2f}%]",
                "—",
                "—"
            ])
        
        headers = ['Account', 'Mean', 'Std', 'Median', 'IQR (Q25-Q75)', 'Min', 'Max']
        table = ax_stats.table(
            cellText=stats_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(style='italic', size=8)
        
        ax_stats.set_title("Summary Statistics (Monthly & Annualized)", 
                        pad=15, fontsize=11, fontweight='bold')
        
        # ========== Formateo de fechas (si aplica) ==========
        if start is not None:
            ax_traj.tick_params(axis='x', rotation=45)
            ax_traj.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            # Espaciado inteligente de ticks
            ax_traj.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, T//12)))
        
        # ========== Main title y anotaciones ==========
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        param_text = f"T={T} months | n_sims={n_sims} | seed={seed}"
        fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if return_fig_ax:
            return fig, {
                'trajectories': ax_traj,
                'histograms': ax_hist,
                'statistics': ax_stats
            }
    
    def plot_cumulative(
        self,
        T: int = 24,
        n_sims: int = 1000,
        correlation: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        start: Optional[date] = None,
        figsize: tuple = (14, 6),
        title: Optional[str] = r'Cumulative Returns per Account $(\prod_{t=0}^{T-1}(1 + R_t^m) - 1)$',
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        show_trajectories: bool = True,
        trajectory_alpha: float = 0.08,
        show_percentiles: bool = True,
        percentiles: tuple = (5, 95),
        hist_bins: int = 40,
        hist_color: str = 'red',
    ):
        """
        Visualize cumulative return trajectories with final distribution histogram.
        
        For M=1: Single plot with lateral histogram.
        For M>1: Separate subplot per account with individual histograms.
        
        Parameters
        ----------
        T : int, default 24
            Time horizon (months).
        n_sims : int, default 1000
            Number of Monte Carlo simulations.
        correlation : np.ndarray, optional
            Correlation matrix override.
        seed : Optional[int]
            Random seed for reproducibility.
        start : Optional[date], default None
            Start date for temporal axis. If None, uses numeric month indices (0, 1, 2, ...).
            If provided, x-axis shows calendar dates (first-of-month).
            Aligns with income.py and portfolio.py temporal representation.
        figsize : tuple, default (14, 6)
            Figure size (width, height).
        title : str, optional
            Main title for the figure.
        save_path : str, optional
            Path to save figure.
        return_fig_ax : bool, default False
            If True, returns (fig, ax, ax_hist) for M=1 or (fig, axes, ax_hists) for M>1.
        show_trajectories : bool, default True
            Whether to show individual simulation paths.
        trajectory_alpha : float, default 0.08
            Transparency for trajectory lines.
        show_percentiles : bool, default True
            Whether to show percentile bands (P5-P95 by default).
        percentiles : tuple, default (5, 95)
            Percentile bounds for confidence band.
        hist_bins : int, default 40
            Number of bins for final distribution histogram.
        hist_color : str, default 'red'
            Color for the lateral histogram.
        
        Returns
        -------
        None or tuple
            If return_fig_ax=True:
            - M=1: (fig, ax, ax_hist)
            - M>1: (fig, axes, ax_hists)
            
        Examples
        --------
        >>> from datetime import date
        >>> 
        >>> # Numeric time axis (default)
        >>> returns.plot_cumulative(T=24, n_sims=500, seed=42)
        >>> 
        >>> # Calendar time axis
        >>> returns.plot_cumulative(T=24, n_sims=500, seed=42, start=date(2025, 1, 1))
        """
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        # Generate returns and compute cumulative
        R = self.generate(T, n_sims, correlation, seed)
        gross_returns = 1.0 + R
        cumulative_gross = np.ones((n_sims, T + 1, self.M))
        cumulative_gross[:, 1:, :] = np.cumprod(gross_returns, axis=1)
        cumulative_returns_pct = (cumulative_gross - 1.0) * 100
        
        # ========== Construcción del eje temporal ==========
        if start is not None:
            from .utils import month_index
            time_axis = month_index(start, T + 1)
            xlabel = "Date"
        else:
            time_axis = np.arange(T + 1)
            xlabel = "Month"
        
        names = [acc.name for acc in self.accounts]
        colors = plt.cm.tab10(np.linspace(0, 1, self.M))
        
        if self.M == 1:
            # ===== SINGLE ACCOUNT MODE =====
            fig, ax = plt.subplots(figsize=figsize)
            m = 0
            
            if show_trajectories:
                for i in range(n_sims):
                    ax.plot(time_axis, cumulative_returns_pct[i, :, m],
                        color=colors[m], alpha=trajectory_alpha, linewidth=0.8,
                        label='_nolegend_')
            
            mean_cumulative = cumulative_returns_pct[:, :, m].mean(axis=0)
            ax.plot(time_axis, mean_cumulative, color=colors[m], linewidth=2.5,
                label=f"{names[m]} (mean)")
            
            if show_percentiles:
                p_low = np.percentile(cumulative_returns_pct[:, :, m], percentiles[0], axis=0)
                p_high = np.percentile(cumulative_returns_pct[:, :, m], percentiles[1], axis=0)
                ax.fill_between(time_axis, p_low, p_high, color=colors[m], alpha=0.25,
                            label=f'P{percentiles[0]}-P{percentiles[1]}')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax.set_title(title or f'Monte Carlo Simulation ({n_sims} Scenarios): {names[m]}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', labelsize=11)
            ax.legend(loc='upper left', fontsize=10)
            
            # Lateral histogram
            ax_hist = None
            if n_sims > 1:
                divider = make_axes_locatable(ax)
                ax_hist = divider.append_axes("right", size=1.2, pad=0.15)
                final_returns = cumulative_returns_pct[:, -1, m]
                ax_hist.hist(final_returns, bins=hist_bins, orientation='horizontal',
                            color=hist_color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax_hist.set_xlabel('Count', fontsize=11)
                ax_hist.set_ylim(ax.get_ylim())
                ax_hist.tick_params(labelsize=10)
                ax_hist.grid(True, alpha=0.2)
                ax_hist.set_title('Final\nDistribution', fontsize=10)
            
            # Theoretical validation
            final_returns = cumulative_returns_pct[:, -1, m]
            mean_sim, std_sim = final_returns.mean(), final_returns.std()
            
            mu_monthly = self.accounts[m].monthly_params["mu"]
            expected_theoretical = ((1 + mu_monthly) ** T - 1) * 100
            
            validation_text = (f"Simulation:\nMean: {mean_sim:.1f}%\nStd: {std_sim:.1f}%\n\n"
                            f"Theoretical:\nMean: {expected_theoretical:.1f}%")
            ax.text(0.02, 0.98, validation_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # ========== Formateo de fechas (si aplica) ==========
            if start is not None:
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, T//12)))
            
            param_text = f"T={T} | n_sims={n_sims} | seed={seed}"
            fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            
            return (fig, ax, ax_hist) if return_fig_ax else None
        
        else:
            # ===== MULTIPLE ACCOUNTS MODE =====
            fig, axes = plt.subplots(self.M, 1, figsize=(figsize[0], figsize[1] * self.M * 0.8))
            if self.M == 1:
                axes = [axes]
            
            ax_hists = []
            
            for m in range(self.M):
                ax = axes[m]
                
                if show_trajectories:
                    for i in range(n_sims):
                        ax.plot(time_axis, cumulative_returns_pct[i, :, m],
                            color=colors[m], alpha=trajectory_alpha, linewidth=0.8,
                            label='_nolegend_')
                
                mean_cumulative = cumulative_returns_pct[:, :, m].mean(axis=0)
                ax.plot(time_axis, mean_cumulative, color=colors[m], linewidth=2.5,
                    label='Mean')
                
                if show_percentiles:
                    p_low = np.percentile(cumulative_returns_pct[:, :, m], percentiles[0], axis=0)
                    p_high = np.percentile(cumulative_returns_pct[:, :, m], percentiles[1], axis=0)
                    ax.fill_between(time_axis, p_low, p_high, color=colors[m], alpha=0.25,
                                label=f'P{percentiles[0]}-P{percentiles[1]}')
                
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel('Cumulative Return (%)', fontsize=11)
                ax.set_title(names[m], fontsize=12, fontweight='bold', loc='left')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(labelsize=10)
                ax.legend(loc='upper left', fontsize=9)
                
                if n_sims > 1:
                    divider = make_axes_locatable(ax)
                    ax_hist = divider.append_axes("right", size=1.0, pad=0.15)
                    final_returns = cumulative_returns_pct[:, -1, m]
                    ax_hist.hist(final_returns, bins=hist_bins, orientation='horizontal',
                                color=colors[m], alpha=0.6, edgecolor='black', linewidth=0.5)
                    ax_hist.set_xlabel('Count', fontsize=10)
                    ax_hist.set_ylim(ax.get_ylim())
                    ax_hist.tick_params(labelsize=9)
                    ax_hist.grid(True, alpha=0.2)
                    ax_hists.append(ax_hist)
                else:
                    ax_hists.append(None)
                
                # ========== Formateo de fechas (si aplica) ==========
                if start is not None:
                    ax.tick_params(axis='x', rotation=45)
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, T//12)))
            
            main_title = title or f'Cumulative Return Evolution by Account ({n_sims} Scenarios, T={T} months)'
            fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)
            
            param_text = f"n_sims={n_sims} | seed={seed}"
            fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.99])
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            
            return (fig, axes, ax_hists) if return_fig_ax else None
    
    def plot_horizon_analysis(
        self,
        horizons: np.ndarray = np.array([1, 2, 3, 5, 10, 15, 20]),
        figsize: tuple = (15, 5),
        title: Optional[str] = 'Horizon Analysis: Average Portfolio Behavior',
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        show_table: bool = True,
    ):
        """
        Analyze expected returns, volatility, and probability of loss across investment horizons.
        

        """
        from scipy import stats
        
        T_months = horizons * 12
        
        if self.M == 1:
            # ===== SINGLE ACCOUNT MODE =====
            mu_monthly = self.accounts[0].monthly_params["mu"]  # Usar monthly_params
            sigma_monthly = self.accounts[0].monthly_params["sigma"]
            account_label = self.accounts[0].name
            
            expected_return = np.power(1 + mu_monthly, T_months) - 1
            volatility = sigma_monthly * np.sqrt(T_months)
            prob_loss = stats.norm.cdf(0, expected_return, volatility)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.where(volatility > 0, expected_return / volatility, np.inf)
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Panel 1: Return vs Volatility
            axes[0].plot(horizons, expected_return * 100, 'o-', linewidth=2.5, markersize=8,
                        label='Expected Return', color='green')
            axes[0].plot(horizons, volatility * 100, 's-', linewidth=2.5, markersize=8,
                        label='Volatility (±1σ)', color='red')
            axes[0].set_xlabel('Horizon (years)', fontsize=12)
            axes[0].set_ylabel('Percentage (%)', fontsize=12)
            axes[0].set_title('Return vs Volatility by Horizon', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].tick_params(labelsize=11)
            
            if len(horizons) >= 2:
                snr_1y = snr[0] if horizons[0] == 1 else snr[np.argmin(np.abs(horizons - 1))]
                snr_last = snr[-1]
                axes[0].text(0.98, 0.02, f"SNR improvement:\n{snr_1y:.2f} → {snr_last:.2f} ({snr_last/snr_1y:.1f}x)",
                            transform=axes[0].transAxes, fontsize=9, va='bottom', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Panel 2: Probability of Loss
            axes[1].plot(horizons, prob_loss * 100, 'o-', linewidth=2.5, markersize=8, color='darkred')
            
            if 1.0 in horizons:
                p_loss_1y = prob_loss[horizons == 1.0][0] * 100
                axes[1].axhline(y=p_loss_1y, color='orange', linestyle='--', alpha=0.5,
                            linewidth=1.5, label=f'1-year: {p_loss_1y:.1f}%')
                axes[1].legend(fontsize=10)
            
            axes[1].set_xlabel('Horizon (years)', fontsize=12)
            axes[1].set_ylabel('Probability of Loss (%)', fontsize=12)
            axes[1].set_title('P(Return < 0%) by Horizon', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].tick_params(labelsize=11)
            
            if len(horizons) >= 2:
                p_loss_first, p_loss_last = prob_loss[0] * 100, prob_loss[-1] * 100
                reduction = p_loss_first - p_loss_last
                axes[1].text(0.98, 0.98, f"Risk reduction:\n{p_loss_first:.1f}% → {p_loss_last:.1f}%\n(-{reduction:.1f} pp)",
                            transform=axes[1].transAxes, fontsize=9, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            param_text = f"{account_label} | μ_monthly={mu_monthly*100:.2f}% | σ_monthly={sigma_monthly*100:.2f}%"
            fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            
            if show_table:
                _print_horizon_table(horizons, expected_return, volatility, prob_loss, snr, account_label)
            
            return (fig, axes) if return_fig_ax else None
        
        else:
            # ===== MULTIPLE ACCOUNTS MODE =====
            fig, axes = plt.subplots(self.M, 2, figsize=(figsize[0], figsize[1] * self.M * 0.75))
            colors = plt.cm.tab10(np.linspace(0, 1, self.M))
            
            for m in range(self.M):
                mu_monthly = self.accounts[m].monthly_params["mu"]  # Usar monthly_params
                sigma_monthly = self.accounts[m].monthly_params["sigma"]
                account_label = self.accounts[m].name
                
                expected_return = np.power(1 + mu_monthly, T_months) - 1
                volatility = sigma_monthly * np.sqrt(T_months)
                prob_loss = stats.norm.cdf(0, expected_return, volatility)
                with np.errstate(divide='ignore', invalid='ignore'):
                    snr = np.where(volatility > 0, expected_return / volatility, np.inf)
                
                # Panel 1: Return vs Volatility
                ax0 = axes[m, 0]
                ax0.plot(horizons, expected_return * 100, 'o-', linewidth=2.5, markersize=8,
                        label='Expected Return', color='green')
                ax0.plot(horizons, volatility * 100, 's-', linewidth=2.5, markersize=8,
                        label='Volatility (±1σ)', color='red')
                ax0.set_xlabel('Horizon (years)', fontsize=11)
                ax0.set_ylabel('Percentage (%)', fontsize=11)
                ax0.set_title(f'{account_label}: Return vs Volatility', fontsize=12, fontweight='bold', loc='left')
                ax0.legend(fontsize=9)
                ax0.grid(True, alpha=0.3, linestyle='--')
                ax0.tick_params(labelsize=10)
                
                if len(horizons) >= 2:
                    snr_1y = snr[0] if horizons[0] == 1 else snr[np.argmin(np.abs(horizons - 1))]
                    snr_last = snr[-1]
                    ax0.text(0.98, 0.02, f"SNR: {snr_1y:.2f} → {snr_last:.2f}",
                            transform=ax0.transAxes, fontsize=8, va='bottom', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Panel 2: Probability of Loss
                ax1 = axes[m, 1]
                ax1.plot(horizons, prob_loss * 100, 'o-', linewidth=2.5, markersize=8, color=colors[m])
                
                if 1.0 in horizons:
                    p_loss_1y = prob_loss[horizons == 1.0][0] * 100
                    ax1.axhline(y=p_loss_1y, color='orange', linestyle='--', alpha=0.5,
                            linewidth=1.5, label=f'1y: {p_loss_1y:.1f}%')
                    ax1.legend(fontsize=8)
                
                ax1.set_xlabel('Horizon (years)', fontsize=11)
                ax1.set_ylabel('P(Loss) (%)', fontsize=11)
                ax1.set_title(f'{account_label}: Probability of Loss', fontsize=12, fontweight='bold', loc='left')
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.tick_params(labelsize=10)
                
                if len(horizons) >= 2:
                    p_loss_first, p_loss_last = prob_loss[0] * 100, prob_loss[-1] * 100
                    reduction = p_loss_first - p_loss_last
                    ax1.text(0.98, 0.98, f"{p_loss_first:.1f}% → {p_loss_last:.1f}%\n(-{reduction:.1f} pp)",
                            transform=ax1.transAxes, fontsize=8, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                if show_table:
                    _print_horizon_table(horizons, expected_return, volatility, prob_loss, snr, account_label)
            
            main_title = title or f'Horizon Analysis by Account'
            fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.99])
            
            if save_path:
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
            
            return (fig, axes) if return_fig_ax else None

    
    # ========== Legacy properties (backward compatibility) ==========
    
    @property
    def mean_returns(self) -> np.ndarray:
        """Expected arithmetic returns (monthly, length M)."""
        return np.exp(self._mu_log + 0.5 * self._sigma_log**2) - 1.0
    
    @property
    def volatilities(self) -> np.ndarray:
        """Arithmetic volatilities (monthly, length M)."""
        return np.sqrt(
            (np.exp(self._sigma_log**2) - 1) * 
            np.exp(2 * self._mu_log + self._sigma_log**2)
        )
    
    @property
    def account_names(self) -> List[str]:
        """List of account names."""
        return [acc.name for acc in self.accounts]


def _print_horizon_table(horizons, expected_return, volatility, prob_loss, snr, account_label):
    """Helper function to print horizon analysis table."""
    from scipy import stats

    print("\n" + "="*85)
    print(f"HORIZON ANALYSIS - {account_label}")
    print("="*85)
    print(f"{'Horizon':>8} | {'Expected':>9} | {'Volatility':>10} | {'P(Loss)':>8} | {'P25-P75':>10} | {'SNR':>6}")
    print(f"{'(years)':>8} | {'Return':>9} | {'(±1σ)':>10} | {'':>8} | {'Range':>10} | {'':>6}")
    print("-"*85)

    for i, T in enumerate(horizons):
        exp_ret = expected_return[i] * 100
        vol = volatility[i] * 100
        p_loss = prob_loss[i] * 100
        
        p25 = stats.norm.ppf(0.25, expected_return[i], volatility[i]) * 100
        p75 = stats.norm.ppf(0.75, expected_return[i], volatility[i]) * 100
        p_range = p75 - p25
        
        snr_val = snr[i]
        
        print(f"{T:7.1f}  | {exp_ret:8.1f}% | {vol:9.1f}% | {p_loss:7.1f}% | {p_range:9.1f}% | {snr_val:5.2f}")

    print("="*85)
    if account_label == _print_horizon_table.__dict__.get('_last_account'):
        return  # Skip notes for subsequent tables
    _print_horizon_table.__dict__['_last_account'] = account_label