"""
Stochastic return generation for FinOpt portfolios.

Mathematical Model
------------------
Gross returns follow a correlated lognormal distribution:
    1 + R_t^m ~ LogNormal(μ_log^m, Σ)

where Σ = D @ ρ @ D is the covariance matrix constructed from:
    - D = diag(σ_log): vector of log-volatilities
    - ρ: correlation matrix (user-specified)

Parameter conversion from arithmetic to log-space:
    μ_log = log(1 + μ_arith) - σ_log² / 2
    σ_log = sqrt(log(1 + σ_arith² / (1 + μ_arith)²))

This ensures:
    E[R_t] ≈ μ_arith  (arithmetic mean)
    Std[R_t] ≈ σ_arith (arithmetic volatility)

Design Principles
-----------------
- Vectorized generation: (n_sims, T, M) in single call
- No temporal dependencies (IID across t)
- Correlation across portfolios (cross-sectional dependence)
- Guaranteed R_t > -1 (no bankruptcy scenarios)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

# Account is owned by portfolio.py (core domain entity)
from .portfolio import Account


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
    correlation_matrix : np.ndarray, shape (M, M)
        Cross-sectional correlation matrix (symmetric, positive semi-definite).
        Must satisfy: ρ_ii = 1, -1 ≤ ρ_ij ≤ 1.
    
    Methods
    -------
    generate(T, n_sims, seed) -> np.ndarray
        Returns correlated lognormal samples: (n_sims, T, M).
    
    Notes
    -----
    - Returns are IID across time (no GARCH/AR structure)
    - Correlation is constant (no regime switching)
    - Lognormal ensures R_t > -1 (realistic constraint)
    
    Mathematical Details
    --------------------
    For each portfolio m, we convert user-specified arithmetic parameters
    (μ_arith, σ_arith) to log-normal parameters (μ_log, σ_log) using:
    
        σ_log = sqrt(log(1 + σ_arith² / (1 + μ_arith)²))
        μ_log = log(1 + μ_arith) - σ_log² / 2
    
    Then sample:
        Z_t ~ N(μ_log, Σ)  where Σ = D @ ρ @ D
        R_t = exp(Z_t) - 1
    
    This construction guarantees:
        E[R_t] ≈ μ_arith
        Std[R_t] ≈ σ_arith
        R_t > -1  (always)
    """
    
    def __init__(self, accounts: List[Account], correlation_matrix: np.ndarray):
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        M = len(accounts)
        if correlation_matrix.shape != (M, M):
            raise ValueError(
                f"correlation_matrix shape {correlation_matrix.shape} "
                f"does not match number of accounts {M}"
            )
        
        # Validate correlation matrix
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("correlation_matrix must be symmetric")
        if not np.allclose(np.diag(correlation_matrix), 1.0):
            raise ValueError("correlation_matrix diagonal must be 1.0")
        
        # Check positive semi-definite (all eigenvalues ≥ 0)
        eigvals = np.linalg.eigvalsh(correlation_matrix)
        if np.any(eigvals < -1e-10):
            raise ValueError(
                f"correlation_matrix must be positive semi-definite "
                f"(min eigenvalue: {eigvals.min():.6f})"
            )
        
        self.accounts = accounts
        self.correlation = correlation_matrix
        self.M = M
        
        # Convert arithmetic parameters to log-normal parameters
        self._mu_log = np.zeros(M)
        self._sigma_log = np.zeros(M)
        
        for i, acc in enumerate(accounts):
            mu_arith = acc.return_strategy["mu"]
            sigma_arith = acc.return_strategy["sigma"]
            
            # Lognormal parameter conversion
            # σ_log = sqrt(log(1 + σ²/(1+μ)²))
            sigma_log = np.sqrt(np.log(1 + sigma_arith**2 / (1 + mu_arith)**2))
            
            # μ_log = log(1+μ) - σ_log²/2
            mu_log = np.log(1 + mu_arith) - 0.5 * sigma_log**2
            
            self._mu_log[i] = mu_log
            self._sigma_log[i] = sigma_log
        
        # Covariance matrix: Σ = D @ ρ @ D
        D = np.diag(self._sigma_log)
        self._cov_matrix = D @ correlation_matrix @ D
    
    def generate(
        self,
        T: int,
        n_sims: int = 1,
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
        seed : Optional[int]
            Random seed for reproducibility.
        
        Returns
        -------
        R : np.ndarray, shape (n_sims, T, M)
            Arithmetic returns for each simulation, period, and portfolio.
            Guaranteed: R[i, t, m] > -1 for all (i, t, m).
        
        Algorithm
        ---------
        1. Sample Z ~ N(μ_log, Σ) of shape (n_sims, T, M) directly
        2. Transform to gross returns: G = exp(Z)
        3. Convert to arithmetic returns: R = G - 1
        
        Complexity
        ----------
        Time: O(n_sims * T * M²) dominated by Cholesky decomposition in multivariate_normal
        Memory: O(n_sims * T * M) single allocation, no intermediate arrays
        
        Notes
        -----
        Vectorized implementation matching income.py pattern:
        - Direct 3D generation (no reshape overhead)
        - Single RNG seed for full batch reproducibility
        - Correlation handled internally by multivariate_normal
        """
        if T <= 0:
            return np.zeros((n_sims, 0, self.M), dtype=float)
        if n_sims <= 0:
            raise ValueError(f"n_sims must be positive, got {n_sims}")
        
        rng = np.random.default_rng(seed)
        
        # Direct 3D generation: (n_sims, T, M) without reshape
        Z = rng.multivariate_normal(
            mean=self._mu_log,
            cov=self._cov_matrix,
            size=(n_sims, T)
        )
        
        # Lognormal transformation: R = exp(Z) - 1
        # Vectorized over all dimensions
        R = np.exp(Z) - 1.0
        
        return R
    
    @property
    def mean_returns(self) -> np.ndarray:
        """Expected arithmetic returns (approximate, length M)."""
        # E[R] ≈ exp(μ_log + σ_log²/2) - 1
        return np.exp(self._mu_log + 0.5 * self._sigma_log**2) - 1.0
    
    @property
    def volatilities(self) -> np.ndarray:
        """Arithmetic volatilities (approximate, length M)."""
        # Std[R] ≈ sqrt((exp(σ_log²) - 1) * exp(2μ_log + σ_log²))
        return np.sqrt(
            (np.exp(self._sigma_log**2) - 1) * 
            np.exp(2 * self._mu_log + self._sigma_log**2)
        )