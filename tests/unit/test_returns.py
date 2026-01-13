"""
Unit tests for returns.py module.

Tests ReturnModel class for correlated lognormal return generation.
"""

import pytest
import numpy as np

from src.portfolio import Account
from src.returns import ReturnModel


# ============================================================================
# RETURNMODEL INSTANTIATION TESTS
# ============================================================================

class TestReturnModelInstantiation:
    """Test ReturnModel initialization."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    def test_basic_instantiation(self, accounts):
        """Test basic ReturnModel creation."""
        model = ReturnModel(accounts)

        assert model.M == 2
        assert model.accounts == accounts

    def test_with_correlation(self, accounts):
        """Test ReturnModel with custom correlation."""
        corr = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])
        model = ReturnModel(accounts, default_correlation=corr)

        np.testing.assert_array_almost_equal(model.default_correlation, corr)

    def test_default_correlation_is_identity(self, accounts):
        """Test default correlation is identity matrix."""
        model = ReturnModel(accounts)

        expected = np.eye(len(accounts))
        np.testing.assert_array_almost_equal(model.default_correlation, expected)

    def test_single_account(self):
        """Test ReturnModel with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        model = ReturnModel(accounts)

        assert model.M == 1

    def test_three_accounts(self):
        """Test ReturnModel with three accounts."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05),
            Account.from_annual("B", 0.08, 0.10),
            Account.from_annual("C", 0.14, 0.18),
        ]
        model = ReturnModel(accounts)

        assert model.M == 3

    def test_mismatched_correlation_shape_raises(self, accounts):
        """Test that wrong correlation shape raises ValueError."""
        wrong_corr = np.eye(3)  # Wrong size for 2 accounts
        with pytest.raises(ValueError, match="correlation"):
            ReturnModel(accounts, default_correlation=wrong_corr)


# ============================================================================
# RETURNMODEL GENERATE TESTS
# ============================================================================

class TestReturnModelGenerate:
    """Test ReturnModel.generate() method."""

    @pytest.fixture
    def return_model(self):
        """Create test return model."""
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return ReturnModel(accounts)

    def test_generate_shape(self, return_model):
        """Test generate() returns correct shape."""
        T = 24
        n_sims = 100

        R = return_model.generate(T=T, n_sims=n_sims, seed=42)

        # Shape: (n_sims, T, M)
        assert R.shape == (n_sims, T, return_model.M)

    def test_generate_dtype(self, return_model):
        """Test generate() returns float array."""
        R = return_model.generate(T=10, n_sims=50, seed=42)

        assert R.dtype == np.float64

    def test_generate_single_simulation(self, return_model):
        """Test generate() with n_sims=1."""
        R = return_model.generate(T=24, n_sims=1, seed=42)

        assert R.shape == (1, 24, return_model.M)

    def test_generate_single_period(self, return_model):
        """Test generate() with T=1."""
        R = return_model.generate(T=1, n_sims=100, seed=42)

        assert R.shape == (100, 1, return_model.M)

    def test_generate_finite_values(self, return_model):
        """Test that all generated returns are finite."""
        R = return_model.generate(T=100, n_sims=1000, seed=42)

        assert np.all(np.isfinite(R))

    def test_generate_returns_greater_than_minus_one(self, return_model):
        """Test lognormal guarantee: R > -1."""
        R = return_model.generate(T=100, n_sims=1000, seed=42)

        assert np.all(R > -1.0), f"Found returns <= -1: min={R.min()}"


# ============================================================================
# RETURNMODEL REPRODUCIBILITY TESTS
# ============================================================================

class TestReturnModelReproducibility:
    """Test seed-based reproducibility."""

    @pytest.fixture
    def return_model(self):
        """Create test return model."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]
        return ReturnModel(accounts)

    def test_same_seed_same_output(self, return_model):
        """Test same seed produces identical results."""
        R1 = return_model.generate(T=24, n_sims=100, seed=42)
        R2 = return_model.generate(T=24, n_sims=100, seed=42)

        np.testing.assert_array_equal(R1, R2)

    def test_different_seeds_different_output(self, return_model):
        """Test different seeds produce different results."""
        R1 = return_model.generate(T=24, n_sims=100, seed=42)
        R2 = return_model.generate(T=24, n_sims=100, seed=43)

        assert not np.allclose(R1, R2)

    def test_no_seed_varies(self, return_model):
        """Test that no seed produces varying results."""
        R1 = return_model.generate(T=24, n_sims=100, seed=None)
        R2 = return_model.generate(T=24, n_sims=100, seed=None)

        # Very unlikely to be identical
        assert not np.allclose(R1, R2)


# ============================================================================
# RETURNMODEL CORRELATION TESTS
# ============================================================================

class TestReturnModelCorrelation:
    """Test correlation structure."""

    def test_identity_correlation_uncorrelated(self):
        """Test that identity correlation produces uncorrelated returns."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.08, 0.10),
        ]
        model = ReturnModel(accounts, default_correlation=np.eye(2))

        R = model.generate(T=100, n_sims=1000, seed=42)

        # Compute empirical correlation
        R_flat = R.reshape(-1, 2)
        empirical_corr = np.corrcoef(R_flat[:, 0], R_flat[:, 1])[0, 1]

        # Should be close to 0
        assert abs(empirical_corr) < 0.15

    def test_positive_correlation(self):
        """Test that positive correlation is preserved."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.08, 0.10),
        ]
        corr = np.array([
            [1.0, 0.7],
            [0.7, 1.0],
        ])
        model = ReturnModel(accounts, default_correlation=corr)

        R = model.generate(T=100, n_sims=1000, seed=42)

        # Compute empirical correlation
        R_flat = R.reshape(-1, 2)
        empirical_corr = np.corrcoef(R_flat[:, 0], R_flat[:, 1])[0, 1]

        # Should be significantly positive
        assert empirical_corr > 0.5

    def test_correlation_override(self):
        """Test correlation override in generate()."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.08, 0.10),
        ]
        model = ReturnModel(accounts)  # Default: identity

        override_corr = np.array([
            [1.0, 0.9],
            [0.9, 1.0],
        ])
        R = model.generate(T=100, n_sims=1000, correlation=override_corr, seed=42)

        # Compute empirical correlation
        R_flat = R.reshape(-1, 2)
        empirical_corr = np.corrcoef(R_flat[:, 0], R_flat[:, 1])[0, 1]

        # Should reflect override, not default
        assert empirical_corr > 0.7


# ============================================================================
# RETURNMODEL STATISTICS TESTS
# ============================================================================

class TestReturnModelStatistics:
    """Test statistical properties of generated returns."""

    def test_mean_return_reasonable(self):
        """Test that mean return is in reasonable range."""
        accounts = [Account.from_annual("Test", 0.08, 0.10)]
        model = ReturnModel(accounts)

        R = model.generate(T=100, n_sims=5000, seed=42)

        mean_return = R.mean()
        monthly_expected = 0.08 / 12  # Rough approximation

        # Should be within reasonable bounds
        assert abs(mean_return - monthly_expected) < 0.01

    def test_volatility_reasonable(self):
        """Test that volatility is in reasonable range."""
        accounts = [Account.from_annual("Test", 0.08, 0.10)]
        model = ReturnModel(accounts)

        R = model.generate(T=100, n_sims=5000, seed=42)

        std_return = R.std()
        monthly_expected = 0.10 / np.sqrt(12)  # Rough approximation

        # Should be within reasonable bounds
        assert abs(std_return - monthly_expected) < 0.02

    def test_lognormal_positivity(self):
        """Test that 1 + R is always positive (lognormal property)."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05),
            Account.from_annual("B", 0.14, 0.18),
        ]
        model = ReturnModel(accounts)

        R = model.generate(T=100, n_sims=1000, seed=42)
        gross_returns = 1.0 + R

        assert np.all(gross_returns > 0)


# ============================================================================
# RETURNMODEL EDGE CASES
# ============================================================================

class TestReturnModelEdgeCases:
    """Test edge cases."""

    def test_long_horizon(self):
        """Test very long time horizon."""
        accounts = [Account.from_annual("A", 0.08, 0.10)]
        model = ReturnModel(accounts)

        R = model.generate(T=360, n_sims=10, seed=42)  # 30 years

        assert R.shape == (10, 360, 1)
        assert np.all(np.isfinite(R))

    def test_many_simulations(self):
        """Test large simulation count."""
        accounts = [Account.from_annual("A", 0.08, 0.10)]
        model = ReturnModel(accounts)

        R = model.generate(T=12, n_sims=10000, seed=42)

        assert R.shape == (10000, 12, 1)
        assert np.all(np.isfinite(R))

    def test_high_volatility(self):
        """Test high volatility account."""
        accounts = [Account.from_annual("HighVol", 0.20, 0.50)]
        model = ReturnModel(accounts)

        R = model.generate(T=24, n_sims=100, seed=42)

        assert np.all(np.isfinite(R))
        assert np.all(R > -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
