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


# ============================================================================
# RETURNMODEL CORRELATION FROM GROUPS TESTS
# ============================================================================

class TestCorrelationFromGroups:
    """Tests for dict-based correlation groups in ReturnModel.__init__."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    def test_basic_pair(self, accounts):
        """Two accounts with single pair."""
        model = ReturnModel(accounts, default_correlation={
            ("Conservative", "Aggressive"): 0.5
        })
        rho = model.default_correlation

        assert rho.shape == (2, 2)
        assert rho[0, 0] == 1.0
        assert rho[1, 1] == 1.0
        assert rho[0, 1] == 0.5
        assert rho[1, 0] == 0.5  # Symmetric

    def test_group_of_three(self):
        """Group of 3 accounts sets all pairwise correlations."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.10, 0.15),
            Account.from_annual("C", 0.12, 0.20),
        ]
        model = ReturnModel(accounts, default_correlation={
            ("A", "B", "C"): 0.6  # Sets A↔B, A↔C, B↔C = 0.6
        })

        expected = np.array([
            [1.0, 0.6, 0.6],
            [0.6, 1.0, 0.6],
            [0.6, 0.6, 1.0]
        ])
        np.testing.assert_array_almost_equal(model.default_correlation, expected)

    def test_mixed_groups_and_pairs(self):
        """Combination of groups and individual pairs."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.10, 0.15),
            Account.from_annual("C", 0.12, 0.20),
            Account.from_annual("D", 0.06, 0.08),
        ]
        model = ReturnModel(accounts, default_correlation={
            ("A", "B", "C"): 0.6,  # A↔B, A↔C, B↔C = 0.6
            ("C", "D"): 0.4,       # C↔D = 0.4
            # A↔D, B↔D not specified → 0.0
        })

        expected = np.array([
            [1.0, 0.6, 0.6, 0.0],
            [0.6, 1.0, 0.6, 0.0],
            [0.6, 0.6, 1.0, 0.4],
            [0.0, 0.0, 0.4, 1.0]
        ])
        np.testing.assert_array_almost_equal(model.default_correlation, expected)

    def test_empty_dict_gives_identity(self, accounts):
        """Empty dict returns identity matrix."""
        model = ReturnModel(accounts, default_correlation={})
        np.testing.assert_array_equal(model.default_correlation, np.eye(2))

    def test_order_in_tuple_does_not_matter(self, accounts):
        """(A,B) and (B,A) produce same result."""
        model1 = ReturnModel(accounts, default_correlation={
            ("Conservative", "Aggressive"): 0.7
        })
        model2 = ReturnModel(accounts, default_correlation={
            ("Aggressive", "Conservative"): 0.7
        })
        np.testing.assert_array_equal(
            model1.default_correlation,
            model2.default_correlation
        )

    def test_invalid_account_name_raises(self, accounts):
        """Unknown account name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown account"):
            ReturnModel(accounts, default_correlation={
                ("Conservative", "Unknown"): 0.5
            })

    def test_correlation_out_of_range_raises(self, accounts):
        """Correlation outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            ReturnModel(accounts, default_correlation={
                ("Conservative", "Aggressive"): 1.5
            })

    def test_group_with_single_account_raises(self):
        """Group with only 1 account raises ValueError."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.10, 0.15),
        ]
        with pytest.raises(ValueError, match="at least 2"):
            ReturnModel(accounts, default_correlation={
                ("A",): 0.5  # Invalid: only 1 account
            })

    def test_non_psd_matrix_raises(self):
        """Correlation values that produce non-PSD matrix raise error."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.10, 0.15),
            Account.from_annual("C", 0.12, 0.20),
        ]
        with pytest.raises(ValueError, match="positive semi-definite"):
            ReturnModel(accounts, default_correlation={
                ("A", "B"): 0.9,
                ("B", "C"): 0.9,
                ("A", "C"): -0.9,
            })

    def test_backward_compatible_with_ndarray(self, accounts):
        """Existing ndarray API still works."""
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        model = ReturnModel(accounts, default_correlation=corr)
        np.testing.assert_array_equal(model.default_correlation, corr)

    def test_large_group_scalability(self):
        """Large group (5 accounts) with single correlation."""
        accounts = [Account.from_annual(f"Acc{i}", 0.08 + i*0.01, 0.10 + i*0.02)
                    for i in range(5)]
        model = ReturnModel(accounts, default_correlation={
            ("Acc0", "Acc1", "Acc2", "Acc3", "Acc4"): 0.5
        })

        # All off-diagonal correlations should be 0.5
        rho = model.default_correlation
        for i in range(5):
            for j in range(5):
                if i == j:
                    assert rho[i, j] == 1.0
                else:
                    assert rho[i, j] == 0.5

    def test_generates_correlated_returns(self):
        """Dict correlation produces actually correlated returns."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.08, 0.10),
        ]
        model = ReturnModel(accounts, default_correlation={
            ("A", "B"): 0.8
        })

        R = model.generate(T=100, n_sims=1000, seed=42)

        # Compute empirical correlation
        R_flat = R.reshape(-1, 2)
        empirical_corr = np.corrcoef(R_flat[:, 0], R_flat[:, 1])[0, 1]

        # Should be significantly positive (close to 0.8)
        assert empirical_corr > 0.6


# ============================================================================
# RETURNMODEL PROPERTY TESTS
# ============================================================================

class TestReturnModelProperties:
    """Test ReturnModel properties."""

    @pytest.fixture
    def return_model(self):
        """Create test return model."""
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return ReturnModel(accounts)

    def test_monthly_params_returns_list(self, return_model):
        """Test monthly_params returns list of dicts."""
        params = return_model.monthly_params

        assert isinstance(params, list)
        assert len(params) == 2
        assert "mu" in params[0]
        assert "sigma" in params[0]

    def test_monthly_params_reasonable_values(self, return_model):
        """Test monthly_params values are reasonable."""
        params = return_model.monthly_params

        # Conservative: 4% annual -> ~0.33% monthly
        assert 0.001 < params[0]["mu"] < 0.01
        # Aggressive: 14% annual -> ~1.1% monthly
        assert 0.005 < params[1]["mu"] < 0.02

    def test_annual_params_returns_list(self, return_model):
        """Test annual_params returns list of dicts."""
        params = return_model.annual_params

        assert isinstance(params, list)
        assert len(params) == 2
        assert "return" in params[0]
        assert "volatility" in params[0]

    def test_annual_params_match_input(self, return_model):
        """Test annual_params match original input."""
        params = return_model.annual_params

        # Conservative: 4% return, 5% volatility
        assert abs(params[0]["return"] - 0.04) < 0.001
        assert abs(params[0]["volatility"] - 0.05) < 0.001

        # Aggressive: 14% return, 15% volatility
        assert abs(params[1]["return"] - 0.14) < 0.001
        assert abs(params[1]["volatility"] - 0.15) < 0.001


# ============================================================================
# RETURNMODEL PARAMS TABLE TESTS
# ============================================================================

class TestReturnModelParamsTable:
    """Test ReturnModel.params_table() method."""

    @pytest.fixture
    def return_model(self):
        """Create test return model."""
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return ReturnModel(accounts)

    def test_params_table_returns_dataframe(self, return_model):
        """Test params_table returns DataFrame."""
        import pandas as pd

        table = return_model.params_table()

        assert isinstance(table, pd.DataFrame)

    def test_params_table_columns(self, return_model):
        """Test params_table has expected columns."""
        table = return_model.params_table()

        expected_cols = ["μ (monthly)", "μ (annual)", "σ (monthly)", "σ (annual)"]
        for col in expected_cols:
            assert col in table.columns

    def test_params_table_index(self, return_model):
        """Test params_table has account names as index."""
        table = return_model.params_table()

        assert "Conservative" in table.index
        assert "Aggressive" in table.index

    def test_params_table_row_count(self, return_model):
        """Test params_table has correct row count."""
        table = return_model.params_table()

        assert len(table) == 2


# ============================================================================
# RETURNMODEL REPR TESTS
# ============================================================================

class TestReturnModelRepr:
    """Test ReturnModel __repr__ method."""

    def test_repr_with_identity_correlation(self):
        """Test __repr__ shows 'eye' for identity correlation."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]
        model = ReturnModel(accounts)

        repr_str = repr(model)

        assert "ReturnModel" in repr_str
        assert "M=2" in repr_str
        assert "ρ=eye" in repr_str

    def test_repr_with_custom_correlation(self):
        """Test __repr__ shows 'custom' for non-identity correlation."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        model = ReturnModel(accounts, default_correlation=corr)

        repr_str = repr(model)

        assert "ρ=custom" in repr_str

    def test_repr_includes_account_info(self):
        """Test __repr__ includes account names and returns."""
        accounts = [
            Account.from_annual("MyAccount", 0.08, 0.10),
        ]
        model = ReturnModel(accounts)

        repr_str = repr(model)

        assert "MyAccount" in repr_str
        assert "8.0%/year" in repr_str


# ============================================================================
# RETURNMODEL GENERATE VALIDATION TESTS
# ============================================================================

class TestReturnModelGenerateValidation:
    """Test ReturnModel.generate() validation."""

    @pytest.fixture
    def return_model(self):
        """Create test return model."""
        accounts = [Account.from_annual("Test", 0.08, 0.10)]
        return ReturnModel(accounts)

    def test_T_must_be_positive(self, return_model):
        """Test generate raises for T <= 0."""
        from src.exceptions import ValidationError

        with pytest.raises(ValidationError, match="T must be positive"):
            return_model.generate(T=0, n_sims=100, seed=42)

    def test_T_negative_raises(self, return_model):
        """Test generate raises for negative T."""
        from src.exceptions import ValidationError

        with pytest.raises(ValidationError, match="T must be positive"):
            return_model.generate(T=-5, n_sims=100, seed=42)

    def test_n_sims_must_be_positive(self, return_model):
        """Test generate raises for n_sims <= 0."""
        with pytest.raises(ValueError, match="n_sims must be positive"):
            return_model.generate(T=12, n_sims=0, seed=42)

    def test_n_sims_negative_raises(self, return_model):
        """Test generate raises for negative n_sims."""
        with pytest.raises(ValueError, match="n_sims must be positive"):
            return_model.generate(T=12, n_sims=-10, seed=42)


# ============================================================================
# RETURNMODEL CORRELATION VALIDATION TESTS
# ============================================================================

class TestReturnModelCorrelationValidation:
    """Test correlation matrix validation."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]

    def test_asymmetric_correlation_raises(self, accounts):
        """Test asymmetric correlation matrix raises ValueError."""
        corr = np.array([
            [1.0, 0.5],
            [0.3, 1.0],  # Asymmetric: 0.5 != 0.3
        ])

        with pytest.raises(ValueError, match="symmetric"):
            ReturnModel(accounts, default_correlation=corr)

    def test_diagonal_not_one_raises(self, accounts):
        """Test correlation with diagonal != 1.0 raises ValueError."""
        corr = np.array([
            [0.9, 0.5],  # Diagonal should be 1.0
            [0.5, 1.0],
        ])

        with pytest.raises(ValueError, match="diagonal must be 1.0"):
            ReturnModel(accounts, default_correlation=corr)

    def test_correlation_override_validation(self, accounts):
        """Test correlation override is validated in generate()."""
        model = ReturnModel(accounts)

        # Asymmetric correlation in generate() should fail
        bad_corr = np.array([
            [1.0, 0.5],
            [0.3, 1.0],  # Asymmetric
        ])

        with pytest.raises(ValueError, match="symmetric"):
            model.generate(T=12, n_sims=100, correlation=bad_corr, seed=42)


# ============================================================================
# RETURNMODEL EMPTY ACCOUNTS TESTS
# ============================================================================

class TestReturnModelEmptyAccounts:
    """Test ReturnModel with empty accounts."""

    def test_empty_accounts_raises(self):
        """Test empty accounts list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            ReturnModel([])


# ============================================================================
# RETURNMODEL COVARIANCE TESTS
# ============================================================================

class TestReturnModelCovariance:
    """Test covariance matrix construction."""

    def test_build_covariance_diagonal(self):
        """Test covariance with identity correlation is diagonal."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]
        model = ReturnModel(accounts)

        cov = model._build_covariance(np.eye(2))

        # Off-diagonal should be 0
        assert abs(cov[0, 1]) < 1e-10
        assert abs(cov[1, 0]) < 1e-10

    def test_build_covariance_with_correlation(self):
        """Test covariance with non-zero correlation."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.08, 0.10),
        ]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        model = ReturnModel(accounts, default_correlation=corr)

        cov = model._build_covariance(corr)

        # Off-diagonal should be positive
        assert cov[0, 1] > 0
        assert cov[1, 0] > 0


# ============================================================================
# RETURNMODEL INTEGRATION TESTS
# ============================================================================

class TestReturnModelIntegration:
    """Integration tests for ReturnModel."""

    def test_full_workflow(self):
        """Test complete workflow from creation to generation."""
        # Create accounts
        accounts = [
            Account.from_annual("Emergency", 0.04, 0.05),
            Account.from_annual("Growth", 0.12, 0.20),
        ]

        # Create model with custom correlation
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        model = ReturnModel(accounts, default_correlation=corr)

        # Check properties
        assert model.M == 2
        assert len(model.monthly_params) == 2
        assert len(model.annual_params) == 2

        # Generate returns
        R = model.generate(T=24, n_sims=100, seed=42)

        # Validate output
        assert R.shape == (100, 24, 2)
        assert np.all(np.isfinite(R))
        assert np.all(R > -1.0)

    def test_dict_correlation_end_to_end(self):
        """Test dict-based correlation from creation to generation."""
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.10, 0.12),
            Account.from_annual("C", 0.12, 0.15),
        ]

        model = ReturnModel(accounts, default_correlation={
            ("A", "B"): 0.6,
            ("B", "C"): 0.4,
        })

        # Generate returns
        R = model.generate(T=50, n_sims=500, seed=42)

        # Validate output
        assert R.shape == (500, 50, 3)
        assert np.all(np.isfinite(R))

        # Check empirical correlation A-B is higher than A-C
        R_flat = R.reshape(-1, 3)
        corr_AB = np.corrcoef(R_flat[:, 0], R_flat[:, 1])[0, 1]
        corr_AC = np.corrcoef(R_flat[:, 0], R_flat[:, 2])[0, 1]

        assert corr_AB > corr_AC  # A-B should be more correlated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
