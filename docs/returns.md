# `returns` — Philosophy & Role in FinOpt

> **Purpose:** Generate **correlated stochastic returns** under lognormal assumptions, providing the probabilistic foundation for Monte Carlo simulation and optimization under uncertainty.  
> `returns.py` is the **stochastic engine**: it consumes account metadata (from `portfolio.py`) and produces vectorized return samples that drive wealth dynamics in `portfolio.simulate()`.

---

## Why a dedicated returns module?

Financial planning under uncertainty requires **explicit stochastic modeling**:
- `income.py` → cash flow uncertainty (seasonal + noise)
- `returns.py` → **market return uncertainty** (correlated assets)
- `portfolio.py` → wealth evolution executor
- `optimization.py` → policy search under constraints

This separation enables:
- **Loose coupling:** Portfolio never generates returns (delegated to ReturnModel)
- **Correlation modeling:** Cross-sectional dependence between accounts
- **Lognormal guarantee:** $R_t > -1$ (no catastrophic losses)
- **Dual temporal API:** Seamless monthly ↔ annual parameter conversion

---

## Mathematical framework

### Lognormal return model

Gross returns follow a **correlated lognormal distribution**:

$$
1 + R_t^m \sim \text{LogNormal}(\mu_{\text{log}}^m, \Sigma)
$$

where the covariance matrix is constructed as:

$$
\Sigma = D \rho D
$$

with:
- $D = \text{diag}(\sigma_{\text{log}}^1, \ldots, \sigma_{\text{log}}^M)$ (log-volatilities)
- $\rho \in \mathbb{R}^{M \times M}$ (correlation matrix, symmetric PSD with diagonal = 1)

### Parameter conversion

Given **arithmetic parameters** $(\mu_{\text{arith}}, \sigma_{\text{arith}})$, convert to **log-space**:

$$
\begin{aligned}
\sigma_{\text{log}} &= \sqrt{\log\left(1 + \frac{\sigma_{\text{arith}}^2}{(1 + \mu_{\text{arith}})^2}\right)} \\[8pt]
\mu_{\text{log}} &= \log(1 + \mu_{\text{arith}}) - \frac{\sigma_{\text{log}}^2}{2}
\end{aligned}
$$

**Rationale:** The adjustment $-\sigma_{\text{log}}^2/2$ ensures:
$$
\mathbb{E}[1 + R_t^m] = \exp\left(\mu_{\text{log}} + \frac{\sigma_{\text{log}}^2}{2}\right) = 1 + \mu_{\text{arith}}
$$

### Generation algorithm

1. Sample log-returns: $Z \sim \mathcal{N}(\mu_{\text{log}}, \Sigma)$ with shape $(n_{\text{sims}}, T, M)$
2. Transform to arithmetic: $R = \exp(Z) - 1$
3. **Guarantee:** $R_t^m > -1$ for all realizations (lognormal property)

**Complexity:** $O(n_{\text{sims}} \cdot T \cdot M^3)$ dominated by Cholesky decomposition of $\Sigma$ (one-time cost).

---

## Design principles

1. **Lognormal constraint**
   - Ensures $R_t > -1$ (realistic: no portfolio loses more than 100%)
   - Alternative to Normal (which allows $R_t < -1$) or Bootstrap (limited to historical support)

2. **Correlation modeling**
   - Default: uncorrelated accounts ($\rho = I$)
   - Override per `generate()` call for sensitivity analysis
   - Validation: symmetric, PSD, diagonal = 1

3. **Dual temporal representation**
   - User-facing: annual parameters (intuitive)
   - Internal: monthly log-space (canonical for sampling)
   - Properties provide views without conversion overhead

4. **No portfolio dependency**
   - Consumes `Account` metadata (loose coupling)
   - Never imports `Portfolio` (inverted dependency)
   - Testable in isolation

5. **IID assumption**
   - Returns are independent across time (no GARCH/AR)
   - Extension to time-series models is straightforward

---

## The core surface: `ReturnModel`

### Constructor

**Signature:**
```python
ReturnModel(
    accounts: List[Account],
    default_correlation: Optional[np.ndarray] = None
)
```

**Parameters:**
- `accounts`: list of `Account` objects with `return_strategy` metadata
- `default_correlation`: $M \times M$ correlation matrix (default: $I_M$)

**Initialization:**
1. Validate correlation matrix (symmetric, PSD, diagonal = 1)
2. Extract arithmetic parameters from `accounts`
3. Precompute log-space parameters $(\mu_{\text{log}}, \sigma_{\text{log}})$
4. Store for efficient sampling

**Key behaviors:**
- Raises `ValueError` if correlation invalid
- Eigenvalue check: $\lambda_{\min}(\rho) \geq -10^{-10}$ (numerical tolerance)

---

### Properties

**Dual temporal access:**

```python
@property
def monthly_params(self) -> List[Dict[str, float]]
    # [{"mu": float, "sigma": float}, ...]

@property
def annual_params(self) -> List[Dict[str, float]]
    # [{"return": float, "volatility": float}, ...]
```

**Introspection:**
```python
returns.params_table()  # DataFrame with monthly vs annual comparison
print(returns)          # Human-readable summary
```

**Example output:**
```
ReturnModel(M=2, ρ=eye, accounts=['Emergency': 4.0%/year, 'Growth': 12.0%/year])
```

---

### Core generation method

**Signature:**
```python
def generate(
    self,
    T: int,
    n_sims: int = 1,
    correlation: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray
```

**Parameters:**
- `T`: time horizon (months)
- `n_sims`: number of Monte Carlo trajectories
- `correlation`: override default correlation (sensitivity analysis)
- `seed`: RNG seed for reproducibility

**Returns:**
- `R`: shape $(n_{\text{sims}}, T, M)$ with $R_{i,t,m} > -1$ for all $(i,t,m)$

**Algorithm:**
```python
# 1. Build covariance: Σ = D @ ρ @ D
cov = diag(σ_log) @ correlation @ diag(σ_log)

# 2. Sample log-returns
rng = np.random.default_rng(seed)
Z = rng.multivariate_normal(μ_log, cov, size=(n_sims, T))  # (n_sims, T, M)

# 3. Transform to arithmetic
R = np.exp(Z) - 1.0
```

**Complexity:**
- Time: $O(M^3 + n_{\text{sims}} \cdot T \cdot M^2)$
  - $O(M^3)$: Cholesky decomposition of $\Sigma$ (one-time)
  - $O(n_{\text{sims}} \cdot T \cdot M^2)$: sampling via transform
- Memory: $O(n_{\text{sims}} \cdot T \cdot M)$

**Key behaviors:**
- Deterministic when `seed` is specified
- Validates correlation matrix on each call (allows runtime override)
- Returns empty array if $T \leq 0$: shape $(n_{\text{sims}}, 0, M)$

---

## Visualization methods

### 1) `plot()` — Distribution analysis

**Panel layout:**
1. **Top-left:** Return trajectories (Monte Carlo paths)
2. **Top-right:** Marginal histograms (monthly distribution)
3. **Bottom:** Summary statistics (monthly + annualized)

**Key features:**
- Individual trajectories at low alpha
- Mean path in bold
- Dual metrics: monthly + annualized (compounded for mean, time-scaled for std)

**Usage:**
```python
returns.plot(T=24, n_sims=500, seed=42, show_trajectories=True)
```

---

### 2) `plot_cumulative()` — Wealth evolution

**Visualization:**
- Cumulative returns: $\left(\prod_{s=0}^{t-1}(1+R_s^m)\right) - 1$
- Lateral histogram of final returns
- Percentile bands (default: 5th-95th)

**Modes:**
- **M=1:** Single plot with lateral histogram
- **M>1:** Separate subplot per account

**Theoretical validation:**
```python
# Annotation box shows:
# - Simulation mean vs theoretical: E[(1+μ)^T - 1]
# - Jensen's inequality: sample mean > theoretical (convexity)
```

**Usage:**
```python
returns.plot_cumulative(T=24, n_sims=1000, show_percentiles=True)
```

---

### 3) `plot_horizon_analysis()` — Time diversification

**Analysis across investment horizons** (default: 1, 2, 3, 5, 10, 20 years):

**Panel 1: Return vs Volatility**
- Expected return: $(1+\mu_{\text{monthly}})^{T_{\text{months}}} - 1$
- Volatility: $\sigma_{\text{monthly}} \sqrt{T_{\text{months}}}$ (time scaling)
- Signal-to-noise ratio: $\text{SNR} = \mathbb{E}[R] / \sigma[R]$

**Panel 2: Probability of Loss**
- $P(R_T < 0)$ via Normal approximation
- Risk reduction annotation (e.g., 40% → 5% over 20 years)

**Printed table:**
```
HORIZON ANALYSIS - Emergency
================================================================================
Horizon | Expected | Volatility | P(Loss) | P25-P75 |  SNR
(years) |   Return |    (±1σ)   |         |  Range  |
--------------------------------------------------------------------------------
    1.0 |     4.0% |      5.0%  |   21.2% |    6.7% |  0.80
    5.0 |    21.7% |     11.2%  |    2.6% |   15.0% |  1.94
   20.0 |   119.1% |     22.4%  |    0.0% |   30.0% |  5.32
================================================================================
```

**Usage:**
```python
returns.plot_horizon_analysis(horizons=np.array([1, 5, 10, 20]))
```

---

## Integration with FinOpt pipeline

### Workflow

```python
# 1. Define accounts (annual parameters)
accounts = [
    Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
    Account.from_annual("Housing", annual_return=0.07, annual_volatility=0.12)
]

# 2. Create return generator
returns = ReturnModel(accounts, default_correlation=np.eye(2))

# 3. Generate samples
R = returns.generate(T=24, n_sims=500, seed=42)  # (500, 24, 2)

# 4. Feed to portfolio
result = portfolio.simulate(A=A, R=R, X=X)
```

### Data flow

```
Account metadata → ReturnModel → R → Portfolio.simulate() → W
                        ↑
                   correlation
                   (user/optimizer)
```

### Optimization integration

**Chance constraint evaluation:**
```python
# Generate scenario ensemble
R_scenarios = returns.generate(T=24, n_sims=500, seed=42)

# Evaluate constraint: P(W_T^m >= b) >= 1-ε
for scenario in R_scenarios:
    W_T_m = portfolio.simulate(A, scenario, X)["wealth"][:, T, m]
    violations += (W_T_m < b).sum()

feasible = (violations / n_sims <= ε)
```

**Sensitivity analysis:**
```python
# Test correlation impact
correlations = [np.eye(2), np.array([[1, 0.5], [0.5, 1]])]

for rho in correlations:
    R = returns.generate(T=24, n_sims=500, correlation=rho)
    result = portfolio.simulate(A, R, X)
    # Compare outcomes
```

---

## Recommended usage patterns

### A) Basic generation (uncorrelated)

```python
accounts = [
    Account.from_annual("Conservative", 0.04, 0.05),
    Account.from_annual("Aggressive", 0.12, 0.20)
]
returns = ReturnModel(accounts)  # default: ρ = I

R = returns.generate(T=24, n_sims=500, seed=42)
```

### B) Correlated accounts

```python
# Positive correlation (typical for equity/bond)
rho = np.array([
    [1.00, 0.30],
    [0.30, 1.00]
])

returns = ReturnModel(accounts, default_correlation=rho)
R = returns.generate(T=24, n_sims=500, seed=42)
```

### C) Correlation override (sensitivity)

```python
# Default: ρ = 0.3
returns = ReturnModel(accounts, default_correlation=rho_low)

# Test high correlation
rho_high = np.array([[1.0, 0.8], [0.8, 1.0]])
R_high = returns.generate(T=24, n_sims=500, correlation=rho_high, seed=42)

# Compare portfolio outcomes
result_low = portfolio.simulate(A, R_low, X)
result_high = portfolio.simulate(A, R_high, X)
```

### D) Introspection and validation

```python
# Parameter table
print(returns.params_table())
#            μ (monthly)  μ (annual)  σ (monthly)  σ (annual)
# Emergency       0.0033       4.00%       0.0144       5.00%
# Growth          0.0095      12.00%       0.0577      20.00%

# Generate and validate
R = returns.generate(T=240, n_sims=1000, seed=42)

# Check lognormal property
assert np.all(R > -1.0)  # guaranteed by construction

# Check empirical moments
mu_empirical = R.mean(axis=(0,1))  # average over sims and time
mu_theoretical = returns.mean_returns  # from properties
np.testing.assert_allclose(mu_empirical, mu_theoretical, rtol=0.05)
```

### E) Horizon analysis

```python
# Understand time diversification
returns.plot_horizon_analysis(
    horizons=np.array([1, 2, 3, 5, 10, 15, 20]),
    show_table=True,
    save_path="horizon_analysis.png"
)

# Typical findings:
# - SNR increases with √T (signal grows faster than noise)
# - P(Loss) decreases exponentially with T
# - Volatility grows with √T (absolute risk increases)
```

---

## Key design decisions

### 1. Lognormal vs Normal vs Bootstrap

**Lognormal (chosen):**
- ✅ Guarantees $R_t > -1$ (realistic)
- ✅ Closed-form moments
- ✅ Positive skewness (matches empirical asset returns)
- ❌ Assumes IID (no time-series structure)

**Normal:**
- ✅ Simpler mathematics
- ❌ Allows $R_t < -1$ (unrealistic)
- ❌ Symmetric tails (doesn't match data)

**Bootstrap:**
- ✅ Matches historical distribution exactly
- ❌ Limited to observed range (no extrapolation)
- ❌ Requires historical data (not forward-looking)

**Justification:** Lognormal provides analytical tractability, realistic constraints, and forward-looking flexibility.

---

### 2. Correlation as parameter (not covariance)

**Rationale:**
- Correlation is **scale-invariant** (easier to specify)
- Natural interpretation: $\rho_{12} = 0.5$ means "moderate positive dependence"
- Covariance mixes magnitude and correlation (confusing)

**Construction:** $\Sigma = D\rho D$ separates scale (volatility) from dependence (correlation).

---

### 3. Dual temporal API matching `portfolio.py`

**Consistency:**
- `Account.from_annual()` → user specifies annual parameters
- `ReturnModel.monthly_params` → internal canonical form
- `ReturnModel.annual_params` → user-friendly view

**Conversion:**
- Geometric for returns: $(1+\mu_m)^{12} - 1$
- Time-scaling for volatility: $\sigma_m \sqrt{12}$

---

### 4. Precomputed log-parameters

**Performance:**
```python
# ❌ Bad: compute in hot loop
for sim in range(n_sims):
    sigma_log = np.sqrt(np.log(1 + sigma_arith**2 / (1 + mu_arith)**2))
    mu_log = np.log(1 + mu_arith) - 0.5 * sigma_log**2
    # ... sample

# ✅ Good: precompute once in __init__
self._mu_log = ...  # computed once
self._sigma_log = ...

for sim in range(n_sims):
    # ... sample using cached values
```

**Justification:** Conversion formulas involve `log`, `sqrt` (expensive). Amortize cost by precomputing.

---

### 5. Correlation override per `generate()` call

**Use case: Sensitivity analysis**
```python
# Sweep correlation from 0 to 0.9
for corr_val in np.linspace(0, 0.9, 10):
    rho = np.array([[1, corr_val], [corr_val, 1]])
    R = returns.generate(T=24, n_sims=500, correlation=rho, seed=42)
    # ... evaluate portfolio performance
```

**Alternative (rejected):** Creating new `ReturnModel` instances is expensive (rebuilds everything).

---

## Implementation notes

### Parameter conversion formulas

**Arithmetic → Log-space:**
```python
sigma_log = np.sqrt(np.log(1 + sigma_arith**2 / (1 + mu_arith)**2))
mu_log = np.log(1 + mu_arith) - 0.5 * sigma_log**2
```

**Log-space → Arithmetic (verification):**
```python
# Expected gross return
E_gross = np.exp(mu_log + 0.5 * sigma_log**2)
mu_arith_recovered = E_gross - 1

# Volatility (more complex)
var_gross = (np.exp(sigma_log**2) - 1) * np.exp(2*mu_log + sigma_log**2)
sigma_arith_recovered = np.sqrt(var_gross)
```

**Round-trip test:**
```python
assert np.isclose(mu_arith_recovered, mu_arith)
assert np.isclose(sigma_arith_recovered, sigma_arith)
```

---

### Correlation matrix validation

**Three checks:**
1. **Symmetry:** $\rho = \rho^T$
2. **Diagonal:** $\rho_{ii} = 1$ for all $i$
3. **Positive semi-definite:** $\lambda_{\min}(\rho) \geq 0$

**Implementation:**
```python
# PSD check via eigenvalues (numerically stable)
eigvals = np.linalg.eigvalsh(rho)  # sorted ascending
if np.any(eigvals < -1e-10):  # numerical tolerance
    raise ValueError(f"Not PSD: λ_min = {eigvals.min():.6f}")
```

**Alternative (rejected):** Cholesky decomposition fails silently for near-PSD matrices.

---

### Sampling via multivariate normal

**NumPy implementation:**
```python
rng = np.random.default_rng(seed)
Z = rng.multivariate_normal(mean=μ_log, cov=Σ, size=(n_sims, T))
```

**Internal algorithm:**
1. Cholesky: $\Sigma = LL^T$ where $L$ is lower triangular ($O(M^3)$)
2. Sample: $Z_i \sim \mathcal{N}(0, I_M)$ ($O(n_{\text{sims}} \cdot T \cdot M)$)
3. Transform: $X_i = \mu + LZ_i$ ($O(n_{\text{sims}} \cdot T \cdot M^2)$)

**Bottleneck:** Cholesky is $O(M^3)$ but computed once per `generate()` call.

---

### Numerical stability

**Lognormal sampling:**
- Samples in log-space (stable)
- Exponentiation: $R = \exp(Z) - 1$ (no underflow for $Z \ll 0$)
- No division (gradients flow cleanly)

**Edge case: $\sigma_{\text{log}} = 0$ (deterministic)**
```python
# Covariance becomes singular but multivariate_normal handles it
cov = np.zeros((M, M))  # degenerate
Z = rng.multivariate_normal(μ_log, cov, size=(n_sims, T))
# Result: Z[:, :, m] = μ_log[m] for all realizations (constant)
```

---

## Mathematical results

**Proposition 1 (Lognormal Moments):**  
If $1 + R \sim \text{LogNormal}(\mu_{\log}, \sigma_{\log}^2)$, then:
$$
\begin{aligned}
\mathbb{E}[R] &= \exp\left(\mu_{\log} + \frac{\sigma_{\log}^2}{2}\right) - 1 \\[6pt]
\text{Var}[R] &= \left(\exp(\sigma_{\log}^2) - 1\right) \exp(2\mu_{\log} + \sigma_{\log}^2)
\end{aligned}
$$

**Proof:** Standard lognormal distribution formulas. ∎

---

**Proposition 2 (Return Bound):**  
For lognormal returns, $R_t^m > -1$ almost surely.

**Proof:** $1 + R_t^m = \exp(Z_t^m)$ where $Z_t^m \in \mathbb{R}$. Since $\exp(z) > 0$ for all $z \in \mathbb{R}$, we have $1 + R_t^m > 0$, thus $R_t^m > -1$. ∎

---

**Proposition 3 (Correlation Preservation):**  
If $(Z_1, Z_2)$ are bivariate normal with correlation $\rho$, then $\text{Corr}(\exp(Z_1), \exp(Z_2))$ is a monotone increasing function of $\rho$.

**Consequence:** Higher correlation in log-space → higher correlation in gross returns.

**Proof:** Follows from monotonicity of $\exp$ and Gaussian copula properties. ∎

---

**Proposition 4 (Time Diversification):**  
For IID returns with $\mu > 0$, the probability of loss decreases exponentially:
$$
P\left(\prod_{t=1}^T (1+R_t) < 1\right) \approx \Phi\left(-\frac{\mu \sqrt{T}}{\sigma}\right) \xrightarrow{T \to \infty} 0
$$

where $\Phi$ is the standard normal CDF.

**Proof:** Central Limit Theorem applied to log-returns. ∎

---

## Extensions: Temporal Dependence in Returns

### Motivation

The current IID assumption ($R_t^m \perp R_s^m$ for $t \neq s$) is appropriate for monthly horizons and typical investment periods ($T = 24-60$ months), where empirical autocorrelation is weak ($|\rho_1| < 0.1$). However, temporal structure becomes relevant for:

1. **Long horizons:** $T > 120$ months where autocorrelation accumulates
2. **Momentum strategies:** Assets with persistent trends ($\phi > 0.2$)
3. **Volatility clustering:** Crisis periods with persistent high volatility
4. **Mean reversion:** Fixed-income or commodity markets with negative autocorrelation

---

### Mathematical Framework

#### AR(1) Process

$$
R_t^m = \phi^m R_{t-1}^m + \epsilon_t^m, \quad \epsilon_t^m \sim \text{LogNormal}(\mu_\epsilon, \sigma_\epsilon)
$$

**Key insight:** Preserves **affine wealth representation** because returns remain exogenous:

$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$

where $F_{s,t}^m = \prod_{r=s}^{t-1}(1+R_r^m)$ now depends on past returns but **not on policy $X$**.

**Properties:**
- Gradient: $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ (unchanged)
- Convexity: optimization problem remains convex
- Variance scaling: $\text{Var}[\sum_{t=1}^T R_t] \approx T\sigma^2 \cdot \frac{1+\phi}{1-\phi}$

**Implementation sketch:**
```python
def _generate_ar1(self, T, n_sims, phi, seed):
    epsilon = self._generate_iid(T, n_sims, seed)  # IID innovations
    R = np.zeros((n_sims, T, self.M))
    R[:, 0, :] = epsilon[:, 0, :]
    
    for t in range(1, T):
        R[:, t, :] = phi * R[:, t-1, :] + epsilon[:, t, :]
    return R
```

**Complexity:** $O(n_{\text{sims}} \cdot T \cdot M)$ vs $O(n_{\text{sims}} \cdot M^3)$ for IID (sequential vs parallel generation).

---

#### GARCH(1,1) Process

$$
\begin{aligned}
R_t^m &= \sigma_t^m Z_t^m, \quad Z_t^m \sim \text{LogNormal}(\mu, 1) \\
(\sigma_t^m)^2 &= \omega^m + \alpha^m (R_{t-1}^m)^2 + \beta^m (\sigma_{t-1}^m)^2
\end{aligned}
$$

**Purpose:** Captures volatility clustering without affecting expected returns.

**Constraints:** $\omega > 0$, $\alpha, \beta \geq 0$, $\alpha + \beta < 1$ (stationarity).

**Affine property:** Preserved (returns exogenous to policy).

**Parameter estimation:** Requires $n > 200$ monthly observations; MLE is non-convex.

---

### Implementation Strategy

**Extensible API:**
```python
class ReturnModel:
    def __init__(
        self, 
        accounts: List[Account],
        default_correlation: Optional[np.ndarray] = None,
        temporal_model: Literal["iid", "ar1", "garch"] = "iid",
        temporal_params: Optional[Dict] = None  # {"phi": [...], "omega": [...], ...}
    ):
        self.temporal_model = temporal_model
        self.temporal_params = temporal_params or {}
        # ... existing init
    
    def generate(self, T, n_sims, correlation=None, seed=None):
        if self.temporal_model == "iid":
            return self._generate_iid(T, n_sims, correlation, seed)
        elif self.temporal_model == "ar1":
            phi = self.temporal_params.get("phi", np.zeros(self.M))
            return self._generate_ar1(T, n_sims, phi, correlation, seed)
        elif self.temporal_model == "garch":
            return self._generate_garch(T, n_sims, correlation, seed, **self.temporal_params)
```

**Backward compatibility:** Default `temporal_model="iid"` preserves existing behavior.

---

### When to Implement

**Signals that temporal structure matters:**

1. **Backtesting shows systematic bias:** IID predictions consistently over/underestimate risk
2. **Long-horizon planning:** $T > 120$ months where autocorrelation compounds
3. **Asset-specific evidence:** Ljung-Box test rejects IID at 5% significance
4. **Volatility analysis:** ARCH test detects conditional heteroskedasticity

**Estimation requirements:**
- AR(1): minimum $n = 60$ monthly returns per account
- GARCH(1,1): minimum $n = 120$ monthly or $n = 500$ daily returns

---

### Trade-offs

| Aspect | IID | AR(1) | GARCH(1,1) |
|--------|-----|-------|------------|
| Affine structure | ✓ | ✓ | ✓ |
| Computational cost | 1x | 5-10x | 10-20x |
| Parameters per account | 2 | 3 | 4 |
| Closed-form moments | ✓ | ✗ | ✗ |
| Empirical necessity (monthly) | ✓ | ∼ | ∼ |

**Recommendation:** Implement AR(1) first if empirical autocorrelation $|\phi| > 0.15$. GARCH only if volatility clustering dominates (e.g., crisis modeling).

---

### References for Implementation

- **AR estimation:** Yule-Walker equations via `statsmodels.tsa.ar_model.AutoReg`
- **GARCH estimation:** `arch` package (Engle 2001 implementation)
- **Model selection:** AIC/BIC comparison, Ljung-Box test for residuals
- **Validation:** Out-of-sample log-likelihood, forecast MSE

---

**Prompt for resuming work:**  
"Implement temporal dependence in `ReturnModel` using AR(1) process. Preserve affine wealth representation and convex optimization structure. Validate that $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ holds. Benchmark computational overhead vs IID baseline for $T=24, n_{\text{sims}}=500$."
---