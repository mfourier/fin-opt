# `returns` — Stochastic Return Generation for FinOpt

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
- **Flexible correlation:** Matrix or group-based specification

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
   - Supports both matrix and group-based specification
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
   - Extension to time-series models is straightforward (see Extensions section)

---

## The core surface: `ReturnModel`

### Constructor

**Signature:**
```python
ReturnModel(
    accounts: List[Account],
    default_correlation: Optional[Union[np.ndarray, Dict[Tuple[str, ...], float]]] = None
)
```

**Parameters:**
- `accounts`: list of `Account` objects with `return_strategy` metadata
- `default_correlation`: Correlation specification (see below)

**Correlation specification options:**

1. **None (default):** Identity matrix $\rho = I_M$ (uncorrelated)

2. **Matrix (np.ndarray):** Full $M \times M$ correlation matrix
   ```python
   rho = np.array([[1.0, 0.5], [0.5, 1.0]])
   returns = ReturnModel(accounts, default_correlation=rho)
   ```

3. **Groups (Dict):** Account groups with shared correlation
   ```python
   groups = {
       ("Emergency", "Housing"): 0.3,        # Pair correlation
       ("Stock1", "Stock2", "Stock3"): 0.6,  # Group correlation (all pairs)
   }
   returns = ReturnModel(accounts, default_correlation=groups)
   ```

**Group semantics:**
- Tuple of 2 accounts: Single pair correlation
- Tuple of 3+ accounts: All pairwise combinations get the same correlation
- Unspecified pairs default to 0.0
- Diagonal always 1.0

**Initialization:**
1. Validate correlation matrix (symmetric, PSD, diagonal = 1)
2. Extract arithmetic parameters from `accounts`
3. Precompute log-space parameters $(\mu_{\text{log}}, \sigma_{\text{log}})$

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

**Legacy properties (backward compatible):**
```python
@property
def mean_returns(self) -> np.ndarray   # Expected arithmetic returns (monthly)

@property
def volatilities(self) -> np.ndarray   # Arithmetic volatilities (monthly)

@property
def account_names(self) -> List[str]   # Account names
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

**Raises:**
- `ValidationError`: If $T \leq 0$
- `ValueError`: If correlation matrix is invalid

**Algorithm:**
```python
# 1. Use correlation override or default
rho = correlation if correlation is not None else self.default_correlation

# 2. Build Σ = D @ ρ @ D
cov = diag(σ_log) @ rho @ diag(σ_log)

# 3. Sample log-returns
rng = np.random.default_rng(seed)
Z = rng.multivariate_normal(μ_log, cov, size=(n_sims, T))  # (n_sims, T, M)

# 4. Transform to arithmetic
R = np.exp(Z) - 1.0
```

---

## Visualization methods

### 1) `plot()` — Distribution analysis

**Signature:**
```python
def plot(
    self,
    T: int = 32,
    n_sims: int = 300,
    correlation: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    start: Optional[date] = None,  # Calendar-aware x-axis
    figsize: tuple = (16, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    return_fig_ax: bool = False,
    show_trajectories: bool = True,
    trajectory_alpha: float = 0.05
)
```

**Panel layout:**
1. **Top-left:** Return trajectories (Monte Carlo paths)
2. **Top-right:** Marginal histograms (monthly distribution)
3. **Bottom:** Summary statistics table (monthly + annualized)

**Calendar-aware plotting:**
When `start` is provided, the x-axis shows calendar dates instead of numeric month indices:
```python
from datetime import date
returns.plot(T=24, n_sims=500, seed=42, start=date(2025, 1, 1))
```

---

### 2) `plot_cumulative()` — Wealth evolution

**Signature:**
```python
def plot_cumulative(
    self,
    T: int = 24,
    n_sims: int = 1000,
    correlation: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    start: Optional[date] = None,  # Calendar-aware x-axis
    figsize: tuple = (14, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    return_fig_ax: bool = False,
    show_trajectories: bool = True,
    trajectory_alpha: float = 0.08,
    show_percentiles: bool = True,
    percentiles: tuple = (5, 95),
    hist_bins: int = 40,
    hist_color: str = 'red'
)
```

**Visualization:**
- Cumulative returns: $\left(\prod_{s=0}^{t-1}(1+R_s^m)\right) - 1$
- Lateral histogram of final returns
- Percentile bands (default: 5th-95th)

**Modes:**
- **M=1:** Single plot with lateral histogram
- **M>1:** Separate subplot per account

**Theoretical validation (annotation box):**
- Simulation mean vs theoretical: $\mathbb{E}[(1+\mu)^T - 1]$
- Jensen's inequality: sample mean > theoretical (convexity)

---

### 3) `plot_horizon_analysis()` — Time diversification

**Signature:**
```python
def plot_horizon_analysis(
    self,
    horizons: np.ndarray = np.array([1, 2, 3, 5, 10, 15, 20]),
    figsize: tuple = (15, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    return_fig_ax: bool = False,
    show_table: bool = True
)
```

**Analysis across investment horizons** (default: 1, 2, 3, 5, 10, 15, 20 years):

**Panel 1: Return vs Volatility**
- Expected return: $(1+\mu_{\text{monthly}})^{T_{\text{months}}} - 1$
- Volatility: $\sigma_{\text{monthly}} \sqrt{T_{\text{months}}}$ (time scaling)
- Signal-to-noise ratio: $\text{SNR} = \mathbb{E}[R] / \sigma[R]$

**Panel 2: Probability of Loss**
- $P(R_T < 0)$ via Normal approximation
- Risk reduction annotation

**Printed table (when `show_table=True`):**
```
HORIZON ANALYSIS - Emergency
================================================================================
Horizon (years) | Expected Return (%) | Volatility (%) | P(Loss) (%) | SNR
--------------------------------------------------------------------------------
            1.0 |                 4.0 |            5.0 |        21.2 | 0.80
            5.0 |                21.7 |           11.2 |         2.6 | 1.94
           20.0 |               119.1 |           22.4 |         0.0 | 5.32
================================================================================
```

---

## Integration with FinOpt pipeline

### Workflow

```python
from datetime import date

# 1. Define accounts (annual parameters + display names)
accounts = [
    Account.from_annual("EM", annual_return=0.04, annual_volatility=0.05,
                        display_name="Emergency Fund"),
    Account.from_annual("HS", annual_return=0.07, annual_volatility=0.12,
                        display_name="Housing Savings")
]

# 2. Create return generator with group correlation
groups = {("EM", "HS"): 0.3}
returns = ReturnModel(accounts, default_correlation=groups)

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
                   (matrix or groups)
```

### Optimization integration

**Chance constraint evaluation:**
```python
# Generate scenario ensemble
R_scenarios = returns.generate(T=24, n_sims=500, seed=42)

# Evaluate constraint: P(W_T^m >= b) >= 1-ε
W = portfolio.simulate(A, R_scenarios, X)["wealth"]
W_T_m = W[:, -1, m]
violation_rate = (W_T_m < b).mean()
feasible = (violation_rate <= epsilon)
```

**Sensitivity analysis:**
```python
# Test correlation impact
correlations = [np.eye(2), np.array([[1, 0.5], [0.5, 1]])]

for rho in correlations:
    R = returns.generate(T=24, n_sims=500, correlation=rho, seed=42)
    result = portfolio.simulate(A, R, X)
    print(f"ρ={rho[0,1]}: mean W_T = {result['total_wealth'][:, -1].mean():,.0f}")
```

---

## Usage patterns

### A) Basic generation (uncorrelated)

```python
accounts = [
    Account.from_annual("Conservative", 0.04, 0.05),
    Account.from_annual("Aggressive", 0.12, 0.20)
]
returns = ReturnModel(accounts)  # default: ρ = I

R = returns.generate(T=24, n_sims=500, seed=42)
```

### B) Correlated accounts (matrix)

```python
rho = np.array([
    [1.00, 0.30],
    [0.30, 1.00]
])

returns = ReturnModel(accounts, default_correlation=rho)
R = returns.generate(T=24, n_sims=500, seed=42)
```

### C) Correlated accounts (groups)

```python
# Define accounts
accounts = [
    Account.from_annual("Stock1", 0.10, 0.18),
    Account.from_annual("Stock2", 0.12, 0.20),
    Account.from_annual("Stock3", 0.08, 0.15),
    Account.from_annual("Bond", 0.04, 0.05)
]

# Stocks are correlated with each other, bonds uncorrelated
groups = {
    ("Stock1", "Stock2", "Stock3"): 0.6,  # All stock pairs get 0.6
    # Bond not mentioned → 0 correlation with everything
}

returns = ReturnModel(accounts, default_correlation=groups)
```

### D) Correlation override (sensitivity)

```python
# Default: ρ = 0.3
returns = ReturnModel(accounts, default_correlation=rho_low)

# Test high correlation without recreating model
rho_high = np.array([[1.0, 0.8], [0.8, 1.0]])
R_high = returns.generate(T=24, n_sims=500, correlation=rho_high, seed=42)
```

### E) Introspection and validation

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
mu_empirical = R.mean(axis=(0, 1))  # average over sims and time
mu_theoretical = returns.mean_returns  # from properties
np.testing.assert_allclose(mu_empirical, mu_theoretical, rtol=0.05)
```

### F) Calendar-aware visualization

```python
from datetime import date

# Plot with calendar dates on x-axis
returns.plot(T=24, n_sims=500, seed=42, start=date(2025, 1, 1))
returns.plot_cumulative(T=36, n_sims=500, seed=42, start=date(2025, 1, 1))
```

---

## Key design decisions

### 1. Lognormal vs Normal vs Bootstrap

| Method | Pros | Cons |
|--------|------|------|
| **Lognormal** (chosen) | Guarantees $R_t > -1$, closed-form moments, positive skewness | Assumes IID |
| Normal | Simpler mathematics | Allows $R_t < -1$ (unrealistic) |
| Bootstrap | Matches historical distribution | Limited to observed range |

### 2. Correlation as parameter (not covariance)

**Rationale:**
- Correlation is **scale-invariant** (easier to specify)
- Natural interpretation: $\rho_{12} = 0.5$ means "moderate positive dependence"
- Covariance mixes magnitude and correlation (confusing)

**Construction:** $\Sigma = D\rho D$ separates scale (volatility) from dependence (correlation).

### 3. Group-based correlation

**Rationale:**
- Financial intuition: "these three stocks are correlated"
- Easier than building full matrix for many accounts
- Automatic validation and symmetry

### 4. Precomputed log-parameters

**Performance:** Conversion formulas involve `log`, `sqrt` (expensive). Precompute once in `__init__`.

### 5. Correlation override per `generate()` call

**Use case:** Sensitivity analysis without recreating the model.

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

**Proposition 2 (Return Bound):**
For lognormal returns, $R_t^m > -1$ almost surely.

**Proof:** $1 + R_t^m = \exp(Z_t^m)$ where $Z_t^m \in \mathbb{R}$. Since $\exp(z) > 0$ for all $z \in \mathbb{R}$, we have $R_t^m > -1$. ∎

**Proposition 3 (Correlation Preservation):**
If $(Z_1, Z_2)$ are bivariate normal with correlation $\rho$, then $\text{Corr}(\exp(Z_1), \exp(Z_2))$ is a monotone increasing function of $\rho$.

**Proposition 4 (Time Diversification):**
For IID returns with $\mu > 0$, the probability of loss decreases exponentially:
$$
P\left(\prod_{t=1}^T (1+R_t) < 1\right) \approx \Phi\left(-\frac{\mu \sqrt{T}}{\sigma}\right) \xrightarrow{T \to \infty} 0
$$

---

## Extensions: Temporal Dependence (Roadmap)

> **Note:** This section describes potential future extensions. The current implementation only supports IID returns.

### Motivation

The current IID assumption ($R_t^m \perp R_s^m$ for $t \neq s$) is appropriate for monthly horizons and typical investment periods ($T = 24-60$ months), where empirical autocorrelation is weak ($|\rho_1| < 0.1$). However, temporal structure becomes relevant for:

1. **Long horizons:** $T > 120$ months where autocorrelation accumulates
2. **Momentum strategies:** Assets with persistent trends ($\phi > 0.2$)
3. **Volatility clustering:** Crisis periods with persistent high volatility

### Proposed AR(1) Extension

$$
R_t^m = \phi^m R_{t-1}^m + \epsilon_t^m, \quad \epsilon_t^m \sim \text{LogNormal}(\mu_\epsilon, \sigma_\epsilon)
$$

**Key insight:** Preserves **affine wealth representation** because returns remain exogenous (not dependent on policy $X$).

### Proposed API

```python
class ReturnModel:
    def __init__(
        self,
        accounts: List[Account],
        default_correlation: Optional[np.ndarray] = None,
        temporal_model: Literal["iid", "ar1", "garch"] = "iid",  # Future
        temporal_params: Optional[Dict] = None  # {"phi": [...], ...}
    ):
        ...
```

**Backward compatibility:** Default `temporal_model="iid"` preserves existing behavior.

### When to Implement

**Signals that temporal structure matters:**
1. Backtesting shows systematic bias
2. Long-horizon planning ($T > 120$ months)
3. Ljung-Box test rejects IID at 5% significance

---

## Exceptions

The module raises `ValidationError` for invalid parameters:

```python
from finopt.src.exceptions import ValidationError

try:
    R = returns.generate(T=-1, n_sims=500)
except ValidationError as e:
    print(f"Invalid parameters: {e}")
```
