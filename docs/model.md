```markdown
# `model` — Philosophy & Role in FinOpt

> **Purpose:** Unified **orchestrator** for Monte Carlo simulation, integrating income generation, return modeling, and portfolio dynamics into a single coherent interface with intelligent caching, reproducibility guarantees, and optimization-ready outputs.  
> `model.py` is the **facade layer**: while `income.py` generates cash flows, `returns.py` produces stochastic returns, and `portfolio.py` executes wealth dynamics, `model.py` coordinates the entire pipeline and packages results for analysis and optimization.

---

## Why a dedicated model module?

Financial planning requires **end-to-end simulation** with multiple moving parts. `model.py` provides:

- **Single entry point:** Unified `FinancialModel` class orchestrates all components
- **Intelligent caching:** Parameter-based memoization avoids redundant computation
- **Reproducibility:** Explicit seed management with automatic propagation
- **Rich analytics:** `SimulationResult` container with financial metrics computation
- **Seamless visualization:** Auto-simulation in `plot()` methods
- **Optimization integration:** Affine wealth representation for gradient-based solvers

---

## Design principles

1. **Facade pattern:** Coordinates but doesn't re-implement (loose coupling)
2. **Explicit reproducibility:** Seed propagation ensures statistical independence (income uses `seed`, returns use `seed+1`)
3. **Type-safe results:** `SimulationResult` as explicit dataclass (not `dict`)
4. **Zero-overhead visualization:** `plot()` auto-simulates with caching
5. **Optimization-ready:** Default `method="affine"` exposes gradients $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$

---

## Core components

### 1) `SimulationResult`

Immutable container for complete Monte Carlo simulation output with lazy-computed analytics.

**Attributes:**
```python
@dataclass(frozen=False)
class SimulationResult:
    # Primary outputs
    wealth: np.ndarray              # (n_sims, T+1, M)
    total_wealth: np.ndarray        # (n_sims, T+1)
    contributions: np.ndarray       # (n_sims, T) or (T,)
    returns: np.ndarray             # (n_sims, T, M)
    income: dict                    # {"fixed", "variable", "total"}
    allocation: np.ndarray          # (T, M)
    
    # Metadata for reproducibility
    T, n_sims, M: int
    start: date
    seed: Optional[int]
    account_names: List[str]
```

**Analytical methods:**

#### `metrics(account=None) → pd.DataFrame`

Computes **per-simulation** financial metrics:

$$
\begin{aligned}
\text{CAGR}_i &= \left(\frac{W_{T,i}}{W_{0,i}}\right)^{12/T} - 1 \\
\text{Sharpe}_i &= \frac{\bar{R}_i}{\sigma_i} \quad \text{(assumes } r_f = 0\text{)} \\
\text{Sortino}_i &= \frac{\bar{R}_i}{\sigma^-_i}, \quad \sigma^-_i = \sqrt{\mathbb{E}[\min(R_t, 0)^2]} \\
\text{MDD}_i &= \min_{t} \frac{W_{t,i} - \max_{s \leq t} W_{s,i}}{\max_{s \leq t} W_{s,i}}
\end{aligned}
$$

**Returns:** DataFrame with columns `['cagr', 'volatility', 'sharpe', 'sortino', 'max_drawdown']`

#### `aggregate_metrics(account=None) → pd.Series | pd.DataFrame`

Computes **distribution-level** risk metrics:

$$
\begin{aligned}
\text{VaR}_{0.95}(W_T) &= F_{W_T}^{-1}(0.05) \quad \text{(5th percentile)} \\
\text{CVaR}_{0.95}(W_T) &= \mathbb{E}[W_T \mid W_T \leq \text{VaR}_{0.95}]
\end{aligned}
$$

Plus summary statistics: mean, median, std, min, max of $W_T$.

**Use in optimization:**
```python
# Chance constraint: P(W_T^m >= b) >= 1-ε via CVaR
cvar = result.aggregate_metrics(account="Housing")['cvar_95']
if cvar >= goal:
    # Feasible with α=0.95 confidence
```

#### `summary(confidence=0.95) → pd.DataFrame`

Statistical summary with confidence intervals:
$$
\text{CI}_{1-\alpha}(W_T) = \left[F^{-1}_{W_T}(\alpha/2), F^{-1}_{W_T}(1-\alpha/2)\right]
$$

#### `convergence_analysis() → pd.DataFrame`

Monte Carlo convergence diagnostics via standard error: $\text{SE}(n) = \frac{\sigma_{W_T}}{\sqrt{n}}$

---

### 2) `FinancialModel`

Unified orchestrator coordinating the flow: `income → contributions (A) → returns (R) → wealth (W)`

**Constructor:**
```python
FinancialModel(
    income: IncomeModel,
    accounts: List[Account],
    default_correlation: Optional[np.ndarray] = None,
    enable_cache: bool = True
)
```

**Internal components:**
```python
self.returns = ReturnModel(accounts, default_correlation)
self.portfolio = Portfolio(accounts)
```

---

## Simulation workflow

### `simulate()` method

```python
def simulate(
    T: int,
    X: np.ndarray,                    # (T, M)
    n_sims: int = 1,
    start: Optional[date] = None,
    seed: Optional[int] = None,
    use_cache: bool = True
) -> SimulationResult
```

**Pipeline execution:**

1. **Cache lookup:** SHA256 hash of `(T, X.tobytes(), n_sims, start, seed)`
2. **Seed propagation:**
   ```python
   A = income.contributions(T, start, seed=seed, n_sims=n_sims)
   R = returns.generate(T, n_sims, seed=None if seed is None else seed+1)
   ```
3. **Wealth dynamics:**
   ```python
   portfolio_result = portfolio.simulate(A, R, X, method="affine")
   ```
   Uses affine representation:
   $$
   W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
   $$
4. **Result packaging:** Wrap arrays in `SimulationResult` dataclass

**Complexity:** $O(n_{\text{sims}} \cdot T \cdot M)$ time, same space

**Example:**
```python
X = np.tile([0.7, 0.3], (24, 1))
result = model.simulate(T=24, X=X, n_sims=1000, seed=42)

# Second call: O(1) (cached)
result2 = model.simulate(T=24, X=X, n_sims=1000, seed=42)
assert result is result2  # same object
```

---

## Unified plotting interface

### `plot()` method

```python
def plot(
    mode: str,
    *,
    T: Optional[int] = None,
    X: Optional[np.ndarray] = None,
    n_sims: int = 500,
    start: Optional[date] = None,
    seed: Optional[int] = None,
    result: Optional[SimulationResult] = None,
    **kwargs
)
```

**Available modes:**

| Mode | Requires Simulation | Description |
|------|---------------------|-------------|
| `"income"` | No | Fixed + variable + total income streams |
| `"contributions"` | No | Monthly contribution schedule |
| `"returns"` | No | Return distributions and trajectories |
| `"returns_cumulative"` | No | Cumulative return evolution |
| `"returns_horizon"` | No | Risk-return by investment horizon |
| `"wealth"` | Yes | Portfolio dynamics (4 panels) |
| `"comparison"` | Yes | Multi-strategy comparison |

**Auto-simulation logic:**
```python
IF mode in {"wealth", "comparison"}:
    IF result provided:
        use result
    ELSE:
        result = self.simulate(T, X, n_sims, start, seed)  # cached
```

**Examples:**
```python
# Direct plotting (auto-simulates + caches)
model.plot("wealth", T=24, X=X, n_sims=500, seed=42, start=date(2025, 1, 1))

# Reuse result across plots
result = model.simulate(T=24, X=X, n_sims=500, seed=42)
model.plot("wealth", result=result, show_trajectories=True)
model.plot("wealth", result=result, show_trajectories=False)

# Strategy comparison
results = {
    "Conservative": model.simulate(T=24, X=X_cons, n_sims=500),
    "Aggressive": model.simulate(T=24, X=X_agg, n_sims=500)
}
model.plot("comparison", results=results)
```

---

## Integration with optimization

### Affine wealth for convex problems

`simulate()` uses `method="affine"` internally, enabling:

1. **Linear constraints:** $W_t^m(X) \geq b_t^m$ is affine in $X$
2. **Analytical gradients:** $\nabla_{x_s^m} W_t^m(X) = A_s F_{s,t}^m$
3. **SAA for chance constraints:**
   $$
   \mathbb{P}(W_t^m \geq b) \geq 1-\varepsilon \quad \Rightarrow \quad \frac{1}{n_{\text{sims}}} \sum_{i=1}^{n_{\text{sims}}} \mathbb{1}_{W_t^{m,(i)} \geq b} \geq 1-\varepsilon
   $$

### Goal feasibility check

```python
# Goal: $12M in Housing at T=24 with 95% confidence
result = model.simulate(T=24, X=X_uniform, n_sims=1000, seed=42)
cvar = result.aggregate_metrics(account="Housing")['cvar_95']

if cvar >= 12_000_000:
    print(f"Feasible! CVaR₉₅ = ${cvar:,.0f}")
else:
    print(f"Deficit: ${12_000_000 - cvar:,.0f}")
```

---

## Cache management

### Inspection

```python
info = model.cache_info()
# {'size': 3, 'memory_mb': 28.7}
```

**Memory estimate:** $\approx n_{\text{sims}} \cdot T \cdot M \cdot 24$ bytes (wealth + returns + contributions)

### Trade-offs

**With cache (default):**
- ✅ Instant repeated calls
- ❌ RAM scales with parameter space

**Without cache:**
```python
model = FinancialModel(income, accounts, enable_cache=False)
```
- ✅ Minimal memory
- ❌ Re-simulation on every call

**Rule:** Enable cache if `n_params × n_sims × T × M × 24 < 0.5 × RAM`.

---

## Usage patterns

### A) Basic simulation with reproducibility

```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.portfolio import Account
from finopt.src.model import FinancialModel

# Setup
income = IncomeModel(
    fixed=FixedIncome(base=1_500_000, annual_growth=0.04),
    variable=VariableIncome(base=300_000, sigma=0.15, seed=100)
)

accounts = [
    Account.from_annual("Emergency", annual_return=0.035, annual_volatility=0.06),
    Account.from_annual("Housing", annual_return=0.08, annual_volatility=0.15)
]

model = FinancialModel(income, accounts)

# Simulate
X = np.tile([0.7, 0.3], (36, 1))
result = model.simulate(T=36, X=X, n_sims=2000, seed=42, start=date(2025, 1, 1))
```

### B) Statistical analysis

```python
# Summary statistics
print(result.summary(confidence=0.95))

# Per-simulation metrics
metrics = result.metrics(account="Emergency")
print(f"Sharpe: {metrics['sharpe'].mean():.3f} ± {metrics['sharpe'].std():.3f}")

# Distribution-level risk
agg = result.aggregate_metrics()
print(f"VaR₉₅: ${agg.loc['Housing', 'var_95']:,.0f}")
```

### C) Visualization

```python
# Pre-simulation plots
model.plot("income", months=24, start=date(2025, 1, 1))
model.plot("returns_cumulative", T=120, n_sims=500, start=date(2025, 1, 1))

# Simulation-based (auto-simulates + caches)
model.plot("wealth", T=24, X=X, n_sims=500, seed=42, 
           start=date(2025, 1, 1), show_trajectories=True)
```

### D) Time-varying allocation (glide path)

```python
T = 60
equity_fractions = np.linspace(0.80, 0.40, T)
X_glide = np.column_stack([equity_fractions, 1 - equity_fractions])

result = model.simulate(T=T, X=X_glide, n_sims=500, seed=42)
model.plot("wealth", result=result, title="Glide Path Strategy")
```

---

## Key design decisions

### 1. Seed propagation for independence

Income uses `seed`, returns use `seed+1` to avoid artificial coupling while maintaining reproducibility.

### 2. SHA256 hashing for cache keys

Deterministic, collision-resistant ($P(\text{collision}) \approx 2^{-256}$), fast ($O(T \cdot M)$).

### 3. SimulationResult as dataclass

Type safety, immutability enforcement, method encapsulation vs. dict with magic keys.

### 4. Default `method="affine"`

Optimization-ready (exposes gradients), acceptable memory for $T \leq 100$.

### 5. Calendar propagation

`start` flows through: `user → simulate(start) → result.start → plot(result) → x-axis dates`

---

## Mathematical results

**Proposition 1 (Affine Wealth):**  
For any allocation policy $X$ and return realization $\{R_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$
is affine in $X$, where $F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)$.

**Proposition 2 (Stochastic Gradient):**  
For independent $A_s$ and $F_{s,t}^m$:
$$
\mathbb{E}\left[\nabla_{x_s^m} W_t^m(X)\right] = \mathbb{E}[A_s] \cdot \mathbb{E}[F_{s,t}^m]
$$

**Proposition 3 (Convergence Rate):**  
For IID simulations:
$$
\left|\hat{\mathbb{E}}_n[W_T] - \mathbb{E}[W_T]\right| = O_p\left(\frac{1}{\sqrt{n}}\right)
$$

---

## Limitations

1. **Cache doesn't detect component mutations:** Call `model.clear_cache()` after modifying `income` or `accounts` in-place.

2. **Fixed horizon T:** For dynamic $T$ optimization, use outer loop or `optimization.py` bilevel formulation.

3. **Constant correlation matrix:** $\Sigma$ fixed over time (no regime-switching).

---

## Complete example

```python
from datetime import date
import numpy as np
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.portfolio import Account
from finopt.src.model import FinancialModel

# Setup
income = IncomeModel(
    fixed=FixedIncome(base=1_500_000, annual_growth=0.04,
                     salary_raises={date(2025, 7, 1): 200_000}),
    variable=VariableIncome(base=300_000, sigma=0.15, 
                           seasonality=[1.0, 0.9, 1.1, 1.0, 1.2, 1.1,
                                       1.0, 0.9, 0.95, 1.05, 1.1, 1.3],
                           seed=100)
)

accounts = [
    Account.from_annual("Emergency", annual_return=0.035, annual_volatility=0.06),
    Account.from_annual("Housing", annual_return=0.08, annual_volatility=0.15)
]

model = FinancialModel(income, accounts)

# Simulate with glide path
T = 36
equity_fractions = np.linspace(0.7, 0.4, T)
X_glide = np.column_stack([1 - equity_fractions, equity_fractions])

result = model.simulate(T=T, X=X_glide, n_sims=2000, seed=42, 
                       start=date(2025, 1, 1))

# Analysis
print("=== Summary Statistics ===")
print(result.summary(confidence=0.95))

print("\n=== Aggregate Risk Metrics ===")
print(result.aggregate_metrics())

metrics = result.metrics(account="Emergency")
print(f"\nEmergency - Mean Sharpe: {metrics['sharpe'].mean():.3f}")

conv = result.convergence_analysis()
print(f"\nConvergence - Final SE: ${conv['std_error'].iloc[-1]:,.0f}")

# Visualization
model.plot("wealth", result=result, title="Glide Path Strategy",
           show_trajectories=True, save_path="glide_path.png")

# Cache info
print(f"\nCache: {model.cache_info()}")
```

---

## References

**Internal modules:**
- **income.py:** Generates $A_t$ (contributions) with fixed + variable streams
- **returns.py:** Generates $R_t^m$ (correlated lognormal returns)
- **portfolio.py:** Executes $W_{t+1}^m = (W_t^m + A_t x_t^m)(1 + R_t^m)$ with affine representation
- **optimization.py:** Consumes `SimulationResult` for bilevel goal-seeking problems
- **utils.py:** Rate conversions, financial metrics (CAGR, drawdown, Sharpe)
