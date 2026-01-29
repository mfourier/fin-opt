# `model` — Unified Orchestration for FinOpt

> **Purpose:** Unified **orchestrator** for Monte Carlo simulation, integrating income generation, return modeling, withdrawals, and portfolio dynamics into a single coherent interface with intelligent caching, reproducibility guarantees, and optimization-ready outputs.
> `model.py` is the **facade layer**: while `income.py` generates cash flows, `returns.py` produces stochastic returns, `withdrawal.py` models cash outflows, and `portfolio.py` executes wealth dynamics, `model.py` coordinates the entire pipeline and packages results for analysis and optimization.

---

## Why a dedicated model module?

Financial planning requires **end-to-end simulation** with multiple moving parts. `model.py` provides:

- **Single entry point:** Unified `FinancialModel` class orchestrates all components
- **Intelligent caching:** Parameter-based memoization avoids redundant computation
- **Reproducibility:** Explicit seed management with automatic propagation
- **Rich analytics:** `SimulationResult` container with financial metrics computation
- **Seamless visualization:** Auto-simulation in `plot()` methods
- **Optimization integration:** Direct `optimize()` method for goal-seeking
- **Withdrawal support:** Scheduled and stochastic withdrawals via `WithdrawalModel`

---

## Design principles

1. **Facade pattern:** Coordinates but doesn't re-implement (loose coupling)
2. **Explicit reproducibility:** Seed propagation ensures statistical independence:
   - Income uses `seed`
   - Returns use `seed+1`
   - Withdrawals use `seed+2`
3. **Type-safe results:** `SimulationResult` as explicit dataclass (not `dict`)
4. **Zero-overhead visualization:** `plot()` auto-simulates with caching
5. **Optimization-ready:** `optimize()` method for bilevel goal-seeking

---

## Core components

### 1) `SimulationResult`

Container for complete Monte Carlo simulation output with lazy-computed analytics.

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
    withdrawals: np.ndarray         # (n_sims, T, M) or (T, M) or None

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

#### `summary(confidence=0.95) → pd.DataFrame`

Statistical summary with confidence intervals:
$$
\text{CI}_{1-\alpha}(W_T) = \left[F^{-1}_{W_T}(\alpha/2), F^{-1}_{W_T}(1-\alpha/2)\right]
$$

#### `convergence_analysis() → pd.DataFrame`

Monte Carlo convergence diagnostics via standard error: $\text{SE}(n) = \frac{\sigma_{W_T}}{\sqrt{n}}$

---

### 2) `FinancialModel`

Unified orchestrator coordinating the flow: `income → contributions (A) → returns (R) → withdrawals (D) → wealth (W)`

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

**Key methods:**

| Method | Description |
|--------|-------------|
| `simulate(T, X, n_sims, ...)` | Run Monte Carlo simulation |
| `optimize(goals, optimizer, ...)` | Find minimum-horizon policy |
| `simulate_from_optimization(opt_result, ...)` | Simulate with optimal policy |
| `verify_goals(result, goals, ...)` | Validate goal satisfaction |
| `plot(mode, ...)` | Unified visualization |
| `cache_info()` | Get cache statistics |
| `clear_cache()` | Free cached results |

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
    use_cache: bool = True,
    withdrawals: Optional[WithdrawalModel] = None
) -> SimulationResult
```

**Pipeline execution:**

1. **Cache lookup:** SHA256 hash of `(T, X.tobytes(), n_sims, start, seed, withdrawals.to_dict())`
2. **Seed propagation:**
   ```python
   A = income.contributions(T, start, seed=seed, n_sims=n_sims)
   R = returns.generate(T, n_sims, seed=None if seed is None else seed+1)
   D = withdrawals.to_array(T, start, accounts, n_sims, seed=seed+2)  # if provided
   ```
3. **Wealth dynamics:**
   ```python
   portfolio_result = portfolio.simulate(A, R, X, D=D)
   ```
   Uses wealth evolution:
   $$
   W_{t+1}^m = (W_t^m + A_t x_t^m - D_t^m)(1 + R_t^m)
   $$
4. **Result packaging:** Wrap arrays in `SimulationResult` dataclass

**Example:**
```python
from datetime import date
from finopt.src.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent

# Basic simulation (no withdrawals)
X = np.tile([0.7, 0.3], (24, 1))
result = model.simulate(T=24, X=X, n_sims=1000, seed=42)

# With withdrawals
withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule(events=[
        WithdrawalEvent("Housing", 2_000_000, date(2026, 6, 1), "Vacation")
    ])
)
result = model.simulate(T=24, X=X, n_sims=1000, seed=42, withdrawals=withdrawals)

# Second call with same params: O(1) (cached)
result2 = model.simulate(T=24, X=X, n_sims=1000, seed=42, withdrawals=withdrawals)
assert result is result2  # same object
```

---

## Optimization workflow

### `optimize()` method

```python
def optimize(
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    optimizer: AllocationOptimizer,
    T_max: int = 240,
    n_sims: int = 500,
    seed: Optional[int] = None,
    start: Optional[date] = None,
    verbose: bool = True,
    search_method: str = "binary",
    withdrawals: Optional[WithdrawalModel] = None,
    withdrawal_epsilon: float = 0.05,
    **solver_kwargs
) -> OptimizationResult
```

**Bilevel optimization:**
- **Outer problem:** Minimize horizon $T$
- **Inner problem:** Find feasible allocation $X^*$ at horizon $T$

**Search strategies:**
- `"binary"`: Binary search (faster, ~50% fewer iterations)
- `"linear"`: Sequential search (safer, guaranteed to find solution)

**Example:**
```python
from finopt.src.goals import IntermediateGoal, TerminalGoal
from finopt.src.optimization import CVaROptimizer

goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing", threshold=20_000_000, confidence=0.90)
]

optimizer = CVaROptimizer(n_accounts=model.M, objective="balanced")

result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    search_method="binary",
    withdrawals=withdrawals,
    withdrawal_epsilon=0.05
)

print(f"Optimal horizon: T*={result.T} months")
print(result.summary())
```

---

### `simulate_from_optimization()` method

Convenience wrapper to simulate with optimal policy from optimization.

```python
def simulate_from_optimization(
    opt_result: OptimizationResult,
    n_sims: int = 500,
    seed: Optional[int] = None,
    start: Optional[date] = None,
    withdrawals: Optional[WithdrawalModel] = None
) -> SimulationResult
```

**Example:**
```python
# Optimize
opt_result = model.optimize(goals, optimizer, T_max=120, n_sims=500, seed=42)

# Validate with 1000 fresh scenarios
sim_result = model.simulate_from_optimization(
    opt_result,
    n_sims=1000,
    seed=999,  # Different seed for out-of-sample validation
    withdrawals=withdrawals
)
```

---

### `verify_goals()` method

Validate goal satisfaction in simulation/optimization result.

```python
def verify_goals(
    result: Union[SimulationResult, OptimizationResult],
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    start: Optional[date] = None
) -> Dict[Goal, Dict[str, float]]
```

**Returns:** For each goal:
- `satisfied`: bool
- `violation_rate`: float (empirical ℙ(W < threshold))
- `required_rate`: float (goal's ε = 1 - confidence)
- `margin`: float (positive → satisfied)
- `median_shortfall`: float
- `n_violations`: int

**Example:**
```python
status = model.verify_goals(sim_result, goals)

for goal, metrics in status.items():
    if not metrics['satisfied']:
        print(f"VIOLATED: {goal}")
        print(f"  Violation rate: {metrics['violation_rate']:.2%}")
        print(f"  Shortfall: ${metrics['median_shortfall']:,.0f}")
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
    goals: Optional[List] = None,
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

**Goals visualization:**
When `goals` parameter is provided to wealth plots:
- TerminalGoal: Horizontal dashed line at threshold
- IntermediateGoal: Dotted line with diamond marker at goal month

**Examples:**
```python
# Direct plotting (auto-simulates + caches)
model.plot("wealth", T=24, X=X, n_sims=500, seed=42,
           start=date(2025, 1, 1), goals=goals)

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

## Cache management

### Inspection

```python
info = model.cache_info()
# {'size': 3, 'memory_mb': 28.7}
```

**Memory estimate:** $\approx n_{\text{sims}} \cdot T \cdot M \cdot 24$ bytes (wealth + returns + contributions)

### Cache key components

The cache key is computed via SHA256 hash of:
- `T`: Horizon
- `X.tobytes()`: Allocation policy bytes
- `n_sims`: Number of simulations
- `start`: Start date
- `seed`: Random seed
- `withdrawals.to_dict()`: Withdrawal configuration (if provided)

### Trade-offs

**With cache (default):**
- Instant repeated calls
- RAM scales with parameter space

**Without cache:**
```python
model = FinancialModel(income, accounts, enable_cache=False)
```
- Minimal memory
- Re-simulation on every call

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

### B) Simulation with withdrawals

```python
from finopt.src.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent

withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule(events=[
        WithdrawalEvent("Housing", 5_000_000, date(2027, 1, 1), "Pie departamento")
    ])
)

result = model.simulate(
    T=36, X=X, n_sims=2000, seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals
)

# Check withdrawals are recorded
print(result.withdrawals.shape)  # (2000, 36, 2)
```

### C) Statistical analysis

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

### D) Full optimization workflow

```python
from finopt.src.goals import IntermediateGoal, TerminalGoal
from finopt.src.optimization import CVaROptimizer

# Define goals
goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing", threshold=20_000_000, confidence=0.90)
]

# Optimize
optimizer = CVaROptimizer(n_accounts=model.M, objective="balanced")
opt_result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals,
    withdrawal_epsilon=0.05
)

print(f"Optimal horizon: T*={opt_result.T}")
print(opt_result.summary())

# Validate with fresh scenarios
sim_result = model.simulate_from_optimization(opt_result, n_sims=1000, seed=999)
status = model.verify_goals(sim_result, goals)

for goal, metrics in status.items():
    print(f"{goal.account}: {'✓' if metrics['satisfied'] else '✗'}")
```

### E) Visualization

```python
# Pre-simulation plots
model.plot("income", months=24, start=date(2025, 1, 1))
model.plot("returns_cumulative", T=120, n_sims=500, start=date(2025, 1, 1))

# Simulation-based with goals (auto-simulates + caches)
model.plot("wealth", T=24, X=X, n_sims=500, seed=42,
           start=date(2025, 1, 1), goals=goals, show_trajectories=True)

# Using optimization result
model.plot("wealth", result=sim_result, goals=goals)
```

---

## Mathematical results

**Proposition 1 (Affine Wealth):**
For any allocation policy $X$, return realization $\{R_t^m\}$, and withdrawal schedule $\{D_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} (A_s x_s^m - D_s^m) F_{s,t}^m
$$
is affine in $X$ (since $D$ is a parameter), where $F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)$.

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

## Seed propagation

To ensure statistical independence while maintaining reproducibility:

| Component | Seed Used | Purpose |
|-----------|-----------|---------|
| Income | `seed` | Stochastic contributions (VariableIncome) |
| Returns | `seed + 1` | Correlated lognormal returns |
| Withdrawals | `seed + 2` | Stochastic withdrawals |

This ensures that changing the income model doesn't affect return scenarios (and vice versa), while maintaining full reproducibility when the same seed is used.

---

## Limitations

1. **Cache doesn't detect component mutations:** Call `model.clear_cache()` after modifying `income` or `accounts` in-place.

2. **Fixed horizon T per simulation:** For dynamic $T$ optimization, use `optimize()` method.

3. **Constant correlation matrix:** $\Sigma$ fixed over time (no regime-switching).

4. **Cache memory:** Large parameter sweeps can consume significant RAM.

---

## Complete example

```python
from datetime import date
import numpy as np
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.portfolio import Account
from finopt.src.model import FinancialModel
from finopt.src.goals import IntermediateGoal, TerminalGoal
from finopt.src.optimization import CVaROptimizer
from finopt.src.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent

# 1. Setup income
income = IncomeModel(
    fixed=FixedIncome(base=1_500_000, annual_growth=0.04,
                     salary_raises={date(2025, 7, 1): 200_000}),
    variable=VariableIncome(base=300_000, sigma=0.15,
                           seasonality=[1.0, 0.9, 1.1, 1.0, 1.2, 1.1,
                                       1.0, 0.9, 0.95, 1.05, 1.1, 1.3],
                           seed=100)
)

# 2. Setup accounts
accounts = [
    Account.from_annual("Emergency", annual_return=0.035, annual_volatility=0.06,
                        display_name="Fondo de Emergencia"),
    Account.from_annual("Housing", annual_return=0.08, annual_volatility=0.15,
                        display_name="Ahorro Vivienda")
]

# 3. Create model
model = FinancialModel(income, accounts)

# 4. Define withdrawals
withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule(events=[
        WithdrawalEvent("Housing", 5_000_000, date(2027, 6, 1), "Pie departamento")
    ])
)

# 5. Define goals
goals = [
    IntermediateGoal(date=date(2025, 12, 1), account="Emergency",
                     threshold=5_000_000, confidence=0.95),
    TerminalGoal(account="Housing", threshold=30_000_000, confidence=0.90)
]

# 6. Optimize
optimizer = CVaROptimizer(n_accounts=model.M, objective="balanced")
opt_result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=60,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals,
    withdrawal_epsilon=0.05,
    verbose=True
)

print(f"\n=== Optimization Result ===")
print(f"Optimal horizon: T*={opt_result.T} months")
print(opt_result.summary())

# 7. Validate with fresh scenarios
sim_result = model.simulate_from_optimization(
    opt_result,
    n_sims=1000,
    seed=999,
    withdrawals=withdrawals
)

# 8. Check goals
status = model.verify_goals(sim_result, goals)
print("\n=== Goal Validation ===")
for goal, metrics in status.items():
    symbol = "✓" if metrics['satisfied'] else "✗"
    print(f"[{symbol}] {goal.account}: {metrics['violation_rate']:.1%} violations")

# 9. Statistical summary
print("\n=== Final Wealth Summary ===")
print(sim_result.summary(confidence=0.95))

# 10. Visualize
model.plot("wealth", result=sim_result, goals=goals,
           title="Optimized Portfolio with Withdrawals",
           show_trajectories=True)

# 11. Cache info
print(f"\nCache: {model.cache_info()}")
```

---

## References

**Internal modules:**
- **income.py:** Generates $A_t$ (contributions) with fixed + variable streams
- **returns.py:** Generates $R_t^m$ (correlated lognormal returns)
- **withdrawal.py:** Generates $D_t^m$ (scheduled + stochastic withdrawals)
- **portfolio.py:** Executes $W_{t+1}^m = (W_t^m + A_t x_t^m - D_t^m)(1 + R_t^m)$
- **goals.py:** Defines chance constraints for optimization
- **optimization.py:** Implements CVaR reformulation and bilevel search
