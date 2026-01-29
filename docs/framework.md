# FinOpt — Technical Framework

> **Tagline:** Intelligent financial planning through stochastic simulation and convex optimization under uncertainty.

This document describes the **theoretical and technical framework** of **FinOpt**, a modular system that connects **user objectives** (emergency funds, housing, retirement) with **optimal investment strategies** under stochastic income and returns via chance-constrained optimization.

---

## 0. System Architecture

FinOpt is composed of **six core modules**:

| Module | Purpose | Key Abstractions |
|--------|---------|------------------|
| **`income.py`** | Cash flow modeling | `FixedIncome`, `VariableIncome`, `IncomeModel` |
| **`portfolio.py`** | Wealth dynamics | `Account`, `Portfolio` (affine wealth executor) |
| **`returns.py`** | Stochastic returns | `ReturnModel` (correlated lognormal) |
| **`goals.py`** | Goal specification | `IntermediateGoal`, `TerminalGoal`, `GoalSet` |
| **`optimization.py`** | Solvers | `SAAOptimizer`, `CVaROptimizer`, `GoalSeeker` |
| **`model.py`** | Orchestration | `FinancialModel` (unified facade) |

**Dependency graph:**
```
model.py (FinancialModel)
    ├─→ income.py (IncomeModel)
    ├─→ portfolio.py (Portfolio)
    ├─→ returns.py (ReturnModel)
    └─→ optimization.py (GoalSeeker)
            ├─→ goals.py (GoalSet)
            └─→ AllocationOptimizer (SAAOptimizer)
```

**Design principles:**
- **Loose coupling**: Each module usable independently
- **Lazy imports**: Optimization only loaded when needed (TYPE_CHECKING)
- **Separation of concerns**: Portfolio executes dynamics, doesn't generate returns
- **Reproducibility**: Explicit seed management with automatic propagation

---

## 1. Income Module

Total monthly income at time $t$ is composed of fixed and variable parts:


$$
Y_t = y_t^{\text{fixed}} + Y_t^{\text{variable}}
$$

### 1.1 Fixed Income

The fixed component, $y_t^{\text{fixed}}$, reflects baseline salary subject to compounded annual growth $g$ and scheduled raises $\{(d_k, \Delta_k)\}$:


$$
y_t^{\text{fixed}} = \text{current\_salary}(t) \cdot (1+m)^{\Delta t}
$$

where $m = (1 + g)^{1/12} - 1$ is the **monthly compounded rate**, and $\Delta t$ represents time since the last raise.

**API:**
```python
fixed = FixedIncome(
    base=1_400_000,           # Current monthly salary
    annual_growth=0.03,       # 3% annual raises
    raises=[(12, 100_000)]    # +100k at month 12
)
```

### 1.2 Variable Income

The variable component, $Y_t^{\text{variable}}$, models irregular income (freelance, bonuses) with:

- **Seasonality**: $s \in \mathbb{R}^{12}$ (multiplicative monthly factors)
- **Noise**: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ (Gaussian shocks)
- **Growth**: same compounded rate $m$ applied to base income
- **Boundaries**: optional floor and cap constraints

The underlying stochastic projection:


$$
\tilde{Y}_t = \max(\text{floor},\ \mu_t (1 + \epsilon_t)), \quad \text{where } \mu_t = \text{base} \cdot (1 + m)^t \cdot s_{(t \bmod 12)}
$$

Then, guardrails:


$$
Y_t^{\text{variable}} = \begin{cases}
0 & \text{if } \tilde{Y}_t < 0 \\
\tilde{Y}_t & \text{if } 0 \leq \tilde{Y}_t \leq \text{cap} \\
\text{cap} & \text{if } \tilde{Y}_t > \text{cap}
\end{cases}
$$

**API:**
```python
variable = VariableIncome(
    base=200_000,                   # Average monthly variable income
    sigma=0.10,                     # 10% volatility
    seasonality=[1.2, 0.8, ...],   # 12-month cycle
    seed=42                         # Reproducibility
)
```

### 1.3 Contributions

A fraction of income is allocated monthly via calendar-rotating schedules:


$$
A_t = \alpha_{(t \bmod 12)}^{f} \cdot y_t^{\text{fixed}} + \alpha_{(t \bmod 12)}^{v} \cdot Y_t^{\text{variable}}
$$

where $\alpha^f, \alpha^v \in [0,1]^{12}$ control fixed/variable contribution rates, rotated according to `start` date.

**API:**
```python
income = IncomeModel(fixed=fixed, variable=variable)
A = income.contributions(
    months=24,
    start=date(2025, 1, 1),
    n_sims=500,
    seed=42
)  # → (500, 24) array
```

---

## 2. Portfolio Dynamics

### 2.1 Wealth Evolution

Multiple accounts $m \in \mathcal{M} = \{1,\dots,M\}$ evolve via:


$$
W_{t+1}^m = \big(W_t^m + A_t x_t^m\big)(1 + R_t^m)
$$

where:
- $W_t^m$ = wealth in account $m$ at month $t$
- $A_t x_t^m$ = allocated contribution ($x_t^m$ fraction of total contribution $A_t$)
- $R_t^m$ = stochastic return of account $m$

**API:**
```python
accounts = [
    Account.from_annual("Emergency", annual_return=0.04, 
                        annual_volatility=0.05, initial_wealth=0),
    Account.from_annual("Housing", annual_return=0.07, 
                        annual_volatility=0.12, initial_wealth=0)
]
portfolio = Portfolio(accounts)
```

### 2.2 Allocation Policy

Contributions allocated via decision variables $x_t^m \in [0,1]$ satisfying:


$$
\sum_{m=1}^M x_t^m = 1, \quad x_t^m \ge 0, \quad \forall t
$$

The **allocation simplex** at horizon $T$ is:


$$
\mathcal{X}_T = \left\{ X \in \mathbb{R}^{T \times M} : 
\begin{aligned}
& x_t^m \ge 0 && \text{(non-negativity)} \\
& \sum_{m=1}^M x_t^m = 1 && \text{(budget constraint)} \\
& \forall t = 0, \dots, T-1
\end{aligned}
\right\}
$$

representing all **budget-feasible allocation policies** (full contribution deployment each month).

**Geometric interpretation:** $\mathcal{X}_T$ is the Cartesian product of $T$ probability simplices:

$$
\mathcal{X}_T = \underbrace{\Delta^{M-1} \times \cdots \times \Delta^{M-1}}_{T \text{ times}}, \quad \Delta^{M-1} = \left\{x \in \mathbb{R}_+^M : \sum_{m=1}^M x^m = 1\right\}
$$

**API:**
```python
X = np.tile([0.6, 0.4], (T, 1))  # 60-40 split ∈ 𝒳_T
```

### 2.3 Affine Wealth Representation

Recursive wealth can be expressed in **closed-form**:

$$
\boxed{
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s \, x_s^m \, F_{s,t}^m
}
$$

where the **accumulation factor** from month $s$ to $t$ is:


$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)
$$

**Key properties:**
1. $W_t^m(X)$ is **affine** in allocation policy $X$ (linearity immediately visible)
2. Gradient: $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ (analytical!)
3. Enables gradient-based convex optimization

**Implementation:**
```python
result = portfolio.simulate(A=A, R=R, X=X, method="affine")
W = result["wealth"]  # (n_sims, T+1, M)
```

---

## 3. Goals Framework

### 3.1 Goal Types

FinOpt supports two goal primitives:

**Intermediate Goal** (fixed time):
```python
IntermediateGoal(
    month=12,                    # Or date(2026, 1, 1)
    account="Emergency",
    threshold=5_500_000,         # 5.5M CLP
    confidence=0.90              # 90% probability
)
```

Mathematical constraint:

$$
\mathbb{P}\big(W_{t}^m \ge b\big) \ge 1-\varepsilon
$$

**Terminal Goal** (variable time):
```python
TerminalGoal(
    account="Housing",
    threshold=20_000_000,        # 20M CLP
    confidence=0.90
)
```

Mathematical constraint:

$$
\mathbb{P}\big(W_{T}^m \ge b\big) \ge 1-\varepsilon
$$

where $T$ is the optimization horizon (decision variable).

### 3.2 Goal Set

The **goal set** $\mathcal{G}$ is partitioned into:

$$
\mathcal{G} = \mathcal{G}_{\text{int}} \cup \mathcal{G}_{\text{term}}
$$

where:
- $\mathcal{G}_{\text{int}}$: intermediate goals (constrain $T_{\min}$)
- $\mathcal{G}_{\text{term}}$: terminal goals (determine $T^*$)

**Properties:**

1. **Minimum horizon constraint:**
   
$$
T \geq T_{\min} := \max_{g \in \mathcal{G}_{\text{int}}} t_g
$$

2. **Goal resolution:** Month indices resolved via `start` date for calendar alignment

3. **Account mapping:** Names → indices via `account_names` parameter

**API:**
```python
from datetime import date

goals = [
    IntermediateGoal(date=date(2026, 1, 1), account="Emergency",
                    threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing",
                threshold=20_000_000, confidence=0.90)
]

goal_set = GoalSet(goals, account_names=["Emergency", "Housing"],
                   start_date=date(2025, 1, 1))
```

### 3.3 Horizon Estimation Heuristic

For **terminal-only goals** ($\mathcal{G}_{\text{int}} = \emptyset$), naive linear search starts at $T=1$, wasting iterations. Instead, FinOpt uses a **conservative heuristic**:


$$
T_{\text{start}} = \max_{g \in \mathcal{G}_{\text{term}}} \left\lceil \frac{b_g - W_0^m \cdot (1 + \mu)^{T_{\min}}}{A_{\text{avg}} \cdot x_{\min}^m \cdot (1 + \mu - \sigma)} \right\rceil
$$

where:
- $A_{\text{avg}}$: average monthly contribution (sampled)
- $\mu, \sigma$: expected return and volatility of account $m$
- $x_{\min}^m$: minimum allocation fraction (conservative: 0.1)
- Safety margin: multiply by 0.8 to start early

**Implementation:** `GoalSet.estimate_minimum_horizon()`

---

## 4. Optimization Framework

### 4.1 Bilevel Problem

Find the **minimum time** $T^*$ to achieve all goals while optimizing an objective $f(X)$:

$$
\boxed{
\min_{T \in \mathbb{N}} \;\; T \quad \text{s.t.} \quad \max_{X \in \mathcal{F}_T} f(X) > -\infty
}
$$

where the **goal-feasible set** at horizon $T$ is:


$$
\mathcal{F}_T := \left\{ X \in \mathcal{X}_T : \begin{aligned}
& \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1-\varepsilon_t^m, \; \forall g \in \mathcal{G}_{\text{int}}, \\
& \mathbb{P}\big(W_T^m(X) \ge b^m\big) \ge 1-\varepsilon^m, \; \forall g \in \mathcal{G}_{\text{term}}
\end{aligned} \right\}
$$

**Equivalent decomposition:**

- **Outer problem:** Find minimum horizon $T \in [T_{\text{start}}, T_{\max}]$ with non-empty feasible set
- **Inner problem:** For fixed $T$, solve:
$$
\begin{aligned}
\max_{X \in \mathcal{X}_T} \;\; & f(X) \\[0.5em]
\text{s.t.} \;\; & X \in \mathcal{F}_T
\end{aligned}
$$

### 4.2 Inner Problem (Fixed Horizon)

For given horizon $T$, solve:


$$
\begin{aligned}
\max_{X \in \mathcal{X}_T} \;\; & f(X) \\[0.5em]
\text{s.t.} \;\; & \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1-\varepsilon_t^m, \quad \forall g \in \mathcal{G}_{\text{int}} \\
& \mathbb{P}\big(W_T^m(X) \ge b^m\big) \ge 1-\varepsilon^m, \quad \forall g \in \mathcal{G}_{\text{term}}
\end{aligned}
$$

**Objective functions:**
- `"terminal_wealth"`: $f(X) = \mathbb{E}\big[\sum_m W_T^m(X)\big]$ (default)
- `"low_turnover"`: $f(X) = \mathbb{E}[W_T] - \lambda \sum_{t,m} |x_{t+1,m} - x_t^m|$
- `"risk_adjusted"`: $f(X) = \mathbb{E}[W_T] - \lambda \cdot \text{Std}(W_T)$
- Custom: user-provided callable

### 4.3 Chance Constraint Reformulation

**Challenge:** Indicator function $\mathbb{1}\{W \geq b\}$ is discontinuous.

#### Sample Average Approximation (SAA)

Discrete approximation with $N$ scenarios $\omega^{(i)}$:


$$
\frac{1}{N}\sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \ge b_t^m\} \ge 1-\varepsilon_t^m
$$

**Issue:** Non-smooth, no gradient.

#### Sigmoid Smoothing (SAAOptimizer)

Replace indicator with **sigmoid** $\sigma(z) = 1/(1 + e^{-z})$:


$$
\boxed{
\frac{1}{N} \sum_{i=1}^N \sigma\left(\frac{W_t^m(X; \omega^{(i)}) - b_t^m}{\tau}\right) \ge 1-\varepsilon_t^m
}
$$

**Properties:**
1. **Differentiability**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ → analytical gradient
2. **Approximation quality**: controlled by temperature $\tau$
   - Small $\tau$ (0.01): $\sigma \approx \mathbb{1}$ (sharp, harder to optimize)
   - Large $\tau$ (1.0): $\sigma \approx 0.5$ (smooth, loose approximation)
   - **Balanced** $\tau$ (0.1): trade-off for practical optimization
3. **Convexity**: smoothed constraint is convex in $X$ (via affine wealth)

**Gradient computation:**

$$
\frac{\partial}{\partial x_s^m} \left[\frac{1}{N} \sum_{i=1}^N \sigma\left(\frac{W_t^m(X; \omega^{(i)}) - b}{\tau}\right)\right] = \frac{1}{N\tau} \sum_{i=1}^N \sigma'(z^{(i)}) \cdot A_s^{(i)} \cdot F_{s,t}^{m,(i)}
$$

where $z^{(i)} = (W_t^m(X; \omega^{(i)}) - b)/\tau$.

**API:**
```python
from finopt.src.optimization import SAAOptimizer

optimizer = SAAOptimizer(
    n_accounts=2,
    tau=0.1,                      # Sigmoid temperature
    objective="terminal_wealth",
    account_names=["Emergency", "Housing"]
)
```

#### CVaR Reformulation (CVaROptimizer - Stub)

Risk-adjusted objective via **Conditional Value-at-Risk**:


$$
\max \; \mathbb{E}[W_T] - \lambda \cdot \text{CVaR}_{ \alpha }(-W_T)
$$

subject to goal constraints. Requires CVXPY for implementation.

### 4.4 Solution Strategy (GoalSeeker)

**Bilevel solver** with linear search and warm start:

```python
class GoalSeeker:
    def seek(goals, A_generator, R_generator, W0, ...):
        # Estimate intelligent starting horizon
        T_start = estimate_horizon(goals, A_generator, W0)
        
        X_prev = None  # Warm start
        for T in range(T_start, T_max + 1):
            # Generate scenarios for current horizon
            A = A_generator(T, n_sims, seed)
            R = R_generator(T, n_sims, seed+1)
            
            # Solve inner problem
            result = optimizer.solve(T, A, R, W0, goals, X_init=X_prev)
            
            # Check feasibility (exact SAA validation)
            if result.feasible:
                return result  # Found T*
            
            # Warm start: extend X for next iteration
            X_prev = extend_policy(result.X)
        
        raise ValueError("No feasible solution in [T_start, T_max]")
```

**Key features:**
1. **Intelligent start**: Skips infeasible horizons via heuristic
2. **Warm start**: Extends previous $X$ policy for faster convergence
3. **Exact validation**: Final feasibility check uses non-smoothed SAA

**API:**
```python
from finopt.src.optimization import GoalSeeker

seeker = GoalSeeker(optimizer, T_max=240, verbose=True)
result = seeker.seek(goals, A_generator, R_generator, W0, 
                    start_date=date(2025,1,1), n_sims=500, seed=42)
```

---

## 5. Integration: FinancialModel

### 5.1 Unified Facade

`FinancialModel` orchestrates all components:

```python
from finopt.src.model import FinancialModel

model = FinancialModel(
    income=income,              # IncomeModel
    accounts=accounts,          # List[Account]
    default_correlation=None,   # Return correlation matrix
    enable_cache=True           # Simulation caching
)
```

**Attributes:**
- `model.income`: IncomeModel instance
- `model.returns`: ReturnModel instance (auto-created)
- `model.portfolio`: Portfolio instance (auto-created)
- `model.M`: Number of accounts

### 5.2 Core Methods

#### Simulation
```python
result = model.simulate(
    T=24,
    X=X,                        # (24, 2) allocation policy
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    use_cache=True
)
# Returns: SimulationResult with wealth, contributions, returns
```

**Features:**
- Automatic seed propagation (income: seed, returns: seed+1)
- SHA256 cache keying for instant re-runs
- Affine wealth computation for optimization readiness

#### Optimization
```python
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    verbose=True
)
# Returns: OptimizationResult with X*, T*, feasibility, diagnostics
```

**Features:**
- Lazy import of optimization module (TYPE_CHECKING)
- Duck typing validation of optimizer interface
- Automatic generator construction for income/returns
- Extracts `W0` from portfolio automatically

#### Validation
```python
status = model.verify_goals(result, goals)
# Returns: dict mapping each goal to violation metrics
```

**Features:**
- Handles both `SimulationResult` and `OptimizationResult`
- Auto-converts `OptimizationResult` → `SimulationResult` (500 fresh scenarios)
- Computes empirical violation rates

#### Visualization
```python
model.plot("wealth", T=24, X=X, n_sims=500, seed=42, 
          start=date(2025,1,1))
```

**Modes:**
- Pre-simulation: `"income"`, `"contributions"`, `"returns"`
- Simulation-based: `"wealth"`, `"comparison"`

### 5.3 Workflow Example

```python
# 1. Setup
income = IncomeModel(
    fixed=FixedIncome(base=1_400_000, annual_growth=0.03),
    variable=VariableIncome(base=200_000, sigma=0.10)
)
accounts = [
    Account.from_annual("Emergency", 0.04, 0.05),
    Account.from_annual("Housing", 0.07, 0.12)
]
model = FinancialModel(income, accounts)

# 2. Define goals
goals = [
    IntermediateGoal(date=date(2026, 1, 1), account="Emergency",
                    threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing",
                threshold=20_000_000, confidence=0.90)
]

# 3. Optimize
optimizer = SAAOptimizer(n_accounts=2, tau=0.1)
opt_result = model.optimize(goals, optimizer, T_max=120, seed=42)

print(f"Optimal horizon: T*={opt_result.T} months")
print(f"Feasible: {opt_result.feasible}")

# 4. Validate with fresh scenarios
sim_result = model.simulate_from_optimization(opt_result, n_sims=1000, seed=999)
status = model.verify_goals(sim_result, goals)

for goal, metrics in status.items():
    print(f"{goal}: {metrics['satisfied']} "
          f"(violation rate: {metrics['violation_rate']:.2%})")

# 5. Visualize
model.plot("wealth", result=sim_result, show_trajectories=True)
```

---

## 6. Implementation Details

### 6.1 Calendar Alignment

All projections use `start: date` parameter:
- Seasonality rotates via offset $= (\text{start.month} - 1)$
- Salary raises applied at specific dates relative to start
- Contribution fractions rotate cyclically to match fiscal year
- Temporal index: `month_index(start, T)` → DatetimeIndex

### 6.2 Seed Management

**Reproducibility architecture:**
```
User seed
    ├─→ Income: seed
    └─→ Returns: seed + 1
```

**Rationale:** Statistical independence between income and return shocks.

**Implementation:**
```python
A = income.contributions(months=T, seed=seed)
R = returns.generate(T, seed=None if seed is None else seed + 1)
```

### 6.3 Memory Management

**Accumulation factors:** $F \in \mathbb{R}^{N \times (T+1) \times (T+1) \times M}$

Memory usage: $N \cdot T^2 \cdot M \cdot 8$ bytes

**Estimates:**
- $N=500, T=24, M=2$: ~115 MB
- $N=500, T=120, M=5$: ~14 GB ⚠️
- $N=1000, T=240, M=10$: ~221 GB (infeasible)

**Mitigation strategies:**
- Use `method="recursive"` for large $T$ (no $F$ precomputation)
- Process simulations in batches (chunk $N$)
- Compute gradients on-the-fly (store only needed $F_{s,t}$ pairs)
- Use sparse storage for intermediate-goal-only problems

### 6.4 Output Formats

All methods support flexible output:
- `output="array"`: NumPy arrays (computational efficiency)
- `output="series"`: Pandas Series with calendar index (reporting)
- `output="dataframe"`: Component breakdown (analysis)

### 6.5 Type Safety

**Lazy imports for optimization:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimization import AllocationOptimizer, OptimizationResult
```

**Runtime validation:**
```python
# Duck typing (no import needed at runtime)
if not hasattr(optimizer, 'solve') or not callable(optimizer.solve):
    raise TypeError("optimizer must implement .solve() method")
```

---

## 7. Key Mathematical Results

**Proposition 1 (Affine Wealth):**  
For any allocation policy $X \in \mathcal{X}_T$ and return realization $\{R_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$
is affine in $X$.

**Corollary 1 (Convex Feasible Set):**  
For deterministic constraints $W_t^m(X) \ge b_t^m$, the feasible allocation set is a convex polytope.

**Proposition 2 (Analytical Gradient):**  
The sensitivity of wealth to allocation at month $s$ is:
$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m, \quad s < t
$$

**Corollary 2 (Monotonicity):**  
If $F_{s,t}^m > 0$ (positive returns), then $W_t^m(X)$ is strictly increasing in $x_s^m$.

**Proposition 3 (Sigmoid Approximation Error):**  
For temperature $\tau > 0$:
$$
\left|\sigma\left(\frac{z}{\tau}\right) - \mathbb{1}\{z \geq 0\}\right| \le \frac{1}{2}
$$
with error decaying exponentially in $|z|/\tau$.

**Proposition 4 (SAA Consistency):**  
Under mild regularity conditions, as $N \to \infty$:
$$
\frac{1}{N}\sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \ge b\} \xrightarrow{a.s.} \mathbb{P}(W_t^m(X) \ge b)
$$

---

## 8. Usage Examples

### 8.1 Basic Simulation

```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.portfolio import Account
from finopt.src.model import FinancialModel
import numpy as np

# Setup
income = IncomeModel(
    fixed=FixedIncome(base=1_400_000, annual_growth=0.03),
    variable=VariableIncome(base=200_000, sigma=0.10, seed=42)
)
accounts = [
    Account.from_annual("Emergency", 0.04, 0.05),
    Account.from_annual("Housing", 0.07, 0.12)
]
model = FinancialModel(income, accounts)

# Simulate with 60-40 allocation
X = np.tile([0.6, 0.4], (24, 1))
result = model.simulate(T=24, X=X, n_sims=500, seed=42,
                       start=date(2025, 1, 1))

# Analyze
print(result.summary(confidence=0.95))
metrics = result.metrics(account="Emergency")
print(f"Mean Sharpe: {metrics['sharpe'].mean():.3f}")
```

### 8.2 Goal-Driven Optimization

```python
from finopt.src.optimization import SAAOptimizer
from finopt.src.goals import IntermediateGoal, TerminalGoal

# Define goals
goals = [
    IntermediateGoal(
        month=12, 
        account="Emergency",
        threshold=5_500_000,
        confidence=0.90
    ),
    TerminalGoal(
        account="Housing",
        threshold=20_000_000,
        confidence=0.90
    )
]

# Create optimizer
optimizer = SAAOptimizer(
    n_accounts=model.M,
    tau=0.1,
    objective="terminal_wealth"
)

# Optimize
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    verbose=True
)

print(f"Optimal horizon: T*={result.T} months")
print(f"Feasible: {result.feasible}")
print(result.summary())
```

### 8.3 Multi-Goal Validation

```python
# Simulate with optimal policy (fresh scenarios)
sim_result = model.simulate_from_optimization(
    result, 
    n_sims=1000, 
    seed=999
)

# Verify goal satisfaction
status = model.verify_goals(sim_result, goals)

for goal, metrics in status.items():
    print(f"\nGoal: {goal}")
    print(f"  Satisfied: {metrics['satisfied']}")
    print(f"  Violation rate: {metrics['violation_rate']:.2%}")
    print(f"  Required rate: {metrics['required_rate']:.2%}")
    print(f"  Margin: {metrics['margin']:.4f}")
    if not metrics['satisfied']:
        print(f"  Median shortfall: ${metrics['median_shortfall']:,.0f}")
```

### 8.4 Strategy Comparison

```python
# Define multiple strategies
X_conservative = np.tile([0.9, 0.1], (24, 1))  # 90% emergency
X_balanced = np.tile([0.6, 0.4], (24, 1))      # 60-40
X_aggressive = np.tile([0.3, 0.7], (24, 1))    # 30% emergency

# Simulate each
r1 = model.simulate(T=24, X=X_conservative, n_sims=500, seed=42)
r2 = model.simulate(T=24, X=X_balanced, n_sims=500, seed=42)
r3 = model.simulate(T=24, X=X_aggressive, n_sims=500, seed=42)

# Compare
model.plot("comparison", results={
    "Conservative": r1,
    "Balanced": r2,
    "Aggressive": r3
})
```

---

## 9. Extensions and Future Work

### 9.1 Implemented Features

✅ **Multi-account portfolios** with correlated returns  
✅ **Intermediate and terminal goals** with chance constraints  
✅ **Sigmoid-smoothed SAA** for gradient-based optimization  
✅ **Intelligent horizon estimation** for terminal-only goals  
✅ **Warm start** for faster bilevel convergence  
✅ **Affine wealth** for analytical gradients  
✅ **Calendar alignment** for seasonality and raises  
✅ **Seed propagation** for reproducibility  

### 9.2 Potential Extensions

**Optimization:**
- 🔄 **CVaR implementation** (CVXPY-based, convex formulation)
- 🔄 **Robust optimization** (worst-case performance over scenarios)
- 🔄 **Multi-period rebalancing** (time-varying $x_t^m$ with turnover penalty)
- 🔄 **Dynamic programming** (Bellman recursion for complex constraints)

**Portfolio features:**
- 🔄 **Transaction costs** ($\kappa \|\Delta x_t\|_1$ friction terms)
- 🔄 **Tax-aware optimization** (capital gains, withdrawal timing)
- 🔄 **Minimum balance constraints** ($W_t^m \geq W_{\min}^m$)
- 🔄 **Leverage constraints** (short-selling, margin limits)

**Income modeling:**
- 🔄 **Multi-source income** (multiple jobs, rental, dividends)
- 🔄 **Income shocks** (unemployment, health events)
- 🔄 **Non-Gaussian noise** (fat tails, asymmetry)

**Risk management:**
- 🔄 **Downside protection** (VaR/CVaR constraints)
- 🔄 **Path-dependent goals** (average wealth, peak wealth)
- 🔄 **Correlation uncertainty** (robust correlation estimation)

**Performance:**
- 🔄 **GPU acceleration** (CuPy for large-scale Monte Carlo)
- 🔄 **Sparse factorization** (memory-efficient $F$ storage)
- 🔄 **Parallel simulation** (multi-process scenario generation)

---

## 10. References and Resources

### Mathematical Foundations
- Rockafellar & Uryasev (2000), "Optimization of conditional value-at-risk"
- Luedtke & Ahmed (2008), "A sample approximation approach for optimization with probabilistic constraints"
- Nemirovski & Shapiro (2006), "Convex approximations of chance constrained programs"

### Implementation
- GitHub: `github.com/maxliionel/finopt`
- Documentation: Full API docs in module docstrings
- Tests: Comprehensive unit and integration tests

### Related Tools
- CVXPY: Convex optimization modeling language
- Scipy: Scientific computing (optimize.minimize with SLSQP)
- NumPy/Pandas: Numerical computing and data analysis

---

**End of Framework Document**

---

## Appendix: Quick Reference

### Class Hierarchy
```
IncomeModel
    ├─ FixedIncome
    └─ VariableIncome

Portfolio
    └─ Account

ReturnModel (uses Account metadata)

FinancialModel (facade)
    ├─ income: IncomeModel
    ├─ portfolio: Portfolio
    └─ returns: ReturnModel

AllocationOptimizer (abstract)
    ├─ SAAOptimizer (sigmoid-smoothed)
    └─ CVaROptimizer (stub)

GoalSeeker (bilevel solver)
    └─ optimizer: AllocationOptimizer
```

### Key Type Signatures
```python
# Simulation
SimulationResult = model.simulate(
    T: int,
    X: np.ndarray,  # (T, M)
    n_sims: int,
    seed: Optional[int],
    start: Optional[date]
) -> SimulationResult

# Optimization
OptimizationResult = model.optimize(
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    optimizer: AllocationOptimizer,
    T_max: int,
    n_sims: int,
    seed: Optional[int],
    start: Optional[date]
) -> OptimizationResult

# Validation
Dict[Goal, Dict[str, float]] = model.verify_goals(
    result: Union[SimulationResult, OptimizationResult],
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    start: Optional[date]
)
```

### Common Patterns
```python
# Pattern 1: Simulation-only workflow
model = FinancialModel(income, accounts)
X = define_allocation_policy(T, M)
result = model.simulate(T, X, n_sims=500, seed=42)
model.plot("wealth", result=result)

# Pattern 2: Optimization workflow
goals = define_goals()
optimizer = SAAOptimizer(n_accounts=M, tau=0.1)
opt_result = model.optimize(goals, optimizer, T_max=120, seed=42)
sim_result = model.simulate_from_optimization(opt_result, n_sims=1000)
status = model.verify_goals(sim_result, goals)

# Pattern 3: Comparison workflow
results = {name: model.simulate(T, X_i, ...) for name, X_i in strategies.items()}
model.plot("comparison", results=results)
```