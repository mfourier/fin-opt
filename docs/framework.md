# FinOpt — Technical Framework

> **Tagline:** Intelligent financial planning through stochastic simulation and convex optimization under uncertainty.

This document describes the **theoretical and technical framework** of **FinOpt**, a modular system that connects **user objectives** (emergency funds, housing, retirement) with **optimal investment strategies** under stochastic income, returns, and withdrawals via chance-constrained optimization.

---

## 0. System Architecture

FinOpt is composed of **nine core modules**:

| Module | Purpose | Key Abstractions |
|--------|---------|------------------|
| **`income.py`** | Cash flow modeling | `FixedIncome`, `VariableIncome`, `IncomeModel` |
| **`portfolio.py`** | Wealth dynamics | `Account`, `Portfolio` (affine wealth executor) |
| **`returns.py`** | Stochastic returns | `ReturnModel` (correlated lognormal) |
| **`goals.py`** | Goal specification | `IntermediateGoal`, `TerminalGoal`, `GoalSet` |
| **`withdrawal.py`** | Cash outflows | `WithdrawalEvent`, `StochasticWithdrawal`, `WithdrawalModel` |
| **`optimization.py`** | Solvers | `CVaROptimizer`, `GoalSeeker` |
| **`model.py`** | Orchestration | `FinancialModel` (unified facade) |
| **`serialization.py`** | Persistence | `save_model`, `load_model`, `save_scenario` |
| **`config.py`** | Configuration | Pydantic configs for type-safe parameters |

**Supporting modules:**

| Module | Purpose |
|--------|---------|
| **`utils.py`** | Rate conversions, formatters, metrics |
| **`exceptions.py`** | Error hierarchy (`FinOptError`, `InfeasibleError`, etc.) |
| **`types.py`** | Type definitions (`MonthlyContributionDict`, etc.) |

**Dependency graph:**
```
model.py (FinancialModel)
    ├─→ income.py (IncomeModel)
    ├─→ portfolio.py (Portfolio)
    ├─→ returns.py (ReturnModel)
    ├─→ withdrawal.py (WithdrawalModel)
    └─→ optimization.py (GoalSeeker)
            ├─→ goals.py (GoalSet)
            └─→ CVaROptimizer

serialization.py ←→ config.py
         ↓
    All modules (to_dict/from_dict)

utils.py ←── All modules
exceptions.py ←── All modules
```

**Design principles:**
- **Loose coupling**: Each module usable independently
- **Lazy imports**: Optimization only loaded when needed (TYPE_CHECKING)
- **Separation of concerns**: Portfolio executes dynamics, doesn't generate returns
- **Reproducibility**: Explicit seed management with automatic propagation
- **Convexity**: CVaR reformulation preserves convexity for global optimality

---

## 1. Income Module

Total monthly income at time $t$ is composed of fixed and variable parts:

$$
Y_t = y_t^{\text{fixed}} + Y_t^{\text{variable}}
$$

**Note:** Either component can be `None` (at least one required).

### 1.1 Fixed Income

The fixed component reflects baseline salary subject to compounded annual growth $g$ and scheduled raises:

$$
y_t^{\text{fixed}} = \text{current\_salary}(t) \cdot (1+m)^{\Delta t}
$$

where $m = (1 + g)^{1/12} - 1$ is the **monthly compounded rate**.

**API:**
```python
fixed = FixedIncome(
    base=1_400_000,                    # Current monthly salary
    annual_growth=0.03,                # 3% annual growth
    salary_raises={                    # Date-based raises
        date(2025, 7, 1): 200_000,
        date(2026, 1, 1): 150_000
    }
)
```

### 1.2 Variable Income

The variable component models irregular income with:

- **Seasonality**: $s \in \mathbb{R}^{12}$ (multiplicative monthly factors)
- **Noise**: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ (Gaussian shocks)
- **Growth**: compounded rate $m$ applied to base income
- **Boundaries**: optional floor and cap constraints

$$
\tilde{Y}_t = \max(\text{floor},\ \mu_t (1 + \epsilon_t)), \quad \text{where } \mu_t = \text{base} \cdot (1 + m)^t \cdot s_{(t \bmod 12)}
$$

**API:**
```python
variable = VariableIncome(
    base=200_000,
    sigma=0.15,
    seasonality=[1.0, 0.95, 1.05, ...],  # 12-month cycle
    floor=50_000,
    cap=400_000,
    annual_growth=0.02,
    seed=42
)
```

### 1.3 Contributions

A fraction of income is allocated monthly via calendar-rotating schedules:

$$
A_t = \alpha_{(t \bmod 12)}^{f} \cdot y_t^{\text{fixed}} + \alpha_{(t \bmod 12)}^{v} \cdot Y_t^{\text{variable}}
$$

where $\alpha^f, \alpha^v \in [0,1]^{12}$ control fixed/variable contribution rates (default: 30% fixed, 100% variable).

**Vectorized API:**
```python
income = IncomeModel(fixed=fixed, variable=variable)
income.monthly_contribution = {"fixed": [0.35]*12, "variable": [1.0]*12}

# Single realization
A = income.contributions(months=24, start=date(2025, 1, 1))

# Monte Carlo (vectorized)
A = income.contributions(months=24, start=date(2025, 1, 1), n_sims=500, output="array")
# → shape: (500, 24)
```

---

## 2. Portfolio Dynamics

### 2.1 Wealth Evolution with Withdrawals

Multiple accounts $m \in \mathcal{M} = \{1,\dots,M\}$ evolve via:

$$
\boxed{W_{t+1}^m = \big(W_t^m + A_t x_t^m - D_t^m\big)(1 + R_t^m)}
$$

where:
- $W_t^m$ = wealth in account $m$ at start of month $t$
- $A_t x_t^m$ = allocated contribution ($x_t^m$ fraction of total $A_t$)
- $D_t^m$ = withdrawal from account $m$ during month $t$
- $R_t^m$ = stochastic return of account $m$

**Timing convention:** Withdrawal occurs at **start of month** (before returns applied).

**API:**
```python
accounts = [
    Account.from_annual("Conservador", annual_return=0.06,
                        annual_volatility=0.08, initial_wealth=1_000_000,
                        display_name="Fondo Conservador"),
    Account.from_annual("Agresivo", annual_return=0.12,
                        annual_volatility=0.15, initial_wealth=500_000)
]
portfolio = Portfolio(accounts)
```

### 2.2 Allocation Policy

Contributions allocated via decision variables $x_t^m \in [0,1]$ satisfying:

$$
\sum_{m=1}^M x_t^m = 1, \quad x_t^m \ge 0, \quad \forall t
$$

The **allocation simplex** at horizon $T$:

$$
\mathcal{X}_T = \left\{ X \in \mathbb{R}^{T \times M} : x_t^m \ge 0,\ \sum_{m=1}^M x_t^m = 1,\ \forall t \right\}
$$

### 2.3 Affine Wealth Representation

Recursive wealth can be expressed in **closed-form**:

$$
\boxed{W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} \big(A_s x_s^m - D_s^m\big) F_{s,t}^m}
$$

where the **accumulation factor** from month $s$ to $t$ is:

$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)
$$

**Key insight:** Withdrawals $D$ are **parameters** (not decision variables), so wealth remains **affine in $X$**, preserving convexity for optimization.

**Implementation:**
```python
result = portfolio.simulate(A=A, R=R, X=X, D=D, method="affine")
W = result.wealth  # (n_sims, T+1, M)
```

---

## 3. Withdrawal Module

### 3.1 Withdrawal Types

**Scheduled Withdrawals** (deterministic):
```python
event = WithdrawalEvent(
    account="Conservador",
    amount=400_000,
    date=date(2025, 6, 1),
    description="Compra bicicleta"
)
schedule = WithdrawalSchedule(events=[event, ...])
```

**Stochastic Withdrawals** (uncertain):
```python
stochastic = StochasticWithdrawal(
    account="Conservador",
    base_amount=300_000,
    sigma=50_000,
    date=date(2025, 9, 1),
    floor=200_000,
    cap=500_000,
    seed=42
)
```

**Unified Model:**
```python
withdrawals = WithdrawalModel(
    scheduled=schedule,
    stochastic=[stochastic]
)

# Generate withdrawal scenarios
D = withdrawals.to_array(T=36, start_date=date(2025, 1, 1),
                         accounts=accounts, n_sims=500, seed=42)
# → shape: (500, 36, 2)
```

### 3.2 Withdrawal Feasibility

The optimizer adds CVaR constraints to ensure sufficient wealth:

$$
\mathbb{P}(W_t^m \geq D_t^m) \geq 1 - \epsilon
$$

Default: `withdrawal_epsilon=0.05` (95% confidence).

---

## 4. Goals Framework

### 4.1 Goal Types

**Intermediate Goal** (fixed calendar date):
```python
IntermediateGoal(
    account="Conservador",
    threshold=5_500_000,
    confidence=0.90,
    date=date(2026, 1, 1)  # Calendar date (not month offset)
)
```

Mathematical constraint:

$$
\mathbb{P}\big(W_{t}^m \ge b\big) \ge 1-\varepsilon
$$

**Terminal Goal** (variable horizon):
```python
TerminalGoal(
    account="Agresivo",
    threshold=30_000_000,
    confidence=0.85
)
```

Mathematical constraint:

$$
\mathbb{P}\big(W_{T}^m \ge b\big) \ge 1-\varepsilon
$$

where $T$ is the optimization horizon (decision variable).

### 4.2 Goal Set

The **goal set** $\mathcal{G}$ is partitioned into:

$$
\mathcal{G} = \mathcal{G}_{\text{int}} \cup \mathcal{G}_{\text{term}}
$$

**Properties:**
1. **Minimum horizon:** $T \geq T_{\min} := \max_{g \in \mathcal{G}_{\text{int}}} t_g$
2. **Calendar resolution:** Dates → month offsets via `resolve_month(start_date)`
3. **Account mapping:** Names → indices via `GoalSet`

**Utility functions:**
```python
from finopt.goals import check_goals, goal_progress, print_goal_status

# Validate goal satisfaction
metrics = check_goals(result, goals, accounts, start_date)

# Compute VaR-based progress
progress = goal_progress(result, goals, accounts, start_date)

# Pretty-print status
print_goal_status(result, goals, accounts, start_date)
```

---

## 5. Optimization Framework

### 5.1 Bilevel Problem

Find the **minimum time** $T^*$ to achieve all goals:

$$
\boxed{\min_{T \in \mathbb{N}} \;\; T \quad \text{s.t.} \quad \mathcal{F}_T \neq \emptyset}
$$

where the **goal-feasible set** at horizon $T$ is:

$$
\mathcal{F}_T := \left\{ X \in \mathcal{X}_T : \begin{aligned}
& \text{CVaR}_{\varepsilon}(b_t^m - W_t^m(X)) \leq 0, \; \forall g \in \mathcal{G}_{\text{int}} \\
& \text{CVaR}_{\varepsilon}(b^m - W_T^m(X)) \leq 0, \; \forall g \in \mathcal{G}_{\text{term}} \\
& \text{CVaR}_{\epsilon_w}(D_t^m - W_t^m(X)) \leq 0, \; \forall \text{withdrawals}
\end{aligned} \right\}
$$

### 5.2 CVaR Reformulation

**Challenge:** Chance constraint $\mathbb{P}(W \geq b) \geq 1-\varepsilon$ is non-convex.

**Solution:** CVaR reformulation (Rockafellar & Uryasev 2000):

$$
\mathbb{P}(W \geq b) \geq 1-\varepsilon \quad \Longleftrightarrow \quad \text{CVaR}_\varepsilon(b - W) \leq 0
$$

**Epigraphic form (LP-compatible):**

$$
\gamma + \frac{1}{\varepsilon N}\sum_{i=1}^N z_i \leq 0, \quad z_i \geq b - W^{(i)} - \gamma, \quad z_i \geq 0
$$

**Key property:** Preserves **convexity** because $W(X)$ is affine in $X$.

### 5.3 CVaROptimizer

```python
from finopt.optimization import CVaROptimizer

optimizer = CVaROptimizer(
    n_accounts=2,
    objective="balanced",     # "risky", "balanced", "conservative", "risky_turnover"
    solver="CLARABEL",        # Default solver (or "ECOS", "SCS", "MOSEK")
    verbose=True
)
```

**Objectives:**

| Objective | Formula | Type |
|-----------|---------|------|
| `"risky"` | $\mathbb{E}[\sum_m W_T^m]$ | LP |
| `"balanced"` | $-\sum_{t,m}(\Delta x_{t,m})^2$ | QP (turnover penalty) |
| `"conservative"` | $\mathbb{E}[W_T] - \lambda \cdot \text{Std}(W_T)$ | QP |
| `"risky_turnover"` | $\mathbb{E}[W_T] - \lambda \sum(\Delta x)^2$ | QP |

### 5.4 GoalSeeker (Bilevel Solver)

```python
from finopt.optimization import GoalSeeker

seeker = GoalSeeker(
    optimizer,
    T_max=120,
    search_method="binary",  # or "linear"
    verbose=True
)

result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=W0,
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    D_generator=D_gen,           # Optional withdrawals
    withdrawal_epsilon=0.05
)
```

**Search methods:**
- `"binary"`: Binary search over $[T_{\min}, T_{\max}]$ — faster for large ranges
- `"linear"`: Linear search from $T_{\min}$ — finds exact minimum

---

## 6. Integration: FinancialModel

### 6.1 Unified Facade

```python
from finopt.model import FinancialModel

model = FinancialModel(
    income=income,
    accounts=accounts,
    default_correlation=None,  # Return correlation matrix
    enable_cache=True
)
```

### 6.2 Core Methods

#### Simulation
```python
result = model.simulate(
    T=36,
    X=X,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals  # Optional WithdrawalModel
)
# Returns: SimulationResult
```

#### Optimization
```python
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals,
    search_method="binary"
)
# Returns: OptimizationResult
```

#### Re-simulation from Optimization
```python
sim_result = model.simulate_from_optimization(
    opt_result,
    n_sims=1000,
    seed=999
)
```

#### Validation
```python
status = model.verify_goals(result, goals)
```

### 6.3 Seed Propagation

```
User seed
    ├─→ Income: seed
    ├─→ Returns: seed + 1
    └─→ Withdrawals: seed + 2
```

**Rationale:** Statistical independence between income, return, and withdrawal shocks.

---

## 7. Serialization

### 7.1 Model Persistence

```python
from finopt.serialization import save_model, load_model
from pathlib import Path

# Save
save_model(model, Path("config.json"))

# Load
model = load_model(Path("config.json"))
```

### 7.2 Scenario Persistence

```python
from finopt.serialization import save_scenario, load_scenario

save_scenario(
    scenario_name="Plan de Retiro",
    goals=goals,
    path=Path("scenarios/retirement.json"),
    model=model,
    withdrawals=withdrawals,
    start_date=date(2025, 1, 1),
    n_sims=1000,
    seed=42,
    T_max=120
)

scenario = load_scenario(Path("scenarios/retirement.json"))
```

---

## 8. Complete Workflow Example

```python
from datetime import date
from finopt.model import FinancialModel
from finopt.income import FixedIncome, VariableIncome, IncomeModel
from finopt.portfolio import Account
from finopt.goals import IntermediateGoal, TerminalGoal
from finopt.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent
from finopt.optimization import CVaROptimizer

# 1. Define income
income = IncomeModel(
    fixed=FixedIncome(base=1_400_000, annual_growth=0.03),
    variable=VariableIncome(base=200_000, sigma=0.15, seed=42)
)

# 2. Define accounts
accounts = [
    Account.from_annual("Conservador", 0.06, 0.08, initial_wealth=1_000_000),
    Account.from_annual("Agresivo", 0.12, 0.15, initial_wealth=500_000)
]

# 3. Define withdrawals
withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule([
        WithdrawalEvent("Conservador", 5_000_000, date(2027, 1, 1), "Pie departamento")
    ])
)

# 4. Define goals
goals = [
    IntermediateGoal(account="Conservador", threshold=8_000_000,
                     confidence=0.90, date=date(2026, 6, 1)),
    TerminalGoal(account="Agresivo", threshold=30_000_000, confidence=0.85)
]

# 5. Create model
model = FinancialModel(income, accounts)

# 6. Optimize
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
opt_result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42,
    start=date(2025, 1, 1),
    withdrawals=withdrawals,
    search_method="binary"
)

print(f"Optimal horizon: T*={opt_result.T} months")
print(f"Feasible: {opt_result.feasible}")

# 7. Validate with fresh scenarios
sim_result = model.simulate_from_optimization(opt_result, n_sims=1000, seed=999)
status = model.verify_goals(sim_result, goals)

# 8. Visualize
model.plot("wealth", result=sim_result, goals=goals,
           start=date(2025, 1, 1), show_trajectories=True)
```

---

## 9. Key Mathematical Results

**Proposition 1 (Affine Wealth):**
For any allocation policy $X \in \mathcal{X}_T$:

$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} (A_s x_s^m - D_s^m) F_{s,t}^m
$$

is affine in $X$ (withdrawals $D$ are parameters).

**Proposition 2 (CVaR Convexity):**
The constraint $\text{CVaR}_\varepsilon(b - W(X)) \leq 0$ is convex in $X$ when $W(X)$ is affine.

**Proposition 3 (Analytical Gradient):**

$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m, \quad s < t
$$

**Proposition 4 (Global Optimality):**
CVaR reformulation with affine wealth yields a convex program → global optimum guaranteed.

---

## 10. Implementation Details

### 10.1 Wealth Array Indexing

Shape: `(n_sims, T+1, M)` using **start-of-period** semantics:
- `wealth[i, 0, m]` = $W_0^m$ (initial wealth)
- `wealth[i, t, m]` = $W_t^m$ (wealth at start of period $t$)
- `wealth[i, T, m]` = terminal wealth

### 10.2 Month Resolution

Goals and withdrawals use **1-indexed** months:
- `resolve_month(date(2025, 6, 1), start=date(2025, 1, 1))` → 6
- Array index = month - 1 = 5

### 10.3 Memory Management

Accumulation factors: $F \in \mathbb{R}^{N \times (T+1) \times (T+1) \times M}$

**Estimates:**
- $N=500, T=24, M=2$: ~115 MB
- $N=500, T=120, M=5$: ~14 GB

**Mitigation:**
- Use `method="recursive"` for large $T$
- Process in batches

---

## 11. Extensions and Future Work

### Implemented
- Multi-account portfolios with correlated returns
- Intermediate and terminal goals with CVaR constraints
- Withdrawal support (scheduled + stochastic)
- Binary/linear search for horizon optimization
- Calendar alignment for seasonality
- Seed propagation for reproducibility
- JSON serialization for persistence

### Roadmap
- AR(1) temporal dependence in returns
- Transaction costs
- Tax-aware optimization
- GPU acceleration
- Multi-period rebalancing

---

## References

- **CVaR reformulation**: Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
- **Affine wealth dynamics**: Standard MPC technique exploiting linearity
- **Bilevel optimization**: Outer search + inner convex program

---

## Appendix: Quick Reference

### Class Hierarchy
```
IncomeModel
    ├─ FixedIncome (frozen dataclass)
    └─ VariableIncome (frozen dataclass)

Portfolio
    └─ Account

ReturnModel

WithdrawalModel
    ├─ WithdrawalSchedule
    │   └─ WithdrawalEvent (frozen dataclass)
    └─ StochasticWithdrawal (frozen dataclass)

GoalSet
    ├─ IntermediateGoal (frozen dataclass)
    └─ TerminalGoal (frozen dataclass)

CVaROptimizer
GoalSeeker

FinancialModel (facade)

Exceptions:
    FinOptError
    ├─ ConfigurationError
    ├─ ValidationError
    │   ├─ TimeIndexError
    │   └─ AllocationConstraintError
    ├─ OptimizationError
    │   └─ InfeasibleError
    └─ MemoryLimitError
```

### Key Type Signatures
```python
# Simulation
SimulationResult = model.simulate(
    T: int,
    X: np.ndarray,           # (T, M)
    n_sims: int,
    seed: Optional[int],
    start: Optional[date],
    withdrawals: Optional[WithdrawalModel]
)

# Optimization
OptimizationResult = model.optimize(
    goals: List[IntermediateGoal | TerminalGoal],
    optimizer: CVaROptimizer,
    T_max: int,
    n_sims: int,
    seed: Optional[int],
    start: Optional[date],
    withdrawals: Optional[WithdrawalModel],
    search_method: Literal["binary", "linear"]
)

# Validation
Dict[Goal, Dict[str, float]] = model.verify_goals(
    result: SimulationResult | OptimizationResult,
    goals: List[IntermediateGoal | TerminalGoal]
)
```
