# `optimization` — Convex Programming for Goal-Driven Portfolios

> **Core idea:** Transform financial goals into convex optimization problems via CVaR reformulation, searching over horizons to find minimum feasible time and optimal allocations.

---

## Philosophy and Role

### Separation of Concerns

- **`income.py`** → Contribution scenarios $A_t$
- **`portfolio.py`** → Wealth dynamics $W_t^m(X)$ via affine representation
- **`goals.py`** → Goal specifications as chance constraints
- **`optimization.py`** → Decision synthesis: minimize $T$, optimize allocations $X$

### Inversion of Traditional Planning

**Traditional:** Given savings $X$ and horizon $T$, compute terminal wealth
**FinOpt:** Given wealth goals, find minimum $T^*$ and optimal $X^*$

Requires:
1. Chance-constrained formulation: $\mathbb{P}(W_t^m \geq b) \geq 1-\varepsilon$
2. Convex reformulation via CVaR
3. Bilevel optimization: outer (minimize $T$), inner (convex program)

---

## Mathematical Foundations

### Wealth Evolution

Multiple accounts $m \in \{1,\dots,M\}$:

$$
W_{t+1}^m = \big(W_t^m + A_t x_t^m - D_t^m\big)(1 + R_t^m)
$$

where $D_t^m$ is the withdrawal from account $m$ at time $t$.

**Closed-form (affine representation):**

$$
\boxed{
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} \big(A_s \, x_s^m - D_s^m\big) \, F_{s,t}^m
}
$$

where accumulation factor:

$$
F_{s,t}^m := \prod_{\tau=s+1}^{t} (1 + R_\tau^m)
$$

**Key consequences:**
- **Affinity:** $W_t^m(X)$ linear in $X$ (withdrawals $D$ are parameters) → enables convex programming
- **Gradient:** $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ → analytical derivatives
- **Efficiency:** $O(1)$ wealth evaluation, no recursion

### Allocation Simplex

Decision variables satisfy budget constraint:

$$
\mathcal{X}_T = \left\{ X \in \mathbb{R}^{T \times M} : x_t^m \geq 0, \; \sum_{m=1}^M x_t^m = 1, \; \forall t \right\}
$$

Cartesian product of $T$ probability simplices: $\mathcal{X}_T = \Delta^{M-1} \times \cdots \times \Delta^{M-1}$.

---

## Bilevel Optimization

### Problem Statement

$$
\boxed{
\min_{T \in \mathbb{N}} \;\; T \quad \text{s.t.} \quad \mathcal{F}_T \neq \emptyset
}
$$

where goal-feasible set:

$$
\mathcal{F}_T := \left\{ X \in \mathcal{X}_T : \begin{aligned}
& \mathbb{P}\big(W_t^m(X) \geq b_t^m\big) \geq 1-\varepsilon_t^m, \; \forall g \in \mathcal{G}_{\text{int}} \\
& \mathbb{P}\big(W_T^m(X) \geq b^m\big) \geq 1-\varepsilon^m, \; \forall g \in \mathcal{G}_{\text{term}} \\
& \mathbb{P}\big(W_t^m(X) \geq D_t^m\big) \geq 1-\delta, \; \forall \text{ withdrawals}
\end{aligned} \right\}
$$

**Inner problem** (fixed $T$):
$$
\max_{X \in \mathcal{F}_T} f(X)
$$

### Implemented Objectives

All objectives exploit affine wealth $W_t^m(X) = b + \Phi X$ for convexity:

| Objective | Formula | Type | Use Case |
|-----------|---------|------|----------|
| `"risky"` | $\mathbb{E}\left[\sum_{m} W_T^m\right]$ | LP | Maximum wealth accumulation |
| `"balanced"` | $-\sum_{t,m}(x_{t+1,m} - x_t^m)^2$ | QP | Stable allocations (default) |
| `"conservative"` | $\mathbb{E}[W_T] - \lambda \cdot \text{Var}(W_T)$ | QP | Risk-averse mean-variance |
| `"risky_turnover"` | $\mathbb{E}[W_T] - \lambda \cdot \sum(\Delta x)^2$ | QP | Wealth + stability tradeoff |

**Note:** The `conservative` objective uses variance (not standard deviation) in the CVXPY formulation for DCP compliance.

---

## CVaR Reformulation

### Epigraphic Formulation (Rockafellar & Uryasev 2000)

Transform chance constraint $\mathbb{P}(W \geq b) \geq 1-\varepsilon$ into convex constraint:

$$
\text{CVaR}_\varepsilon(b - W) \leq 0
$$

**Epigraphic representation:**
$$
\text{CVaR}_\alpha(L) = \min_{\gamma, z} \left\{ \gamma + \frac{1}{\alpha N} \sum_{i=1}^N z^i \right\}
$$
subject to:
$$
z^i \geq L^i - \gamma, \quad z^i \geq 0, \quad \forall i
$$

where $L^i = b - W^i$ is shortfall in scenario $i$.

### Convex Program Formulation

**Decision variables:**
- $X \in \mathbb{R}^{T \times M}$: allocations
- $\gamma_g \in \mathbb{R}$: VaR level per goal $g$
- $z_g \in \mathbb{R}_+^N$: excess shortfall per goal $g$

**Constraints:**

1. **Simplex:** $\sum_m x_t^m = 1, \; x_t^m \geq 0$

2. **CVaR (per goal):**
$$
\begin{aligned}
z_g^i &\geq (\text{threshold}_g - W_{t_g}^{m_g,i}(X)) - \gamma_g \\
\gamma_g &+ \frac{1}{\varepsilon_g N} \sum_{i=1}^N z_g^i \leq 0
\end{aligned}
$$

3. **Withdrawal feasibility (per withdrawal):**
$$
\begin{aligned}
z_w^i &\geq (D_t^m - W_t^{m,i}(X)) - \gamma_w \\
\gamma_w &+ \frac{1}{\delta N} \sum_{i=1}^N z_w^i \leq 0
\end{aligned}
$$

---

## Implementation Architecture

### Class Hierarchy

```
OptimizationResult (frozen dataclass)
    ├─ X: np.ndarray (T, M)
    ├─ T: int
    ├─ objective_value: float
    ├─ feasible: bool
    ├─ goals: List[IntermediateGoal | TerminalGoal]
    ├─ goal_set: GoalSet
    ├─ solve_time: float
    ├─ validate_goals(result) → dict
    ├─ is_valid_allocation(tol) → bool
    └─ summary() → str

AllocationOptimizer (ABC)
    ├─ solve(T, A, R, initial_wealth, goal_set, D, ...) → OptimizationResult
    ├─ _check_feasibility(...) → bool
    └─ _compute_objective(W, X, T, M) → float

CVaROptimizer(AllocationOptimizer)
    ├─ cp: CVXPY module
    ├─ objective: str ∈ {"risky", "balanced", "risky_turnover", "conservative"}
    └─ solve(..., D, withdrawal_epsilon, ...) → OptimizationResult

GoalSeeker
    ├─ optimizer: AllocationOptimizer
    ├─ seek(..., D_generator, withdrawal_epsilon, search_method) → OptimizationResult
    ├─ _linear_search(...) → OptimizationResult
    └─ _binary_search(...) → OptimizationResult
```

### Key Design Patterns

**GoalSet passed explicitly:** Caller creates `GoalSet` once before optimization loop:

```python
# In GoalSeeker.seek()
goal_set = GoalSet(goals, accounts, start_date)

for T in range(T_start, T_max + 1):
    result = optimizer.solve(
        T=T, A=A, R=R, initial_wealth=initial_wealth,
        goal_set=goal_set,  # Pre-validated, reused
        D=D,                # Withdrawals (optional)
        withdrawal_epsilon=0.05,
        **solver_kwargs
    )
```

**Separation of responsibilities:**
- `GoalSet`: Validation, account resolution, minimum horizon estimation
- `AllocationOptimizer`: Convex programming, feasibility checking
- `GoalSeeker`: Bilevel search, warm starting

---

## CVaROptimizer API

### Constructor

```python
from finopt.src.optimization import CVaROptimizer

optimizer = CVaROptimizer(
    n_accounts=3,
    objective='balanced',           # Default: turnover minimization
    objective_params={'lambda': 0.5},  # For conservative/risky_turnover
    account_names=['Emergency', 'Housing', 'Retirement']
)
```

### solve() Method

```python
result = optimizer.solve(
    T=24,                          # Horizon (months)
    A=A_scenarios,                 # (n_sims, T) contributions
    R=R_scenarios,                 # (n_sims, T, M) returns
    initial_wealth=np.array([1e6, 0.5e6, 2e6]),  # (M,)
    goal_set=goal_set,             # Pre-validated GoalSet (REQUIRED)
    X_init=None,                   # Warm start (optional, ignored by CVXPY)
    D=withdrawal_matrix,           # (T, M) or (n_sims, T, M) withdrawals
    withdrawal_epsilon=0.05,       # 95% confidence for withdrawals
    solver='CLARABEL',             # Solver: CLARABEL, ECOS, SCS
    verbose=True,
    max_iters=10000,
    abstol=1e-7,
    reltol=1e-6
)
```

### Withdrawal Support

Withdrawals $D$ can be:
- **Deterministic:** Shape `(T, M)` — same withdrawal across all scenarios
- **Stochastic:** Shape `(n_sims, T, M)` — per-scenario withdrawals

**Withdrawal feasibility constraint (Conservative - Option 1):**
```
ℙ(W_t^m ≥ D_t^m) ≥ 1 - withdrawal_epsilon
```

Ensures sufficient wealth **before** each withdrawal, without relying on contributions.

**Implementation:**
```python
# CVaR reformulation for withdrawal constraint
# Shortfall: D_t - W_t (positive = cannot meet withdrawal)
shortfall = D_tm - W_pre_withdrawal

# CVaR epigraphic constraints
z_wd >= shortfall - gamma_wd
gamma_wd + sum(z_wd) / (withdrawal_epsilon * n_sims) <= 0
```

Since $D$ is a **parameter** (not decision variable), convexity is preserved.

---

## GoalSeeker API

### Constructor

```python
from finopt.src.optimization import GoalSeeker

seeker = GoalSeeker(
    optimizer=optimizer,
    T_max=120,       # Maximum search horizon
    verbose=True
)
```

### seek() Method

```python
from datetime import date

# Generator functions for scenarios
def A_gen(T, n_sims, seed):
    return model.income.contributions(T, n_sims=n_sims, seed=seed, output="array")

def R_gen(T, n_sims, seed):
    return model.returns.generate(T, n_sims=n_sims, seed=seed)

# Optional: withdrawal generator
def D_gen(T, n_sims, seed):
    # Example: $500K withdrawal at month 12 from account 0
    D = np.zeros((T, M))
    if T > 12:
        D[12, 0] = 500_000
    return D

result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=np.array([1e6, 0.5e6]),
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    search_method="binary",        # "binary" or "linear"
    D_generator=D_gen,             # Optional withdrawals
    withdrawal_epsilon=0.05,       # 95% confidence
    solver='CLARABEL',
    verbose=True
)
```

---

## Search Strategies

### Linear Search (Safe, Slower)

**Algorithm:**
```python
X_prev = None
for T in range(T_start, T_max + 1):
    result = optimizer.solve(T, A, R, initial_wealth, goal_set, D, X_init=X_prev)
    if result.feasible:
        return result  # Found T*
    # Warm start: extend policy
    if result.X is not None:
        X_prev = np.vstack([result.X, result.X[-1:, :]])
```

**Complexity:** $O(T^* - T_{\text{start}})$ iterations

**Properties:**
- Always finds true $T^*$ if feasible
- No assumptions required
- Warm start accelerates convergence

### Binary Search (Faster, Assumes Monotonicity)

**Algorithm:**
```python
left, right = T_start, T_max
best_result = None

while left < right:
    mid = (left + right) // 2
    result = optimizer.solve(mid, ...)

    if result.feasible:
        best_result = result
        right = mid  # Search lower half
    else:
        left = mid + 1  # Search upper half

return best_result
```

**Complexity:** $O(\log(T_{\max} - T_{\text{start}}))$ iterations

**Assumption:** Monotonicity $\mathcal{F}_T \subseteq \mathcal{F}_{T+1}$

**When to use:**
- Typical financial planning scenarios (safe)
- Contribution schedules are non-decreasing or stable
- Goal structure is well-behaved

**When to avoid:**
- Contributions have sudden drops
- Pathological goal configurations
- Safety-critical applications → use linear search

---

## Solver Configuration

### Available Solvers

| Solver | Type | Speed | Stability | Default |
|--------|------|-------|-----------|---------|
| **CLARABEL** | Interior-point | Balanced | Good | ✓ |
| ECOS | Interior-point (LP/SOCP) | Fast | Good for well-conditioned |
| SCS | First-order conic | Moderate | Handles ill-conditioned |

### Solver Options Mapping

The API uses standard option names that are mapped to solver-specific options:

```python
# Standard options
result = optimizer.solve(
    ...,
    solver='CLARABEL',
    max_iters=10000,
    abstol=1e-7,
    reltol=1e-6
)
```

**Option mapping:**

| Standard | ECOS | SCS | CLARABEL |
|----------|------|-----|----------|
| `max_iters` | `max_iters` | `max_iters` | `max_iter` |
| `abstol` | `abstol` | `eps_abs` | `tol_gap_abs` |
| `reltol` | `reltol` | `eps_rel` | `tol_gap_rel` |

### Complexity Analysis

**Variables:** $T \cdot M + G \cdot (1 + N) + W \cdot (1 + N)$
- $T \cdot M$ allocations $X$
- $G$ VaR levels $\gamma_g$ + $G \cdot N$ excess shortfalls $z_g^i$
- $W$ withdrawal constraints (if applicable)

**Example:** $T=24, M=3, G=3, N=300, W=2$
- Variables: $72 + 903 + 602 = 1577$
- Solve time: 30-100ms (CLARABEL)

---

## OptimizationResult

### Attributes

```python
@dataclass(frozen=True)
class OptimizationResult:
    X: np.ndarray              # Optimal allocation policy (T, M)
    T: int                     # Horizon
    objective_value: float     # f(X*)
    feasible: bool             # All goals satisfied?
    goals: List[...]           # Original goal specs
    goal_set: GoalSet          # Validated collection
    solve_time: float          # Solver time (seconds)
    n_iterations: Optional[int]
    diagnostics: Optional[dict]
```

### Methods

**summary()** — Human-readable output:
```python
print(result.summary())
# OptimizationResult(
#   Status: ✓ Feasible
#   Horizon: T=24 months
#   Objective: 11234567.89
#   Goals: 3 (1 intermediate, 2 terminal)
#   Solve time: 0.342s
#   Iterations: N/A
# )
```

**validate_goals(result)** — Post-simulation validation:
```python
sim_result = model.simulate(T=result.T, X=result.X, n_sims=1000, seed=999)
status = result.validate_goals(sim_result)

for goal, metrics in status.items():
    print(f"{goal.account}: {metrics['violation_rate']:.2%} violations")
```

**is_valid_allocation(tol)** — Check simplex constraints:
```python
if not result.is_valid_allocation(tol=1e-6):
    print("Warning: allocation violates simplex constraints")
```

---

## Diagnostics and Debugging

### Verbose Output

Enable `verbose=True` for detailed solver information:

```
[CVXPY Solution]
  Status: optimal
  Objective: 21543678.92
  Solve time: 0.067s

[Simplex Validation]
  Max |Σx_t - 1|: 3.45e-09
  X bounds: [0.0000, 1.0000]

[Goal Satisfaction Diagnostics]
  Account 0 (Emergency):
    Threshold:        5,500,000
    Mean wealth:      6,234,567
    Violation rate: 8.20% (max: 10.00%)
    CVaR value:         -1234.56 (target: ≤ 0)

[Withdrawal Feasibility Diagnostics]
  Confidence level: 95% (ε=5%)
  ✓ Period 11, Account 0 (Emergency Fund):
    Withdrawal D_11:      500,000
    Wealth W_11:        1,234,567
    Violation rate:    2.30% (max: 5.00%)
    CVaR value:          -5678.90 (target: ≤ 0)
```

### Common Issues and Solutions

**1. Infeasible problem:**
```
Status: infeasible
```
**Solutions:**
- Increase `T_max` (extend planning horizon)
- Reduce goal thresholds
- Increase `epsilon` (lower confidence: 0.90 → 0.85)
- Increase contributions
- Reduce withdrawal amounts

**2. Numerical instability:**
```
Max |Σx_t - 1|: 1.23e-04
⚠️  Minor simplex violations detected
```
**Solutions:**
- Tighten solver tolerances: `abstol=1e-8, reltol=1e-7`
- Switch solver: `CLARABEL` → `SCS`
- Check for extreme returns

**3. Binary search failure:**
```
InfeasibleError: Binary search failed: T=87 infeasible
Monotonicity assumption may be violated
```
**Solutions:**
- Switch to linear search: `search_method="linear"`
- Inspect contribution schedule for sudden drops
- Verify goal configuration

**4. Withdrawal infeasibility:**
```
Withdrawal constraint violated at T=12
```
**Solutions:**
- Increase `withdrawal_epsilon` (0.05 → 0.10)
- Reduce withdrawal amount
- Extend horizon to allow more accumulation
- Prioritize liquidity in allocation

---

## Complete Examples

### Basic Optimization with Goals

```python
from datetime import date
from finopt.src.optimization import CVaROptimizer, GoalSeeker
from finopt.src.goals import IntermediateGoal, TerminalGoal
from finopt.src.portfolio import Account
import numpy as np

# Setup accounts
accounts = [
    Account.from_annual("Emergency", 0.04, 0.05, initial_wealth=1_000_000),
    Account.from_annual("Housing", 0.07, 0.12, initial_wealth=500_000)
]
initial_wealth = np.array([1_000_000, 500_000])

# Define goals
goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=2_000_000, confidence=0.95),
    TerminalGoal(account="Housing", threshold=20_000_000, confidence=0.90)
]

# Create optimizer
optimizer = CVaROptimizer(
    n_accounts=len(accounts),
    objective='balanced',
    account_names=[a.name for a in accounts]
)

# Generator functions
def A_gen(T, n, s):
    return np.full((n, T), 500_000)  # $500K/month

def R_gen(T, n, s):
    np.random.seed(s)
    return np.random.normal(0.005, 0.02, (n, T, 2))

# Find minimum horizon
seeker = GoalSeeker(optimizer, T_max=120, verbose=True)
result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=initial_wealth,
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    search_method="binary"
)

print(result.summary())
```

### Optimization with Withdrawals

```python
# Define withdrawal schedule
def D_gen(T, n_sims, seed):
    """$1M withdrawal from Housing account at month 24."""
    D = np.zeros((T, 2))
    if T > 24:
        D[24, 1] = 1_000_000  # Account 1 = Housing
    return D

# Seek with withdrawals
result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=initial_wealth,
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    search_method="binary",
    D_generator=D_gen,
    withdrawal_epsilon=0.05  # 95% confidence
)
```

### Custom Objective with Turnover Penalty

```python
optimizer = CVaROptimizer(
    n_accounts=3,
    objective='risky_turnover',
    objective_params={'lambda': 10000},  # Turnover penalty weight
    account_names=["Emergency", "Housing", "Retirement"]
)
```

---

## Theoretical Guarantees

**Theorem 1 (Affine Wealth Representation):**
For any allocation policy $X \in \mathcal{X}_T$, return realization $\{R_t^m\}$, and withdrawal schedule $\{D_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} (A_s x_s^m - D_s^m) F_{s,t}^m
$$
is affine in $X$ (since $D$ is a parameter), enabling convex programming.

**Theorem 2 (CVaR Epigraphic Convexity):**
The epigraphic reformulation defines a convex constraint in $X$ when $W(X)$ is affine.

**Theorem 3 (Global Optimality):**
CVXPY interior-point solvers return global optimum for convex programs. No local minima exist.

**Theorem 4 (Bilevel Optimality):**
- **Linear search:** Finds true $T^*$ if inner solver succeeds
- **Binary search:** Finds $T^*$ under monotonicity assumption $\mathcal{F}_T \subseteq \mathcal{F}_{T+1}$

---

## Exceptions

The module raises `InfeasibleError` when no feasible solution is found:

```python
from finopt.src.exceptions import InfeasibleError

try:
    result = seeker.seek(...)
except InfeasibleError as e:
    print(f"Optimization failed: {e}")
    # Suggestions: increase T_max, relax goals, increase contributions
```

---

## References

**Rockafellar, R.T. and Uryasev, S. (2000).** Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.

**Markowitz, H. (1952).** Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
