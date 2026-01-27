# `optimization` — Convex Programming for Goal-Driven Portfolios

> **Core idea:** Transform financial goals into convex optimization problems via CVaR reformulation, searching over horizons to find minimum feasible time and optimal allocations.

---

## Philosophy and Role

### Separation of Concerns

- **`income.py`** → Contribution scenarios $A_t$
- **`portfolio.py`** → Wealth dynamics $W_t^m(X)$ via affine representation
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
W_{t+1}^m = \big(W_t^m + A_t x_t^m\big)(1 + R_t^m)
$$

**Closed-form (affine representation):**

$$
\boxed{
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s \, x_s^m \, F_{s,t}^m
}
$$

where accumulation factor:

$$
F_{s,t}^m := \prod_{\tau=s+1}^{t} (1 + R_\tau^m)
$$

**Key consequences:**
- **Affinity:** $W_t^m(X)$ linear in $X$ → enables convex programming
- **Gradient:** $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ → analytical derivatives
- **Efficiency:** $O(1)$ wealth evaluation, no recursion

### Allocation Simplex

Decision variables satisfy budget constraint:

$$
\mathcal{X}_T = \left\{ X \in \mathbb{R}^{T \times M} : x_t^m \geq 0, \; \sum_{m=1}^M x_t^m = 1, \; \forall t \right\}
$$

Cartesian product of $T$ probability simplices: $\mathcal{X}_T = \Delta^{M-1} \times \cdots \times \Delta^{M-1}$.

---

## Goal Framework

### Goal Primitives

**IntermediateGoal** (fixed time $t$):
$$
\mathbb{P}(W_t^m(X) \geq b_t) \geq 1 - \varepsilon_t
$$

**TerminalGoal** (variable horizon $T$):
$$
\mathbb{P}(W_T^m(X) \geq b) \geq 1 - \varepsilon
$$

**Example:**
```python
from datetime import date

goals = [
    IntermediateGoal(date=date(2026, 1, 1), account="Emergency",
                     threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing",
                 threshold=20_000_000, confidence=0.90)
]
```

### GoalSet Algebra

Partition: $\mathcal{G} = \mathcal{G}_{\text{int}} \cup \mathcal{G}_{\text{term}}$

**Minimum horizon constraint:**
$$
T \geq T_{\min} := \max_{g \in \mathcal{G}_{\text{int}}} t_g
$$

**Heuristic for terminal-only goals:**

Uses conservative accumulation analysis with account-specific returns:

$$
T_{\text{start}} = \max_{g \in \mathcal{G}_{\text{term}}} \left\lceil \frac{b_g - W_0^m (1+\mu_m)^{T_{\min}}}{A_{\text{avg}} \cdot \alpha \cdot (1+\mu_m - \sigma_m)} \right\rceil
$$

where:
- $\mu_m$, $\sigma_m$ from `Account.monthly_return` and `Account.monthly_volatility`
- $\alpha$ = safety margin (default 0.75)
- $A_{\text{avg}}$ = average monthly contribution

**Implementation:** `GoalSet.estimate_minimum_horizon(monthly_contribution, accounts, safety_margin=0.75)`

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
& \mathbb{P}\big(W_T^m(X) \geq b^m\big) \geq 1-\varepsilon^m, \; \forall g \in \mathcal{G}_{\text{term}}
\end{aligned} \right\}
$$

**Inner problem** (fixed $T$):
$$
\max_{X \in \mathcal{F}_T} f(X)
$$

### Implemented Objectives

All objectives exploit affine wealth $W_t^m(X) = b + \Phi X$ for convexity. The following objectives are currently implemented in `CVaROptimizer`:

**1. risky** (linear program - fastest):
$$
f(X) = \mathbb{E}\left[\sum_{m=1}^M W_T^m(X)\right]
$$
Maximizes expected total terminal wealth. Pure wealth accumulation without risk considerations.

**2. balanced** (turnover minimization):
$$
f(X) = -\sum_{t=1}^{T-1} \sum_{m=1}^M (x_{t+1,m} - x_t^m)^2
$$
Minimizes portfolio rebalancing via squared L2 penalty on allocation changes. Negative sign converts minimization to maximization problem for solver.

**3. risky_turnover** (wealth with turnover penalty):
$$
f(X) = \mathbb{E}\left[\sum_{m=1}^M W_T^m\right] - \lambda \sum_{t=1}^{T-1} \sum_{m=1}^M (x_{t+1,m} - x_t^m)^2
$$
Balances wealth accumulation against transaction costs. Parameter $\lambda$ controls penalty strength (typical: 0.1-1.0 for normalized allocations, 1000-50000 for monetary scale).

**4. conservative** (mean-standard deviation):
$$
f(X) = \mathbb{E}[W_T] - \lambda \cdot \text{Std}(W_T)
$$
Risk-averse objective penalizing volatility. Uses standard deviation (not variance) for intuitive scaling. Parameter $\lambda$ controls risk aversion (typical: 0.1-1.0).

**Note:** While variance $\text{Var}(W_T)$ would be quadratic convex, the code uses standard deviation $\text{Std}(W_T)$ which is concave. This maintains the correct optimization direction (maximizing a concave function is convex) but differs from classical mean-variance formulations.

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

**Affine wealth construction:**
```python
def build_wealth_affine(t, m):
    """W[:,t,m] = b + Φ @ X[:t,m]"""
    b = W0[m] * F[:, 0, t, m]  # Constant term
    if t == 0:
        return b
    Phi = A[:, :t] * F[:, :t, t, m]  # Coefficient matrix
    return b + Phi @ X[:t, m]
```

### DCP Compliance (Disciplined Convex Programming)

**Variance formulation (when used):**

$$
\text{Var}(W) = \frac{1}{N}\sum_{i=1}^N (W_i - \bar{W})^2
$$

Implemented via `cp.sum_squares`:
```python
mean_wealth = cp.sum(W_T_total) / n_sims
variance = cp.sum_squares(W_T_total - mean_wealth) / n_sims
```

**Convexity:** Sum of squares of affine expressions is convex (SOC representable).

**Turnover formulation:**

Squared L2 norm is convex:
```python
turnover = cp.sum_squares(X[1:, :] - X[:-1, :])
```

Equivalent to $\sum_{t,m} (x_{t+1,m} - x_t^m)^2$.

---

## Implementation Architecture

### Class Hierarchy

```
AllocationOptimizer (ABC)
    ├─ solve(T, A, R, W0, goal_set, ...) → OptimizationResult
    ├─ _check_feasibility(...) → bool
    ├─ _compute_objective(W, X, T, M) → float
    ├─ _objective_risky(W, X, T, M)
    ├─ _objective_balanced(W, X, T, M)
    ├─ _objective_risky_turnover(W, X, T, M)
    └─ _objective_conservative(W, X, T, M)

CVaROptimizer(AllocationOptimizer)
    ├─ cp: CVXPY module
    ├─ objective: str ∈ {"risky", "balanced", "risky_turnover", "conservative"}
    └─ solve(...) → OptimizationResult

GoalSeeker
    ├─ optimizer: AllocationOptimizer
    ├─ seek(..., search_method="binary") → OptimizationResult
    ├─ _linear_search(...) → OptimizationResult
    └─ _binary_search(...) → OptimizationResult
```

### Key Design Changes

**GoalSet passed explicitly:** Caller creates `GoalSet` once before optimization loop, avoiding redundant validation:

```python
# In GoalSeeker.seek()
goal_set = GoalSet(goals, accounts, start_date)

for T in range(T_start, T_max + 1):
    result = optimizer.solve(
        T=T, A=A, R=R, W0=W0,
        goal_set=goal_set,  # Pre-validated, reused
        **solver_kwargs
    )
```

**Separation of responsibilities:**
- `GoalSet`: Validation, account resolution, minimum horizon estimation
- `AllocationOptimizer`: Convex programming, feasibility checking
- `GoalSeeker`: Bilevel search, warm starting

---

## CVaROptimizer.solve() Algorithm

```python
# 1. Validate inputs
if T < goal_set.T_min:
    raise ValueError(f"T={T} < goal_set.T_min")

# 2. Precompute accumulation factors
portfolio = Portfolio(goal_set.accounts)
F = portfolio.compute_accumulation_factors(R)  # (n_sims, T+1, T+1, M)

# 3. Decision variables
X = cp.Variable((T, M), nonneg=True)
gamma = {g: cp.Variable() for g in goals}
z = {g: cp.Variable(n_sims, nonneg=True) for g in goals}

# 4. Affine wealth helper
def build_wealth_affine(t, m):
    b = W0[m] * F[:, 0, t, m]
    if t == 0:
        return b
    Phi = A[:, :t] * F[:, :t, t, m]
    return b + Phi @ X[:t, m]

# 5. Constraints
constraints = [cp.sum(X, axis=1) == 1]  # Simplex

for goal in goal_set.terminal_goals:
    m = goal_set.get_account_index(goal)
    W_T_m = build_wealth_affine(T, m)
    shortfall = goal.threshold - W_T_m
    constraints += [
        z[goal] >= shortfall - gamma[goal],
        gamma[goal] + cp.sum(z[goal])/(goal.epsilon * n_sims) <= 0
    ]

for goal in goal_set.intermediate_goals:
    m = goal_set.get_account_index(goal)
    t = goal_set.get_resolved_month(goal)
    W_t_m = build_wealth_affine(t, m)
    shortfall = goal.threshold - W_t_m
    constraints += [
        z[goal] >= shortfall - gamma[goal],
        gamma[goal] + cp.sum(z[goal])/(goal.epsilon * n_sims) <= 0
    ]

# 6. Objective (dispatch by self.objective)
W_T_total = sum(build_wealth_affine(T, m) for m in range(M))
mean_wealth = cp.sum(W_T_total) / n_sims

if self.objective == "risky":
    objective = cp.Maximize(mean_wealth)
elif self.objective == "conservative":
    lambda_ = self.objective_params.get("lambda", 0.5)
    variance = cp.sum_squares(W_T_total - mean_wealth) / n_sims
    objective = cp.Maximize(mean_wealth - lambda_ * variance)
elif self.objective == "risky_turnover":
    lambda_ = self.objective_params.get("lambda", 15000)
    turnover = cp.sum_squares(X[1:, :] - X[:-1, :]) if T > 1 else 0
    objective = cp.Maximize(mean_wealth - lambda_ * turnover)
elif self.objective == "balanced":
    turnover = cp.sum_squares(X[1:, :] - X[:-1, :]) if T > 1 else 0
    objective = cp.Maximize(-turnover)

# 7. Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS, verbose=verbose, max_iters=max_iters)

# 8. Extract and validate
X_star = X.value
# Project to simplex if needed (numerical tolerance)
for t in range(T):
    if abs(X_star[t, :].sum() - 1.0) > 1e-6:
        X_star[t, :] = np.maximum(X_star[t, :], 0)
        X_star[t, :] /= X_star[t, :].sum()

# 9. Exact feasibility check
feasible = self._check_feasibility(X_star, A, R, W0, portfolio, goal_set)

return OptimizationResult(X=X_star, T=T, feasible=feasible, ...)
```

---

## GoalSeeker Search Strategies

### Linear Search (Safe, Slower)

**Algorithm:**
```python
X_prev = None
for T in range(T_start, T_max + 1):
    result = optimizer.solve(T, A, R, W0, goal_set, X_init=X_prev)
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
    
    # Warm start from previous result
    X_init = None
    if best_result and best_result.T < mid:
        n_extend = mid - best_result.T
        X_init = np.vstack([
            best_result.X,
            np.tile(best_result.X[-1:, :], (n_extend, 1))
        ])
    
    result = optimizer.solve(mid, A, R, W0, goal_set, X_init=X_init)
    
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

### CVXPY Solvers

**ECOS** (default):
- Type: Interior-point (LP/SOCP)
- Speed: Fast (30-80ms typical)
- Stability: Good for well-conditioned problems
- Recommended for most use cases

**SCS** (robust):
- Type: First-order conic solver
- Speed: Moderate
- Stability: Handles ill-conditioned problems
- Use when ECOS fails or numerical issues

**CLARABEL** (modern):
- Type: Interior-point
- Speed: Balanced
- Stability: Good numerical properties
- Alternative to ECOS for difficult problems

### Solver Options

```python
result = optimizer.solve(
    T=24, A=A, R=R, W0=W0, goal_set=goal_set,
    solver='ECOS',
    verbose=True,
    max_iters=10000,
    abstol=1e-7,  # Absolute tolerance
    reltol=1e-6   # Relative tolerance
)
```

### Complexity Analysis

**Variables:** $T \cdot M + G \cdot (1 + N) + f(obj)$
- $T \cdot M$ allocations $X$
- $G$ VaR levels $\gamma_g$
- $G \cdot N$ excess shortfalls $z_g^i$
- $f(obj)$ = objective-dependent (e.g., variance auxiliary variables)

**Constraints:** $T + G \cdot (N + 1) + g(obj)$
- $T$ simplex constraints
- $G \cdot N$ CVaR auxiliary constraints
- $G$ CVaR threshold constraints
- $g(obj)$ = objective-dependent

**Example:** $T=24, M=3, G=3, N=300$
- Variables: $72 + 903 = 975$
- Constraints: $24 + 903 = 927$
- Solve time: 30-80ms (ECOS)

**Interior-point complexity:** $O(n^{3.5})$ where $n$ = number of variables

---

## Numerical Considerations

### Memory Management

Accumulation factor tensor $F \in \mathbb{R}^{N \times (T+1) \times (T+1) \times M}$:

**Size:** $N \cdot T^2 \cdot M \cdot 8$ bytes

**Example:** $N=500, T=120, M=5$ → 14 GB ⚠️

**Mitigation:**
1. Use `Portfolio.compute_accumulation_factors()` with optimized storage
2. Batch scenarios (process $N$ in chunks)
3. On-demand computation in optimizer (trade memory for compute)
4. Reduce $N$ for preliminary runs (100-200 sufficient for testing)

### Simplex Projection

Due to finite solver tolerance, may have $|\sum_m x_t^m - 1| \approx 10^{-8}$:

```python
for t in range(T):
    row_sum = X_star[t, :].sum()
    if abs(row_sum - 1.0) > 1e-6:
        X_star[t, :] = np.maximum(X_star[t, :], 0)  # Clip negatives
        X_star[t, :] /= X_star[t, :].sum()  # Renormalize
```

Applied automatically in `CVaROptimizer.solve()` post-optimization.

### Feasibility Validation

**Two-stage validation:**

1. **CVXPY constraints:** CVaR constraints during solve
   - Convex relaxation via epigraphic formulation
   - Solver tolerances may allow small violations

2. **Exact SAA:** Non-smoothed indicator check via `_check_feasibility()`
   ```python
   violations = W[:, t, m] < goal.threshold
   violation_rate = violations.mean()
   feasible = violation_rate <= goal.epsilon
   ```

Prevents false positives from numerical approximations.

**Implementation pattern:**
```python
# After CVXPY solve
X_star = X.value
feasible = self._check_feasibility(
    X_star, A, R, W0, portfolio, goal_set
)
```

---

## Theoretical Guarantees

**Theorem 1 (Affine Wealth Representation):**  
For any allocation policy $X \in \mathcal{X}_T$ and return realization $\{R_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$
is affine in $X$, enabling convex programming formulations.

**Theorem 2 (CVaR Epigraphic Convexity):**  
The epigraphic reformulation
$$
\text{CVaR}_\varepsilon(b - W) = \inf_{\gamma, z} \left\{ \gamma + \frac{1}{\varepsilon N}\sum_{i=1}^N z^i : z^i \geq b - W^i - \gamma, \, z^i \geq 0 \right\}
$$
defines a convex constraint in $X$ when $W(X)$ is affine.

**Theorem 3 (Objective Convexity):**  
All implemented objectives maintain convexity:
- **risky:** Linear in $X$ → convex
- **balanced:** $-\|X\|_2^2$ → concave (maximizing concave is convex)
- **risky_turnover:** Linear - convex → convex
- **conservative:** Linear - concave → convex (when using Std, not Var)

**Corollary (Global Optimality):**  
CVXPY interior-point solvers return global optimum for convex programs. No local minima exist.

**Theorem 4 (SAA Convergence):**  
Under bounded returns and Lipschitz continuity in $X$:
$$
\frac{1}{N}\sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \geq b\} \xrightarrow{a.s.} \mathbb{P}(W_t^m(X) \geq b)
$$
as $N \to \infty$.

**Theorem 5 (Bilevel Optimality):**  
- **Linear search:** Finds true $T^*$ if inner solver succeeds
- **Binary search:** Finds $T^*$ under monotonicity assumption $\mathcal{F}_T \subseteq \mathcal{F}_{T+1}$

**Proposition (Iteration Reduction):**  
Binary search reduces expected iterations from $O(T^* - T_{\text{start}})$ to $O(\log(T_{\max} - T_{\text{start}}))$.

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
    
  Account 1 (Housing) at month 12:
    Threshold:       10,000,000
    Mean wealth:     11,456,789
    Violation rate: 7.50% (max: 10.00%)
    CVaR value:         -2345.67 (target: ≤ 0)
```

### Common Issues and Solutions

**1. Infeasible problem:**
```
Status: infeasible
Optimal: T*=240 (exhausted T_max)
```
**Diagnosis:** Goals too aggressive for available resources

**Solutions:**
- Increase `T_max` (extend planning horizon)
- Reduce goal thresholds (lower target wealth)
- Increase `epsilon` (lower confidence requirement: 0.90 → 0.85)
- Increase contributions (higher income allocation)
- Verify initial wealth $W_0$ is sufficient

**2. Numerical instability:**
```
Max |Σx_t - 1|: 1.23e-04
⚠️  Minor simplex violations detected
```
**Diagnosis:** Solver tolerance issues

**Solutions:**
- Tighten solver tolerances: `abstol=1e-8, reltol=1e-7`
- Switch solver: `ECOS` → `SCS` (more robust)
- Check for extreme returns in $R$ (e.g., $|R_t^m| > 1$)
- Scale problem: normalize wealth to $[0, 1]$ range

**3. Slow convergence:**
```
Solve time: 12.345s (expected: 0.05-0.10s)
```
**Diagnosis:** Problem size or conditioning issues

**Solutions:**
- Reduce scenarios: $N=500$ → $N=200$ for testing
- Simplify objective: `balanced` → `risky` (LP faster than SOCP)
- Use warm start: pass `X_init` from previous horizon
- Reduce horizon for initial exploration: $T=120$ → $T=60$

**4. Goal validation failure:**
```
Status: optimal (CVXPY)
Feasibility check: FAILED
  Violation rate: 12.5% (max: 10.0%)
```
**Diagnosis:** CVaR approximation vs exact SAA mismatch

**Solutions:**
- Increase scenarios: $N=300$ → $N=500$ (reduces sampling error)
- Tighten CVaR tolerance: adjust solver `abstol`
- Verify accumulation factor computation
- Check for numerical underflow/overflow in $F$

**5. Binary search failure:**
```
Binary search failed: T=87 infeasible
Monotonicity assumption may be violated
```
**Diagnosis:** Non-monotonic feasible set

**Solutions:**
- Switch to linear search: `search_method="linear"`
- Inspect contribution schedule for sudden drops
- Verify goal configuration (check intermediate goals)
- Plot feasibility vs $T$ to identify non-monotonicity

---

## Usage Examples

### Basic Optimization

```python
from finopt.src.optimization import CVaROptimizer, GoalSeeker
from finopt.src.goals import TerminalGoal
from datetime import date

# Setup
accounts = [
    Account.from_annual("Emergency", 0.04, 0.05, initial_wealth=1_000_000),
    Account.from_annual("Housing", 0.07, 0.12, initial_wealth=500_000)
]
W0 = np.array([1_000_000, 500_000])

goals = [
    TerminalGoal(account="Emergency", threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing", threshold=20_000_000, confidence=0.90)
]

# Optimizer with terminal wealth objective
optimizer = CVaROptimizer(
    n_accounts=len(accounts),
    objective='risky',
    account_names=[a.name for a in accounts]
)

# Generators for scenarios
def A_gen(T, n, s):
    return model.income.contributions(T, n_sims=n, seed=s, output="array")

def R_gen(T, n, s):
    return model.returns.generate(T, n_sims=n, seed=s)

# Bilevel search
seeker = GoalSeeker(optimizer, T_max=120, verbose=True)
result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    W0=W0,
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    search_method="binary"
)

print(f"Optimal horizon: T*={result.T} months")
print(f"Feasible: {result.feasible}")
```

### Custom Objective with Turnover Penalty

```python
optimizer = CVaROptimizer(
    n_accounts=3,
    objective='risky_turnover',
    objective_params={'lambda': 10000},  # Turnover penalty
    account_names=["Emergency", "Housing", "Retirement"]
)

result = optimizer.solve(
    T=24,
    A=A_scenarios,
    R=R_scenarios,
    W0=np.array([1e6, 0.5e6, 2e6]),
    goal_set=goal_set,
    solver='ECOS',
    verbose=True
)
```

### Post-Optimization Validation

```python
# Simulate with optimal policy
sim_result = model.simulate(
    T=result.T,
    X=result.X,
    n_sims=1000,  # Higher fidelity
    seed=999
)

# Validate goals
status = result.validate_goals(sim_result)

for goal, metrics in status.items():
    print(f"{goal.account}:")
    print(f"  Violation rate: {metrics['violation_rate']:.2%}")
    print(f"  Required confidence: {goal.confidence:.2%}")
    print(f"  Status: {'✓ PASS' if metrics['satisfied'] else '✗ FAIL'}")
```

---

## References

**Rockafellar, R.T. and Uryasev, S. (2000).** Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.

**Markowitz, H. (1952).** Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.