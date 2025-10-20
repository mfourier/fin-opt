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
goals = [
    IntermediateGoal(month=12, account="Emergency", 
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
T_{\text{start}} = \max_{g \in \mathcal{G}_{\text{term}}} \left\lceil \frac{b_g - W_0^m (1+\mu_m)^{T_{\min}}}{A_{\text{avg}} \cdot 0.75 \cdot (1+\mu_m - \sigma_m)} \right\rceil
$$

where $\mu_m$, $\sigma_m$ from `Account` objects, safety margin 0.75.

**Implementation:** `GoalSet.estimate_minimum_horizon(accounts=...)`

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

### Inner Problem Objectives

All objectives exploit affine wealth $W_t^m(X) = b + \Phi X$ for convexity:

**1. terminal_wealth** (linear program):
$$
f(X) = \mathbb{E}\left[\sum_{m=1}^M W_T^m(X)\right]
$$

**2. min_cvar** (risk-averse):
$$
f(X) = -\sum_{g \in \mathcal{G}} \text{CVaR}_{\varepsilon_g}(\text{threshold}_g - W_t^m(X))
$$

**3. low_turnover** (L1 penalty):
$$
f(X) = \mathbb{E}[W_T] - \lambda \sum_{t,m} |x_{t+1,m} - x_t^m|
$$

**4. risk_adjusted** (mean-variance):
$$
f(X) = \mathbb{E}[W_T] - \lambda \cdot \text{Var}(W_T)
$$

**5. balanced** (multi-objective):
$$
f(X) = \mathbb{E}[W_T] - \lambda_r \cdot \text{Var}(W_T) - \lambda_t \cdot \|\Delta X\|_1
$$

**6. min_variance** (Markowitz):
$$
\min \text{Var}(W_T) \quad \text{s.t.} \quad \mathbb{E}[W_T] \geq \text{target}
$$

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
    b = W0[m] * F[:, 0, t, m]
    if t == 0:
        return b
    Phi = A[:, :t] * F[:, :t, t, m]
    return b + Phi @ X[:t, m]
```

### DCP Compliance (Disciplined Convex Programming)

**Variance formulation:**

Use $\text{Var}(W) = \mathbb{E}[(W - \mathbb{E}[W])^2]$ via `cp.sum_squares`:

```python
mean_wealth = cp.sum(W_T_total) / n_sims
variance = cp.sum_squares(W_T_total - mean_wealth) / n_sims
```

**Convexity:** Sum of squares of affine expressions is convex (SOC representable).

**Turnover formulation:**

L1 norm is convex:
```python
turnover = cp.norm1(X[1:, :] - X[:-1, :])
```

---

## Implementation Architecture

### Class Hierarchy

```
AllocationOptimizer (ABC)
    ├─ solve(T, A, R, W0, goals, accounts, ...) → OptimizationResult
    ├─ _validate_inputs(...) → GoalSet
    ├─ _check_feasibility(...) → bool
    └─ _compute_objective(...) → float

CVaROptimizer(AllocationOptimizer)
    ├─ cp: CVXPY module
    ├─ objective: str ∈ {terminal_wealth, min_cvar, low_turnover, ...}
    └─ solve() → OptimizationResult

GoalSeeker
    ├─ optimizer: AllocationOptimizer
    ├─ seek(..., search_method="binary") → OptimizationResult
    ├─ _linear_search(...) → OptimizationResult
    └─ _binary_search(...) → OptimizationResult
```

### CVaROptimizer.solve() Algorithm

```python
# 1. Precompute accumulation factors
F = portfolio.compute_accumulation_factors(R)  # (n_sims, T+1, T+1, M)

# 2. Decision variables
X = cp.Variable((T, M), nonneg=True)
gamma = {g: cp.Variable() for g in goals}
z = {g: cp.Variable(n_sims, nonneg=True) for g in goals}

# 3. Affine wealth expressions
def build_wealth_affine(t, m):
    b = W0[m] * F[:, 0, t, m]
    Phi = A[:, :t] * F[:, :t, t, m]
    return b + Phi @ X[:t, m]

# 4. Constraints
constraints = [cp.sum(X, axis=1) == 1]  # Simplex

for goal in goals:
    W_t_m = build_wealth_affine(t, m)
    shortfall = goal.threshold - W_t_m
    constraints += [
        z[goal] >= shortfall - gamma[goal],
        gamma[goal] + cp.sum(z[goal])/(goal.epsilon * n_sims) <= 0
    ]

# 5. Objective (e.g., terminal_wealth)
W_T_total = sum(build_wealth_affine(T, m) for m in range(M))
objective = cp.Maximize(cp.sum(W_T_total) / n_sims)

# 6. Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS)
```

### GoalSeeker Search Strategies

**Linear search** (safe, slower):
```python
for T in range(T_start, T_max + 1):
    result = optimizer.solve(T, A, R, W0, goals, ...)
    if result.feasible:
        return result  # Found T*
```

**Binary search** (faster, assumes monotonicity):
```python
left, right = T_start, T_max
while left < right:
    mid = (left + right) // 2
    result = optimizer.solve(mid, ...)
    if result.feasible:
        right = mid  # Search lower half
    else:
        left = mid + 1  # Search upper half
return result_at_left
```

**Complexity:**
- Linear: $O(T^* - T_{\text{start}})$ iterations
- Binary: $O(\log(T_{\max} - T_{\text{start}}))$ iterations

**Assumption:** Monotonicity $\mathcal{F}_T \subseteq \mathcal{F}_{T+1}$ (feasibility preserved as horizon increases).

---

## Solver Configuration

### CVXPY Solvers

**ECOS** (default):
- Type: Interior-point (LP/SOCP)
- Speed: Fast (30-80ms typical)
- Stability: Good for well-conditioned problems

**SCS** (robust):
- Type: First-order conic solver
- Speed: Moderate
- Stability: Handles ill-conditioned problems

**CLARABEL** (modern):
- Type: Interior-point
- Speed: Balanced
- Stability: Good numerical properties

### Solver Options

```python
optimizer.solve(
    T=24, A=A, R=R, W0=W0, goals=goals,
    accounts=accounts, start_date=date(2025,1,1),
    solver='ECOS',
    verbose=True,
    max_iters=10000,
    abstol=1e-7,  # Absolute tolerance
    reltol=1e-6   # Relative tolerance
)
```

### Complexity Analysis

**Variables:** $T \cdot M + G \cdot (1 + N)$
- $T \cdot M$ allocations
- $G$ VaR levels ($\gamma_g$)
- $G \cdot N$ excess shortfalls ($z_g^i$)

**Constraints:** $T + G \cdot (N + 1)$
- $T$ simplex constraints
- $G \cdot N$ CVaR auxiliary constraints
- $G$ CVaR threshold constraints

**Example:** $T=24, M=3, G=3, N=300$
- Variables: $72 + 903 = 975$
- Constraints: $24 + 903 = 927$
- Solve time: 30-80ms (ECOS)

**Interior-point complexity:** $O(n^{3.5})$ where $n$ = number of variables.

---

## Numerical Considerations

### Memory Management

Accumulation factor tensor $F \in \mathbb{R}^{N \times (T+1) \times (T+1) \times M}$:

**Size:** $N \cdot T^2 \cdot M \cdot 8$ bytes

**Example:** $N=500, T=120, M=5$ → 14 GB ⚠️

**Mitigation:**
1. Use `portfolio.compute_accumulation_factors()` with sparse storage
2. Batch scenarios (process $N$ in chunks)
3. On-demand computation in optimizer (trade memory for compute)

### Simplex Projection

Due to finite solver tolerance, may have $|\sum_m x_t^m - 1| \approx 10^{-8}$:

```python
for t in range(T):
    if abs(X[t, :].sum() - 1.0) > 1e-6:
        X[t, :] = np.maximum(X[t, :], 0)
        X[t, :] /= X[t, :].sum()
```

Applied automatically in `CVaROptimizer.solve()`.

### Feasibility Validation

**Two-stage validation:**

1. **CVXPY constraints:** CVaR constraints during solve (convex relaxation)
2. **Exact SAA:** Non-smoothed indicator check via `_check_feasibility()`

```python
feasible = self._check_feasibility(X_star, A, R, W0, accounts, goal_set)
```

Prevents false positives from numerical tolerances.

---

## Theoretical Guarantees

**Theorem 1 (SAA Convergence):**  
Under bounded returns and Lipschitz continuity in $X$, SAA solution converges to true solution as $N \to \infty$ almost surely.

**Theorem 2 (Convexity):**  
All objectives maintain convexity via:
- Affine wealth: $W(X)$ linear in $X$
- Variance: $\text{Var}(W) = \mathbb{E}[W^2] - \mathbb{E}[W]^2$ is convex quadratic
- L1 norm: $\|\Delta X\|_1$ convex
- CVaR: Epigraphic representation convex

**Corollary (Global Optimality):**  
CVXPY solvers return global optimum for convex programs (no local minima).

**Theorem 3 (Bilevel Optimality):**  
Linear search finds true $T^*$ if inner solver succeeds. Binary search finds $T^*$ under monotonicity assumption.

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
```

### Common Issues

**1. Infeasible problem:**
```
Status: infeasible
```
**Solutions:**
- Increase $T_{\max}$
- Relax goal thresholds
- Increase $\varepsilon$ (lower confidence)

**2. Numerical instability:**
```
Max |Σx_t - 1|: 1.23e-04
```
**Solutions:**
- Tighten solver tolerances (`abstol`, `reltol`)
- Switch solver (ECOS → SCS)
- Check for extreme returns in $R$

**3. Slow convergence:**
```
Solve time: 12.345s
```
**Solutions:**
- Reduce $N$ (scenarios)
- Simplify objective (terminal_wealth faster than balanced)
- Use warm start from previous horizon

---

## References

**Rockafellar, R.T. and Uryasev, S. (2000).** Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.

**Markowitz, H. (1952).** Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.

**Ben-Tal, A. and Nemirovski, A. (1998).** Robust convex optimization. *Mathematics of Operations Research*, 23(4), 769-805.