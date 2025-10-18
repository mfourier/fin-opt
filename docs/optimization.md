# `optimization` — Mathematical Framework and Implementation

> **Core idea:** Turn **financial goals** into **optimization problems**, searching over time horizons and allocation strategies to determine if, when, and how they can be satisfied.  
> `optimization.py` acts as the **decision layer**: while other modules simulate wealth paths, `optimization.py` asks *what allocations make goals feasible, and in the shortest time possible?*

---

## Philosophy and Role in FinOpt

### Separation of Concerns

- **`income.py`** → How much money is available (cash flow modeling)
- **`portfolio.py`** → How capital grows given contributions and returns (wealth dynamics)
- **`optimization.py`** → Searches over *time* and *allocations* to satisfy goals (decision synthesis)

### Why a Dedicated Optimization Module?

Traditional financial planning answers: *"If I save X per month for Y years, how much will I have?"*

FinOpt inverts this: *"I need Z by a certain time—what's the minimum horizon and optimal allocation to achieve it?"*

This requires:
1. **Goal specification** (what we want)
2. **Uncertainty quantification** (stochastic income and returns)
3. **Chance-constrained optimization** (probabilistic guarantees)
4. **Bilevel search** (minimize time, maximize objective)

---

## Mathematical Foundations

### Wealth Evolution

Multiple accounts $m \in \{1,\dots,M\}$ evolve via:

$$
W_{t+1}^m = \big(W_t^m + A_t x_t^m\big)(1 + R_t^m)
$$

where:
- $W_t^m$ = wealth in account $m$ at month $t$
- $A_t x_t^m$ = allocated contribution ($x_t^m$ fraction of total $A_t$)
- $R_t^m$ = stochastic return of account $m$

**Key insight:** Wealth is **linear in the allocation policy** $X = \{x_t^m\}$ when returns are fixed.

### Allocation Simplex

Decision variables must satisfy budget and non-negativity:

$$
x_t^m \ge 0, \quad \sum_{m=1}^M x_t^m = 1, \quad \forall t
$$

The **feasible allocation set** at horizon $T$ is:

$$
\mathcal{X}_T = \left\{ X \in \mathbb{R}^{T \times M} : 
\begin{aligned}
& x_t^m \ge 0 \\
& \sum_{m=1}^M x_t^m = 1 \\
& \forall t = 0, \dots, T-1
\end{aligned}
\right\}
$$

This is the **Cartesian product of $T$ probability simplices**: $\mathcal{X}_T = \Delta^{M-1} \times \cdots \times \Delta^{M-1}$.

### Affine Wealth Representation

The recursive formula can be rewritten in **closed-form**:

$$
\boxed{
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s \, x_s^m \, F_{s,t}^m
}
$$

where the **accumulation factor** is:

$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)
$$

**Critical consequences:**

1. **Affinity:** $W_t^m(X)$ is affine in $X$ → wealth constraints are linear-affine
2. **Analytical gradient:** $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$ → enables gradient-based optimization
3. **Convexity:** For deterministic constraints $W_t^m(X) \ge b$, feasible set is a convex polytope
4. **Computational efficiency:** Closed-form evaluation without recursive loops

**Proof sketch:**
$$
\begin{aligned}
W_1^m &= (W_0^m + A_0 x_0^m)(1 + R_0^m) \\
&= W_0^m(1 + R_0^m) + A_0 x_0^m (1 + R_0^m) \\
W_2^m &= (W_1^m + A_1 x_1^m)(1 + R_1^m) \\
&= W_0^m(1+R_0^m)(1+R_1^m) + A_0 x_0^m (1+R_0^m)(1+R_1^m) + A_1 x_1^m (1+R_1^m) \\
&= W_0^m F_{0,2}^m + A_0 x_0^m F_{0,2}^m + A_1 x_1^m F_{1,2}^m
\end{aligned}
$$

Induction completes the proof.

---

## Goal Framework

### Goal Primitives

FinOpt supports two goal types:

**Intermediate Goal** (fixed time constraint):
$$
\text{IntermediateGoal}(t, m, b, \varepsilon) \implies \mathbb{P}(W_t^m(X) \ge b) \ge 1 - \varepsilon
$$

- $t$: fixed target month (e.g., $t=12$ for 1-year emergency fund)
- $m$: target account index
- $b$: wealth threshold
- $\varepsilon$: maximum violation probability (e.g., $\varepsilon=0.10$ for 90% confidence)

**Terminal Goal** (variable horizon):
$$
\text{TerminalGoal}(m, b, \varepsilon) \implies \mathbb{P}(W_T^m(X) \ge b) \ge 1 - \varepsilon
$$

- $T$: optimization variable (decision)
- Other parameters same as intermediate

**Example:**
```python
goals = [
    IntermediateGoal(month=12, account="Emergency", 
                    threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing", 
                threshold=20_000_000, confidence=0.90)
]
```

This specifies:
1. Must have ≥5.5M in Emergency at month 12 with 90% probability (fixed constraint)
2. Want ≥20M in Housing at minimum horizon $T^*$ with 90% probability (objective)

### Goal Set Algebra

The **goal set** $\mathcal{G}$ partitions into:

$$
\mathcal{G} = \mathcal{G}_{\text{int}} \cup \mathcal{G}_{\text{term}}
$$

**Key property (minimum horizon constraint):**
$$
T \geq T_{\min} := \max_{g \in \mathcal{G}_{\text{int}}} t_g
$$

Any allocation policy for horizon $T < T_{\min}$ is **automatically infeasible** (cannot satisfy all intermediate goals).

**GoalSet** class provides:
- Account name → index mapping
- Calendar date → month index resolution
- Automatic $T_{\min}$ computation
- Validation of goal consistency

### Horizon Estimation Heuristic

**Problem:** For terminal-only goals ($\mathcal{G}_{\text{int}} = \emptyset$), naive linear search starts at $T=1$, wasting iterations on obviously infeasible horizons.

**Solution:** Conservative lower bound via worst-case accumulation analysis.

**Heuristic formula:**
$$
T_{\text{start}} = \max_{g \in \mathcal{G}_{\text{term}}} \left\lceil \frac{b_g - W_0^m (1 + \mu)^{T_{\min}}}{A_{\text{avg}} \cdot x_{\min}^m \cdot (1 + \mu - \sigma)} \right\rceil
$$

where:
- $A_{\text{avg}}$: average monthly contribution (empirically sampled)
- $\mu, \sigma$: expected return and volatility of account $m$
- $x_{\min}^m$: conservative minimum allocation (default: 0.1)
- Safety margin: multiply by 0.8 (start 20% earlier)

**Intuition:** How many months of contributions (at conservative return) needed to accumulate target wealth?

**Implementation:** `GoalSet.estimate_minimum_horizon()`

---

## Bilevel Optimization

### Problem Statement

Find the **minimum time** $T^*$ to achieve all goals while optimizing objective $f(X)$:

$$
\boxed{
\min_{T \in \mathbb{N}} \;\; T \quad \text{s.t.} \quad \max_{X \in \mathcal{F}_T} f(X) > -\infty
}
$$

where the **goal-feasible set** is:

$$
\mathcal{F}_T := \left\{ X \in \mathcal{X}_T : \begin{aligned}
& \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1-\varepsilon_t^m, \; \forall g \in \mathcal{G}_{\text{int}}, \\
& \mathbb{P}\big(W_T^m(X) \ge b^m\big) \ge 1-\varepsilon^m, \; \forall g \in \mathcal{G}_{\text{term}}
\end{aligned} \right\}
$$

**Decomposition:**

- **Outer problem:** Discrete search $T \in [T_{\text{start}}, T_{\max}]$ for non-empty $\mathcal{F}_T$
- **Inner problem:** For fixed $T$, solve:
$$
\begin{aligned}
\max_{X \in \mathcal{X}_T} \;\; & f(X) \\
\text{s.t.} \;\; & X \in \mathcal{F}_T
\end{aligned}
$$

### Inner Problem Objectives

The objective $f(X)$ is parametrizable:

**1. Terminal wealth (default):**
$$
f(X) = \mathbb{E}\left[\sum_{m=1}^M W_T^m(X)\right]
$$

**2. Low turnover:**
$$
f(X) = \mathbb{E}[W_T] - \lambda \sum_{t=0}^{T-1} \sum_{m=1}^M |x_{t+1,m} - x_t^m|
$$
Penalizes frequent rebalancing with parameter $\lambda > 0$.

**3. Risk-adjusted (mean-variance):**
$$
f(X) = \mathbb{E}[W_T] - \lambda \cdot \text{Std}(W_T)
$$
Sharpe-like objective with risk aversion $\lambda > 0$.

**4. Balanced:**
$$
f(X) = \mathbb{E}[W_T] - \lambda_{\text{risk}} \cdot \text{Std}(W_T) - \lambda_{\text{turn}} \cdot \text{Turnover}(X)
$$

**5. Custom:**
User-provided callable `f(W, X, T, M) → float`.

---

## Chance Constraint Reformulation

### The Challenge

Probabilistic constraints $\mathbb{P}(W_t^m(X) \ge b) \ge 1 - \varepsilon$ involve:

1. **Indicator function:** $\mathbb{1}\{W \ge b\}$ is discontinuous → no gradient
2. **Expectation over scenarios:** $\mathbb{E}[\mathbb{1}\{W \ge b\}] = \mathbb{P}(W \ge b)$ requires integration
3. **Non-convexity:** Even with affine $W(X)$, indicator destroys convexity

### Sample Average Approximation (SAA)

Replace expectation with sample mean over $N$ scenarios $\omega^{(i)}$:

$$
\mathbb{P}(W_t^m(X) \ge b) \ge 1 - \varepsilon \quad \approx \quad \frac{1}{N} \sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \ge b\} \ge 1 - \varepsilon
$$

**Problem:** Still discontinuous (no gradient).

**Theoretical guarantee (Consistency):**  
Under regularity conditions, as $N \to \infty$:
$$
\frac{1}{N}\sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \ge b\} \xrightarrow{a.s.} \mathbb{P}(W_t^m(X) \ge b)
$$

### Sigmoid Smoothing (SAAOptimizer)

**Key idea:** Replace $\mathbb{1}\{z \ge 0\}$ with smooth sigmoid $\sigma(z/\tau)$.

**Sigmoid function:**
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**Smoothed constraint:**
$$
\boxed{
\frac{1}{N} \sum_{i=1}^N \sigma\left(\frac{W_t^m(X; \omega^{(i)}) - b}{\tau}\right) \ge 1 - \varepsilon
}
$$

**Temperature parameter $\tau$:**

| $\tau$ | Approximation | Gradient | Use case |
|--------|---------------|----------|----------|
| 0.01 | $\sigma \approx \mathbb{1}$ (sharp) | Steep, oscillatory | High fidelity, may diverge |
| 0.1 | Balanced | Smooth, stable | **Recommended** |
| 1.0 | $\sigma \approx 0.5$ (loose) | Very smooth | Easy convergence, poor approximation |

**Approximation error bound:**
$$
\left|\sigma\left(\frac{z}{\tau}\right) - \mathbb{1}\{z \ge 0\}\right| \le \frac{1}{2}
$$
with exponential decay in $|z|/\tau$.

### Gradient Computation

**Constraint function:**
$$
c(X) = \frac{1}{N} \sum_{i=1}^N \sigma\left(\frac{W_t^m(X; \omega^{(i)}) - b}{\tau}\right) - (1 - \varepsilon)
$$

**Gradient w.r.t. allocation $x_s^m$:**
$$
\boxed{
\frac{\partial c}{\partial x_s^m} = \frac{1}{N\tau} \sum_{i=1}^N \sigma'\left(z^{(i)}\right) \cdot A_s^{(i)} \cdot F_{s,t}^{m,(i)}
}
$$

where $z^{(i)} = (W_t^m(X; \omega^{(i)}) - b)/\tau$.

**Derivation:**
$$
\begin{aligned}
\frac{\partial c}{\partial x_s^m} &= \frac{1}{N} \sum_{i=1}^N \frac{\partial}{\partial x_s^m} \sigma\left(\frac{W_t^m(X; \omega^{(i)}) - b}{\tau}\right) \\
&= \frac{1}{N} \sum_{i=1}^N \sigma'(z^{(i)}) \cdot \frac{1}{\tau} \cdot \frac{\partial W_t^m(X; \omega^{(i)})}{\partial x_s^m} \\
&= \frac{1}{N\tau} \sum_{i=1}^N \sigma'(z^{(i)}) \cdot A_s^{(i)} F_{s,t}^{m,(i)} \quad \text{(affine wealth gradient)}
\end{aligned}
$$

**Key properties:**
1. **Analytical:** No finite differences needed
2. **Vectorizable:** Batch computation over scenarios
3. **Sparse:** $\partial W_t / \partial x_s = 0$ for $s \ge t$ (causality)

---

## Solution Strategy

### Outer Loop (GoalSeeker)

**Linear search with intelligent start:**

```python
T_start = estimate_horizon(goals, A_generator, W0)  # Skip infeasible T

for T in range(T_start, T_max + 1):
    # Generate scenarios
    A = A_generator(T, n_sims, seed)
    R = R_generator(T, n_sims, seed+1)
    
    # Solve inner problem
    result = optimizer.solve(T, A, R, W0, goals, X_init=X_prev)
    
    # Check feasibility (exact SAA)
    if result.feasible:
        return result  # Found T*
    
    # Warm start for next iteration
    X_prev = extend_policy(result.X)

raise ValueError("No feasible solution in [T_start, T_max]")
```

**Features:**
1. **Intelligent start:** Avoids testing $T=1,2,3,\dots$ when $T^* \gg 1$
2. **Warm start:** Extends $X$ from $T$ to $T+1$ (repeat last row)
3. **Exact validation:** Final check uses non-smoothed indicator (prevents false positives from sigmoid approximation)

### Inner Loop (SAAOptimizer)

**Gradient-based optimization (SLSQP):**

```python
# Decision variables: X ∈ ℝ^(T×M)
X_flat = X.flatten()  # Vectorize for scipy

# Objective: maximize f(X) → minimize -f(X)
def objective(X_flat):
    W = compute_wealth_affine(X_flat, A, R, F, W0)
    return -f(W, X, T, M)

def objective_grad(X_flat):
    # Analytical gradient via affine formula
    grad[s, m] = (A[:, s] * F[:, s, T, m]).mean()
    return -grad.flatten()

# Constraints
constraints = [
    # Simplex: Σ_m x_t^m = 1
    {'type': 'eq', 'fun': simplex_constraint, 'jac': simplex_jacobian},
    
    # Intermediate goals (smoothed)
    {'type': 'ineq', 'fun': intermediate_constraint, 'jac': intermediate_grad},
    
    # Terminal goals (smoothed)
    {'type': 'ineq', 'fun': terminal_constraint, 'jac': terminal_grad}
]

# Solve
result = scipy.optimize.minimize(
    objective, X_flat, method='SLSQP',
    jac=objective_grad, constraints=constraints,
    bounds=[(0, 1)] * (T*M)
)

X_star = result.x.reshape(T, M)
```

**Solver choice:** SLSQP (Sequential Least Squares Programming)
- Handles inequality and equality constraints
- Uses gradients for efficiency
- Converges reliably for smooth problems

---

## Implementation Architecture

### Class Hierarchy

```
AllocationOptimizer (ABC)
    ├─ solve(T, A, R, W0, goals, ...) → OptimizationResult
    ├─ _validate_inputs(...) → GoalSet
    ├─ _check_feasibility(...) → bool
    └─ _compute_objective(...) → float

SAAOptimizer(AllocationOptimizer)
    ├─ tau: float (sigmoid temperature)
    └─ solve() → [scipy.optimize.minimize + sigmoid smoothing]

CVaROptimizer(AllocationOptimizer)  [STUB]
    ├─ lambda_: float (risk aversion)
    ├─ alpha: float (CVaR level)
    └─ solve() → [NotImplementedError - requires CVXPY]

GoalSeeker
    ├─ optimizer: AllocationOptimizer
    └─ seek(...) → OptimizationResult [bilevel search]
```

### OptimizationResult

**Immutable container** holding:
- `X`: optimal allocation policy $(T, M)$
- `T`: optimization horizon
- `objective_value`: $f(X^*)$
- `feasible`: bool (all goals satisfied?)
- `goals`: original specifications
- `solve_time`: seconds
- `n_iterations`: solver iterations
- `diagnostics`: dict (convergence info)

**Methods:**
- `summary()`: human-readable report
- `is_valid_allocation()`: constraint validation
- `validate_goals()`: detailed goal satisfaction metrics

### Usage Pattern

```python
from finopt.src.optimization import SAAOptimizer, GoalSeeker
from finopt.src.goals import IntermediateGoal, TerminalGoal

# 1. Define goals
goals = [
    IntermediateGoal(month=12, account="Emergency", 
                    threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Housing", 
                threshold=20_000_000, confidence=0.90)
]

# 2. Create optimizer
optimizer = SAAOptimizer(
    n_accounts=2,
    tau=0.1,
    objective="terminal_wealth"
)

# 3. Create bilevel solver
seeker = GoalSeeker(optimizer, T_max=120, verbose=True)

# 4. Optimize (via FinancialModel)
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42
)

# 5. Validate
sim_result = model.simulate_from_optimization(result, n_sims=1000, seed=999)
status = model.verify_goals(sim_result, goals)
```

---

## Advanced Topics

### CVaR Reformulation (Future Work)

**Risk-adjusted objective:**
$$
\max \; \mathbb{E}[W_T] - \lambda \cdot \text{CVaR}_{\alpha}(-W_T)
$$

**CVaR auxiliary formulation (Rockafellar & Uryasev 2000):**
$$
\text{CVaR}_{\alpha}(L) = \min_{\xi} \left\{ \xi + \frac{1}{\alpha N} \sum_{i=1}^N \max(L_i - \xi, 0) \right\}
$$

**CVXPY implementation sketch:**
```python
import cvxpy as cp

X = cp.Variable((T, M), nonneg=True)
xi = cp.Variable()
u = cp.Variable(n_sims, nonneg=True)

# Affine wealth (vectorized)
W_T = W0 @ F[:, 0, T, :].T + sum(
    A[:, s, None] * X[s, :] @ F[:, s, T, :].T
    for s in range(T)
)

# Objective
mean_wealth = cp.sum(W_T) / n_sims
cvar = xi + cp.sum(u) / (alpha * n_sims)
objective = cp.Maximize(mean_wealth - lambda_ * cvar)

# Constraints
constraints = [
    u >= -W_T - xi,  # CVaR auxiliary
    cp.sum(X, axis=1) == 1,  # Simplex
    # Goals...
]

prob = cp.Problem(objective, constraints)
prob.solve(solver='ECOS')
```

### Robust Optimization

**Worst-case formulation:**
$$
\max_{X \in \mathcal{X}_T} \min_{\omega \in \Omega} f(X, \omega)
$$

where $\Omega$ is an **uncertainty set** (e.g., ellipsoidal return distribution).

### Multi-Period Rebalancing

Allow $X$ to vary arbitrarily (not constant policy):
$$
x_t^m = g_t(W_0, \dots, W_{t-1}, A_0, \dots, A_{t-1})
$$

Requires dynamic programming or model predictive control.

---

## Numerical Considerations

### Accumulation Factor Computation

**Memory:** $F \in \mathbb{R}^{N \times (T+1) \times (T+1) \times M}$ requires $N \cdot T^2 \cdot M \cdot 8$ bytes.

**Example:** $N=500, T=120, M=5$ → **14 GB** ⚠️

**Mitigation:**
1. Use `method="recursive"` (no $F$ precomputation)
2. Chunk scenarios (batch $N$ into smaller groups)
3. Sparse storage (only needed $F_{s,t}$ pairs)
4. On-the-fly gradient computation

### Solver Stability

**Common issues:**

1. **Infeasible initial guess:**
   - Solution: Use uniform allocation $X_0 = \mathbf{1}/M$
   - Warm start from previous horizon

2. **Gradient explosion:**
   - Solution: Gradient clipping, smaller $\tau$
   - Check $F_{s,t}$ magnitude (clip extreme returns)

3. **Constraint satisfaction:**
   - Solution: Exact SAA validation (non-smoothed)
   - Tighten convergence tolerances

4. **Slow convergence:**
   - Solution: Increase $\tau$ (smoother constraints)
   - Better warm start, reduce $n_{\text{sims}}$ initially

---

## Theoretical Guarantees

**Proposition 1 (SAA Convergence):**  
Under Lipschitz continuity of $W_t^m(X)$ in $X$ and bounded returns, the SAA solution converges to the true solution as $N \to \infty$ with probability 1.

**Proposition 2 (Sigmoid Approximation):**  
For fixed $\tau$, the sigmoid-smoothed solution $X_\tau^*$ approaches the exact SAA solution as $\tau \to 0^+$, but convergence may deteriorate (gradient explosion).

**Proposition 3 (Bilevel Optimality):**  
If the inner problem solver finds a global optimum (guaranteed for convex problems), then GoalSeeker's linear search finds the true $T^*$ (minimum feasible horizon).

**Corollary:** For terminal-only goals with conservative heuristic start, expected iterations ≈ $O(\log T^*)$ vs $O(T^*)$ for naive search.
