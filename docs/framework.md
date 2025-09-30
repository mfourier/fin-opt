# FinOpt — Technical Framework

> **Tagline:** Intelligent financial planning through simulation and optimization under uncertainty.

This document describes the **theoretical and technical framework** of **FinOpt**, a modular system that connects **user objectives** (housing, emergency, retirement) with **investment strategies** under stochastic income and returns.

---

## 0. System Architecture

FinOpt is composed of **four core modules**:

- **`income.py`** → Cash flow modeling (fixed salary + variable income with seasonality/noise)
- **`investment.py`** → Capital accumulation under stochastic returns
- **`optimization.py`** → Goal-driven solvers (min time, min contribution, allocation search)
- **`utils.py`** → Shared utilities (validation, rate conversion, metrics)

---

## 1. Income Module

### 1.1 Fixed Income Stream

Deterministic monthly salary with compounded annual growth and scheduled raises:

$$
y_t^{\text{fixed}} = \begin{cases}
\text{base} \cdot (1+m)^t & \text{no raises} \\
\text{current\_salary}(t) \cdot (1+m)^{\Delta t} & \text{with raises}
\end{cases}
$$

where $m = (1+g_{\text{annual}})^{1/12}-1$ is the monthly growth rate.

**Salary raises** are date-based: $\{(d_k, \Delta_k)\}$ where $d_k$ is a calendar date and $\Delta_k$ is the absolute raise amount applied from that month onward.

### 1.2 Variable Income Stream

Irregular monthly income with seasonality, noise, and guardrails:

$$
\tilde{y}_t^{\text{variable}} = \max\Big(0, \min\big(\text{cap}, \max(\text{floor}, \mu_t \cdot (1 + \epsilon_t))\big)\Big)
$$

where:
- $\mu_t = \text{base} \cdot (1+m)^t \cdot s_{(t \bmod 12)}$ is the seasonal mean
- $s \in \mathbb{R}^{12}_{\ge 0}$ is the seasonality vector (Jan–Dec multiplicative factors)
- $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is Gaussian noise (fraction of mean)
- floor, cap are optional bounds

### 1.3 Total Income and Contributions

Total monthly income:
$$
y_t = y_t^{\text{fixed}} + y_t^{\text{variable}}
$$

Monthly contributions via **rotative 12-month fractions**:
$$
A_t = \alpha_{(t+\text{offset}) \bmod 12}^{\text{fixed}} \cdot y_t^{\text{fixed}} + \alpha_{(t+\text{offset}) \bmod 12}^{\text{variable}} \cdot y_t^{\text{variable}}
$$

where:
- $\alpha^{\text{fixed}}, \alpha^{\text{variable}} \in [0,1]^{12}$ are annual contribution fraction arrays
- offset = $(\text{start\_month} - 1)$ aligns fractions to calendar
- **Default:** $\alpha^{\text{fixed}} = [0.3]_{12}$, $\alpha^{\text{variable}} = [1.0]_{12}$

Contributions are floored at zero: $A_t = \max(0, A_t)$.

---

## 2. Investment Dynamics

### 2.1 Portfolio Evolution

Multiple portfolios $m \in \mathcal{M} = \{1,\dots,M\}$ evolve via:

$$
W_{t+1}^m = \big(W_t^m + A_t^m\big)(1 + R_t^m)
$$

where:
- $W_t^m$ = wealth in portfolio $m$ at month $t$
- $A_t^m$ = contribution to portfolio $m$ at month $t$
- $R_t^m$ = stochastic return of portfolio $m$ at month $t$

### 2.2 Allocation Policy

Contributions are allocated via decision variables $x_t^m \in [0,1]$:

$$
A_t^m = A_t \cdot x_t^m, \quad \sum_{m \in \mathcal{M}} x_t^m = 1, \quad x_t^m \ge 0
$$

**Feasible allocation set** for horizon $T$:

$$
\mathcal{X}_T = \Big\{ X \in \mathbb{R}_{\ge 0}^{T \times M} \;\Big|\; \sum_{m \in \mathcal{M}} x_t^m = 1, \;\forall t \in \{0,\dots,T-1\} \Big\}
$$

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
- $W_t^m(X)$ is **affine** in the allocation policy $X$
- Gradient: $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$
- Enables linear-affine constraint formulation

---

## 3. Financial Goals

### 3.1 Goal Specification

A **goal** is a tuple $(t, m, b_t^m, \varepsilon_t^m)$ specifying:
- $t$ = target month
- $m$ = target portfolio
- $b_t^m$ = wealth threshold
- $\varepsilon_t^m$ = probability tolerance

The **goal set** $\mathcal{G}$ is:

$$
\mathcal{G} = \Big\{(t,m,b_t^m,\varepsilon_t^m) \;\Big|\; \text{require } \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1-\varepsilon_t^m \Big\}
$$

**Examples:**
- Emergency fund: $(12, 1, 2\text{M}, 0.05)$ → 2M in portfolio 1 at month 12 with 95% probability
- Housing: $(24, 2, 12\text{M}, 0.10)$ → 12M in portfolio 2 at month 24 with 90% probability
- Multi-stage emergency: $(12, 1, 2\text{M}, 0.05)$ and $(24, 1, 4\text{M}, 0.05)$ → progressive targets

---

## 4. Optimization Framework

### 4.1 Nested (Bilevel) Problem

Find the **minimum time** $T$ to achieve all goals, while optimizing an objective $f(X)$:

$$
\boxed{
\begin{aligned}
\min_{T \in \mathbb{N}} \;\; T \quad \text{s.t.} \quad & \exists X^\star \in \arg\max_{X \in \mathcal{X}_T} f(X) \\
& \text{with } \mathbb{P}\big(W_t^m(X^\star) \ge b_t^m\big) \ge 1-\varepsilon_t^m, \;\forall (t,m,b_t^m,\varepsilon_t^m) \in \mathcal{G}
\end{aligned}
}
$$

**Outer problem:** discrete search over horizons $T$  
**Inner problem:** convex optimization (or chance-constrained programming) for allocations $X$

### 4.2 Inner Problem (Fixed Horizon)

For a given horizon $T$, solve:

$$
\begin{aligned}
\max_{X \in \mathcal{X}_T} \;\; & f(X) \\
\text{s.t.} \;\; & \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1-\varepsilon_t^m, \quad \forall (t,m,b_t^m,\varepsilon_t^m) \in \mathcal{G}
\end{aligned}
$$

**Objective functions:**
- Expected total wealth: $f(X) = \mathbb{E}\big[\sum_m W_T^m(X)\big]$
- Risk-adjusted return: $f(X) = \text{Sharpe}(X)$ or CVaR-based
- Terminal goal surplus: $f(X) = \mathbb{E}\big[W_T^m(X) - b_T^m\big]$

**Constraint reformulation:**

Probabilistic constraints can be handled via:

1. **CVaR approximation** (convex):
   $$
   \text{CVaR}_{\varepsilon}(b_t^m - W_t^m(X)) \le 0
   $$

2. **Sample Average Approximation** (SAA):
   $$
   \frac{1}{N}\sum_{i=1}^N \mathbb{1}\{W_t^m(X; \omega^{(i)}) \ge b_t^m\} \ge 1-\varepsilon_t^m
   $$

3. **Analytical bounds** (e.g., for log-normal returns):
   $$
   \mathbb{E}[W_t^m(X)] - z_{1-\varepsilon} \cdot \text{Std}[W_t^m(X)] \ge b_t^m
   $$

### 4.3 Solution Strategy

**Outer loop (horizon search):**
```python
for T in range(T_min, T_max + 1):
    X_opt, feasible = solve_inner_problem(T)
    if feasible:
        return T, X_opt
return None  # infeasible
```

**Inner loop (allocation optimization):**
- Convex solver (CVXPY, scipy.optimize) for linear-affine problems
- Monte Carlo sampling for chance constraint evaluation
- Gradient-based methods leverage $\nabla_X W_t^m(X) = [A_s F_{s,t}^m]_{s<t}$

---

## 5. Implementation Notes

### 5.1 Calendar Alignment

- All projections use `start: date` parameter for calendar awareness
- Seasonality rotates via offset = $(\text{start.month} - 1)$
- Salary raises applied at specific dates relative to projection start
- Contribution fractions rotate cyclically to match fiscal year

### 5.2 Stochasticity Control

- `FixedIncome`: deterministic (no seed)
- `VariableIncome`: controlled via `seed` parameter (instance or method level)
- Returns $R_t^m$: seeded random generators or historical bootstrap
- Monte Carlo: use consistent seed across income + returns for reproducibility

### 5.3 Output Formats

All projection methods support flexible output:
- `output="array"`: numpy arrays (computational efficiency)
- `output="series"`: pandas Series with calendar index (reporting)
- `output="dataframe"`: component breakdown (analysis)

---

## 6. Key Mathematical Results

**Proposition 1 (Affine Wealth):**  
For any allocation policy $X \in \mathcal{X}_T$ and return realization $\{R_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$
is affine in $X$.

**Corollary 1 (Linear Constraints):**  
If goals are specified as $W_t^m(X) \ge b_t^m$ (deterministic), the feasible allocation set is a convex polytope.

**Proposition 2 (Gradient):**  
The sensitivity of wealth to allocation at month $s$ is:
$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m, \quad s < t
$$

**Corollary 2 (Monotonicity):**  
If $F_{s,t}^m > 0$ (positive returns), then $W_t^m(X)$ is strictly increasing in $x_s^m$.

---

## 7. Extensions

- **Multi-period rebalancing:** Allow $x_t^m$ to vary by month
- **Transaction costs:** Add friction terms $\kappa \|\Delta x_t\|_1$
- **Tax-aware optimization:** Incorporate capital gains, withdrawal timing
- **Robust optimization:** Worst-case performance over return scenarios
- **Dynamic programming:** Optimize allocation via Bellman recursion for complex constraints

---

**End of Framework Document**