# `portfolio` — Philosophy & Role in FinOpt

> **Purpose:** Execute **wealth dynamics** under allocation policies, providing the mathematical foundation for optimization-driven portfolio construction.  
> `portfolio.py` is the **executor layer**: it receives pre-generated contributions `A` (from `income.py`) and returns `R` (from `returns.py`), applies allocation policy `X`, and computes wealth trajectories `W` via recursive or closed-form methods.

---

## Why a dedicated portfolio module?

Financial optimization requires **separating concerns**:
- `income.py` → cash availability
- `returns.py` → stochastic return generation  
- `portfolio.py` → **wealth evolution given allocations**
- `optimization.py` → policy search

This separation enables:
- **Loose coupling:** Portfolio never generates returns (delegated to `ReturnModel`)
- **Optimization-ready:** Affine wealth representation exposes analytical gradients
- **Batch processing:** Vectorized Monte Carlo execution (no Python loops)
- **Dual temporal API:** Seamless monthly ↔ annual parameter conversion

---

## Design principles

1. **Separation of concerns**
   - Portfolio executes dynamics, does NOT generate stochastic processes
   - Returns/contributions are **external inputs** (not embedded models)
   
2. **Vectorized computation**
   - Full batch processing: `(n_sims, T, M)` arrays without Python-level loops
   - Matches `income.py` pattern: `n_sims` parameter for Monte Carlo generation
   
3. **Optimization-first design**
   - Affine wealth formula enables **analytical gradients**: $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$
   - Direct integration with convex solvers (CVXPY, scipy.optimize)
   
4. **Annual parameters by default**
   - User-facing API uses intuitive annual returns/volatility
   - Internal storage in monthly space (canonical form)
   - Properties provide dual temporal views without conversions
   
5. **Calendar alignment inheritance**
   - Contributions `A` from `income.contributions()` are calendar-aware
   - Portfolio preserves temporal semantics (no date handling needed)

---

## The two core surfaces

### 1) `Account`

Metadata container for investment account with **dual temporal parameter access**.

**Parameters:**
- `name`: account identifier (e.g., "Emergency", "Housing")
- `initial_wealth`: starting balance $W_0^m$ (non-negative)
- `return_strategy`: dict with **monthly** parameters `{"mu": float, "sigma": float}`

**Constructor methods (recommended):**
```python
Account.from_annual(name, annual_return, annual_volatility, initial_wealth=0.0)
Account.from_monthly(name, monthly_mu, monthly_sigma, initial_wealth=0.0)
```

**Properties:**
- `monthly_params`: canonical storage format `{"mu": float, "sigma": float}`
- `annual_params`: user-friendly view `{"return": float, "volatility": float}`

**Parameter conversion:**
$$
\begin{aligned}
\mu_{\text{monthly}} &= (1 + r_{\text{annual}})^{1/12} - 1 \quad \text{[geometric compounding]} \\
\sigma_{\text{monthly}} &= \frac{\sigma_{\text{annual}}}{\sqrt{12}} \quad \text{[time scaling]}
\end{aligned}
$$

**Interpretation:** Lightweight struct consumed by `ReturnModel` for return generation. No embedded stochastic processes—pure metadata.

**Key behaviors:**
- Round-trip conversion: `monthly_to_annual(annual_to_monthly(r)) ≈ r`
- Validation: ensures $W_0^m \geq 0$, $\sigma \geq 0$
- String representation uses annual parameters for readability

---

### 2) `Portfolio`

Multi-account wealth dynamics executor with allocation policy support.

**Parameters:**
- `accounts`: list of `Account` objects (metadata only, no return models)

**Method signature:**
```python
def simulate(
    self,
    A: np.ndarray,      # Contributions: (T,) or (n_sims, T)
    R: np.ndarray,      # Returns: (n_sims, T, M)
    X: np.ndarray,      # Allocations: (T, M)
    method: Literal["recursive", "affine"] = "affine"
) -> dict
```

**Core wealth dynamics:**
$$
W_{t+1}^m = \big(W_t^m + A_t^m\big)(1 + R_t^m)
$$
where:
- $A_t^m = x_t^m \cdot A_t$ (allocated contribution)
- $\sum_{m=1}^M x_t^m = 1$, $x_t^m \geq 0$ (budget constraint)

**Returns:**
```python
{
    "wealth": np.ndarray,        # (n_sims, T+1, M)
    "total_wealth": np.ndarray   # (n_sims, T+1)
}
```

**Computation methods:**

1. **Recursive** (default for simulation):
   - Time: $O(n_{\text{sims}} \cdot T \cdot M)$
   - Memory: $O(n_{\text{sims}} \cdot T \cdot M)$
   - Iterative evolution: $W_{t+1} = (W_t + A_t)(1+R_t)$ vectorized over simulations

2. **Affine** (preferred for optimization):
   - Time: $O(T^2 \cdot M \cdot n_{\text{sims}})$ (factor precomputation + application)
   - Memory: $O(n_{\text{sims}} \cdot T^2 \cdot M)$ (stores accumulation factors $F$)
   - Closed-form: wealth is **linear in allocation policy** $X$

**Key behaviors:**
- Automatic broadcasting: deterministic `A` (shape `(T,)`) broadcast to `(n_sims, T)`
- Validation: checks $X \geq 0$, $\sum_m x_t^m = 1$ for all $t$
- Initial wealth: $W_0^m$ from `self.accounts[m].initial_wealth`
- No stochastic generation: assumes `R` is pre-generated externally

---

## Affine wealth representation

### Closed-form formula

The recursive dynamics admit a **closed-form solution**:

$$
\boxed{
W_t^m(X) = W_0^m \cdot F_{0,t}^m + \sum_{s=0}^{t-1} A_s \, x_s^m \, F_{s,t}^m
}
$$

where the **accumulation factor** from month $s$ to $t$ is:

$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)
$$

**Convention:** $F_{s,s}^m = 1$ (no accumulation over empty interval).

### Mathematical properties

1. **Affine in $X$:**
   $$
   W_t^m(\alpha X + \beta Y) = \alpha W_t^m(X) + \beta W_t^m(Y) + \text{const}
   $$
   where const = $W_0^m F_{0,t}^m$ (independent of policy).

2. **Analytical gradient:**
   $$
   \frac{\partial W_t^m}{\partial x_s^m} = A_s \, F_{s,t}^m, \quad s < t
   $$
   Enables gradient-based optimization without numerical differentiation.

3. **Monotonicity:**
   If $F_{s,t}^m > 0$ (positive returns), then $W_t^m$ is strictly increasing in $x_s^m$.

### Accumulation factors computation

**Method signature:**
```python
def compute_accumulation_factors(self, R: np.ndarray) -> np.ndarray
```

**Input:** Returns matrix `R` of shape `(n_sims, T, M)`  
**Output:** Factors tensor `F` of shape `(n_sims, T+1, T+1, M)`

**Algorithm:** Vectorized product over time dimension
```python
F[:, s, t, :] = np.prod(gross_returns[:, s:t, :], axis=1)
# where gross_returns = 1.0 + R
```

**Complexity:**
- Time: $O(n_{\text{sims}} \cdot T^2 \cdot M)$
- Memory: $O(n_{\text{sims}} \cdot T^2 \cdot M)$ floats (8 bytes each)

**Memory estimates:**
- `n_sims=500, T=24, M=2`: ~115 MB
- `n_sims=500, T=120, M=5`: ~14 GB
- `n_sims=1000, T=240, M=10`: ~221 GB (infeasible)

**Warnings:**
For $T > 100$, consider:
- Using `method="recursive"` (no $F$ precomputation)
- Batching simulations (chunking `n_sims`)
- On-the-fly gradient computation (store only needed $F_{s,t}$ pairs)
- Sparse storage for time-specific constraints

---

## Integration with FinOpt pipeline

### Workflow

```python
# 1. Define accounts (annual parameters recommended)
accounts = [
    Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
    Account.from_annual("Housing", annual_return=0.07, annual_volatility=0.12)
]

# 2. Create portfolio executor
portfolio = Portfolio(accounts)

# 3. Generate stochastic inputs externally
income = IncomeModel(fixed=..., variable=...)
returns = ReturnModel(accounts, default_correlation=np.eye(2))

T, n_sims = 24, 500
A = income.contributions(T, start=date(2025,1,1), n_sims=n_sims)  # (500, 24)
R = returns.generate(T, n_sims=n_sims, seed=42)                    # (500, 24, 2)

# 4. Define allocation policy
X = np.tile([0.6, 0.4], (T, 1))  # 60-40 split

# 5. Execute wealth dynamics
result = portfolio.simulate(A=A, R=R, X=X, method="affine")
W = result["wealth"]              # (500, 25, 2)
W_total = result["total_wealth"]  # (500, 25)

# 6. Visualize
portfolio.plot(result=result, X=X, save_path="portfolio_analysis.png")
```

### Data flow

```
income.py          returns.py
    ↓                  ↓
    A                  R          (external generation)
    ↓                  ↓
    └──► portfolio.simulate(A, R, X) ──► W
                       ↑
                       X          (from user or optimizer)
```

### Optimization integration

**Inner problem (fixed horizon $T$):**

$$
\begin{aligned}
\max_{X \in \mathcal{X}_T} \;\; & f(X) = \mathbb{E}\big[\sum_m W_T^m(X)\big] \\
\text{s.t.} \;\; & \mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1 - \varepsilon_t^m, \quad \forall (t,m,b_t^m,\varepsilon_t^m) \in \mathcal{G}
\end{aligned}
$$

**Gradient computation:**
```python
F = portfolio.compute_accumulation_factors(R)  # (n_sims, T+1, T+1, M)

# Gradient of E[W_t^m] w.r.t. x_s^m
grad = (A[:, s, None] * F[:, s, t, :]).mean(axis=0)  # shape: (M,)
```

**Chance constraint via SAA:**
```python
# Constraint: P(W_t^m >= b) >= 1-ε
W_t_m = portfolio._simulate_affine(A, R, X)[:, t, m]  # (n_sims,)
violation_rate = (W_t_m < b).mean()
constraint_satisfied = (violation_rate <= ε)
```

---

## Visualization

**Method signature:**
```python
def plot(
    self,
    result: dict,      # from simulate()
    X: np.ndarray,     # allocation policy
    figsize: tuple = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    return_fig_ax: bool = False,
    show_trajectories: bool = True,
    trajectory_alpha: float = 0.05,
    colors: Optional[dict] = None,
    hist_bins: int = 30,
    hist_color: str = 'mediumseagreen'
)
```

**Panel layout:**
1. **Top-left:** Wealth per account (time series with Monte Carlo trajectories)
2. **Top-right:** Total portfolio wealth + lateral histogram of final wealth distribution
3. **Bottom-left:** Portfolio composition (stacked area chart)
4. **Bottom-right:** Allocation policy heatmap (X matrix visualization)

**Key features:**
- Individual trajectories at low alpha (visual uncertainty quantification)
- Mean trajectories in bold (expected path)
- Final wealth statistics annotation (mean, median, std)
- Lateral histogram for outcome distribution at $T$
- Allocation heatmap with colorbar (0-1 scale)

---

## Recommended usage patterns

### A) Basic simulation (deterministic contributions)

```python
A = np.full(24, 100_000.0)                      # (24,)
R = returns.generate(T=24, n_sims=500, seed=42) # (500, 24, 2)
X = np.tile([0.5, 0.5], (24, 1))                # equal split

result = portfolio.simulate(A, R, X)
W = result["wealth"]  # (500, 25, 2)
```

### B) Stochastic contributions + returns

```python
# Both A and R are stochastic
A = income.contributions(24, start=date(2025,1,1), n_sims=500)  # (500, 24)
R = returns.generate(T=24, n_sims=500, seed=42)                  # (500, 24, 2)
X = np.tile([0.7, 0.3], (24, 1))

result = portfolio.simulate(A, R, X)
```

### C) Time-varying allocation policy

```python
# Glide path: decrease equity exposure over time
T = 60
equity_fractions = np.linspace(0.8, 0.4, T)
X = np.column_stack([equity_fractions, 1 - equity_fractions])  # (60, 2)

result = portfolio.simulate(A, R, X)
```

### D) Optimization-ready gradient computation

```python
# Compute accumulation factors once
F = portfolio.compute_accumulation_factors(R)  # (n_sims, T+1, T+1, M)

# Gradient of E[W_24^0] w.r.t. X[10, 0]
t_goal, s_contrib, m_account = 24, 10, 0
A_val = A[10] if A.ndim == 1 else A[:, 10].mean()
grad = A_val * F[:, s_contrib, t_goal, m_account].mean()
```

### E) Method selection heuristic

**Use `method="recursive"` when:**
- Pure simulation (no optimization)
- Large horizons ($T > 100$)
- Memory-constrained environments
- No need for gradients

**Use `method="affine"` when:**
- Integrating with optimizer
- Need analytical gradients
- Moderate horizons ($T \leq 100$)
- Sufficient RAM for $O(n_{\text{sims}} \cdot T^2 \cdot M)$ storage

---

## Key design decisions

### 1. Annual parameters as primary API

**Rationale:** Users think in annualized terms (e.g., "4% per year"), not monthly rates.

**Implementation:**
- `Account.from_annual()` is the recommended constructor
- Internal storage uses monthly (canonical for numerical computation)
- Properties provide dual views without runtime overhead

### 2. No embedded return generation

**Rationale:** Separation of concerns enables:
- Loose coupling (Portfolio never imports ReturnModel)
- Flexibility (users can swap return models)
- Testability (deterministic testing with controlled `R`)

**Pattern:**
```python
# ❌ Old pattern (tight coupling)
portfolio.simulate(A, X, return_params={...})

# ✅ Current pattern (loose coupling)
R = returns.generate(...)
portfolio.simulate(A, R, X)
```

### 3. Vectorized batch processing via `n_sims`

**Rationale:** Matches `income.py` API pattern for consistency.

**Performance benefit:**
```python
# Old: Python-level loop (slow)
W_batch = [portfolio.simulate(A[i], R[i], X) for i in range(500)]

# New: Single vectorized call (100x faster)
W_batch = portfolio.simulate(A, R, X)  # A: (500, T), R: (500, T, M)
```

### 4. Affine method as default

**Rationale:**
- Exposes gradients for optimization
- Same asymptotic complexity as recursive for typical use cases
- Memory overhead acceptable for $T \leq 100$

**Tradeoff:** Memory vs. gradient access

### 5. Accumulation factors as explicit artifact

**Rationale:**
- Optimization frameworks need $F_{s,t}^m$ for constraint reformulation
- Precomputation amortizes cost across multiple objective evaluations
- Users can inspect/debug via `compute_accumulation_factors()`

**Complexity:** Documented explicitly with memory estimates to guide method selection.

---

## Implementation notes

### Rate conversion consistency

All rate conversions use `utils.py` helpers:
```python
mu_monthly = annual_to_monthly(r_annual)  # geometric: (1+r)^(1/12)-1
sigma_monthly = sigma_annual / np.sqrt(12)  # time scaling
```

**Verification:**
```python
assert np.isclose(
    monthly_to_annual(annual_to_monthly(0.08)),
    0.08
)  # round-trip identity
```

### Wealth initialization

All simulations start from $W_0^m$ defined in `Account.initial_wealth`:
```python
W[:, 0, :] = self.initial_wealth_vector  # broadcast to (n_sims, M)
```

**Behavior:** If $W_0^m = 0$ for all accounts, wealth accumulates purely from contributions.

### Numerical stability

**Accumulation factors:**
- Use product of gross returns: $\prod (1 + R_r)$ (numerically stable)
- Avoid exp-sum-log for small returns (unnecessary complexity)
- For extreme cases ($T > 240$), consider log-space accumulation

**Gradient computation:**
- Gradients scale linearly with $A_s$ and $F_{s,t}^m$
- No risk of underflow for typical financial returns
- Division-free (multiplication only)

### Visualization edge cases

- **Single simulation ($n_{\text{sims}} = 1$):** histogram omitted (no distribution)
- **Many accounts ($M > 5$):** allocation heatmap text labels suppressed
- **Long horizons ($T > 24$):** heatmap remains readable via automatic aspect ratio

---

## Mathematical results

**Proposition 1 (Affine Wealth):**  
For any allocation policy $X \in \mathcal{X}_T$ and return realization $\{R_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
$$
is affine in $X$.

**Proof:** Direct substitution from recursive formula. ∎

**Corollary 1 (Linear Constraints):**  
If goals are specified as $W_t^m(X) \geq b_t^m$ (deterministic), the feasible allocation set is a convex polytope.

**Proposition 2 (Gradient):**  
The sensitivity of wealth to allocation at month $s$ is:
$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m, \quad s < t
$$

**Proof:** Differentiate affine formula w.r.t. $x_s^m$. ∎

**Corollary 2 (Monotonicity):**  
If $F_{s,t}^m > 0$ (positive returns), then $W_t^m(X)$ is strictly increasing in $x_s^m$.

**Proposition 3 (Stochastic Gradient):**  
For stochastic returns, the expected gradient is:
$$
\mathbb{E}\left[\frac{\partial W_t^m}{\partial x_s^m}\right] = \mathbb{E}[A_s] \cdot \mathbb{E}[F_{s,t}^m]
$$
assuming independence of $A_s$ and $F_{s,t}^m$.

**Remark:** For dependent $A$ and $R$, use sample average: $\frac{1}{N}\sum_{i=1}^N A_s^{(i)} F_{s,t}^{m,(i)}$.

---

## Extensions

**Multi-period rebalancing:** Allow $x_t^m$ to vary by month (already supported).

**Transaction costs:** Add friction terms $\kappa \|\Delta x_t\|_1$ to objective.

**Tax-aware dynamics:** Incorporate capital gains, withdrawal timing:
$$
W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m) - \tau \cdot \max(0, W_t^m R_t^m)
$$

**Robust optimization:** Replace $\mathbb{E}[W_T]$ with worst-case $\inf_{\mathbb{P} \in \mathcal{P}} \mathbb{E}_{\mathbb{P}}[W_T]$.

**Dynamic programming:** For path-dependent constraints, use Bellman recursion:
$$
V_t(W_t) = \max_{x_t} \mathbb{E}\big[V_{t+1}(W_{t+1}) \,|\, W_t, x_t\big]
$$

---
