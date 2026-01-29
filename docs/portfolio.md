# `portfolio` — Philosophy & Role in FinOpt

> **Purpose:** Execute **wealth dynamics** under allocation policies, providing the mathematical foundation for optimization-driven portfolio construction.
> `portfolio.py` is the **executor layer**: it receives pre-generated contributions `A` (from `income.py`), returns `R` (from `returns.py`), and withdrawals `D`, applies allocation policy `X`, and computes wealth trajectories `W` via recursive or closed-form methods.

---

## Why a dedicated portfolio module?

Financial optimization requires **separating concerns**:
- `income.py` → cash availability (contributions `A`)
- `returns.py` → stochastic return generation (`R`)
- `portfolio.py` → **wealth evolution given allocations** (`W`)
- `goals.py` → goal specifications
- `optimization.py` → policy search

This separation enables:
- **Loose coupling:** Portfolio never generates returns (delegated to `ReturnModel`)
- **Optimization-ready:** Affine wealth representation exposes analytical gradients
- **Batch processing:** Vectorized Monte Carlo execution (no Python loops)
- **Dual temporal API:** Seamless monthly ↔ annual parameter conversion
- **Withdrawal support:** Handles scheduled or stochastic withdrawals `D`

---

## Design principles

1. **Separation of concerns**
   - Portfolio executes dynamics, does NOT generate stochastic processes
   - Returns/contributions/withdrawals are **external inputs** (not embedded models)

2. **Vectorized computation**
   - Full batch processing: `(n_sims, T, M)` arrays without Python-level loops
   - Matches `income.py` pattern: `n_sims` parameter for Monte Carlo generation

3. **Optimization-first design**
   - Affine wealth formula enables **analytical gradients**: $\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m$
   - Direct integration with convex solvers (CVXPY)

4. **Annual parameters by default**
   - User-facing API uses intuitive annual returns/volatility
   - Internal storage in monthly space (canonical form)
   - Properties provide dual temporal views without conversions

5. **Flexible naming**
   - Short `name` for goal specification (e.g., "RN", "CC")
   - Optional `display_name` for plots (e.g., "Risky Norris (Fintual)")
   - `label` property returns display_name if set, otherwise name

---

## The two core surfaces

### 1) `Account`

Frozen dataclass metadata container for investment account with **dual temporal parameter access** and **flexible naming**.

**Parameters:**
- `name`: Short account identifier for goal references (e.g., "RN", "CC", "SLV")
- `initial_wealth`: Starting balance $W_0^m$ (non-negative)
- `return_strategy`: dict with **monthly** parameters `{"mu": float, "sigma": float}`
- `display_name`: Optional long descriptive name for plots (e.g., "Risky Norris (Fintual)")

**Constructor methods (recommended):**
```python
# With display_name (recommended for clarity)
Account.from_annual(
    name="RN",
    annual_return=0.12,
    annual_volatility=0.15,
    initial_wealth=0.0,
    display_name="Risky Norris (Fintual)"
)

# Without display_name (backward compatible)
Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05)

# From monthly parameters (advanced)
Account.from_monthly("TAC", monthly_mu=0.0058, monthly_sigma=0.0347)
```

**Properties:**
- `label`: Display name for plots (returns `display_name` if set, otherwise `name`)
- `monthly_params`: Canonical storage format `{"mu": float, "sigma": float}`
- `annual_params`: User-friendly view `{"return": float, "volatility": float}`

**Parameter conversion:**
$$
\begin{aligned}
\mu_{\text{monthly}} &= (1 + r_{\text{annual}})^{1/12} - 1 \quad \text{[geometric compounding]} \\
\sigma_{\text{monthly}} &= \frac{\sigma_{\text{annual}}}{\sqrt{12}} \quad \text{[time scaling]}
\end{aligned}
$$

**Example with display_name:**
```python
# Short name for goal specification
risky = Account.from_annual("RN", annual_return=0.12, annual_volatility=0.15,
                            display_name="Risky Norris (Fintual)")

print(risky.name)   # 'RN' - used in goals
print(risky.label)  # 'Risky Norris (Fintual)' - shown in plots

# Goal uses short name
goal = TerminalGoal(account="RN", threshold=5_000_000, confidence=0.8)
```

---

### 2) `Portfolio`

Multi-account wealth dynamics executor with allocation policy and withdrawal support.

**Parameters:**
- `accounts`: list of `Account` objects (metadata only, no return models)

**Properties:**
- `M`: Number of accounts
- `account_names`: List of account names
- `initial_wealth_vector`: Initial wealth array `W0` from accounts

**Method signature:**
```python
def simulate(
    self,
    A: np.ndarray,      # Contributions: (T,) or (n_sims, T)
    R: np.ndarray,      # Returns: (n_sims, T, M)
    X: np.ndarray,      # Allocations: (T, M)
    D: np.ndarray = None,  # Withdrawals: (T, M) or (n_sims, T, M)
    method: Literal["recursive", "affine"] = "recursive",
    initial_wealth: np.ndarray = None  # Override W0
) -> dict
```

**Core wealth dynamics (with withdrawals):**
$$
W_{t+1}^m = \big(W_t^m + A_t^m - D_t^m\big)(1 + R_t^m)
$$
where:
- $A_t^m = x_t^m \cdot A_t$ (allocated contribution)
- $D_t^m$ = withdrawal from account $m$ at time $t$
- $\sum_{m=1}^M x_t^m = 1$, $x_t^m \geq 0$ (budget constraint)

**Returns:**
```python
{
    "wealth": np.ndarray,        # (n_sims, T+1, M)
    "total_wealth": np.ndarray   # (n_sims, T+1)
}
```

**Computation methods:**

| Method | Default | Time | Memory | Use Case |
|--------|---------|------|--------|----------|
| `"recursive"` | ✓ | $O(n \cdot T \cdot M)$ | $O(n \cdot T \cdot M)$ | Simulation, large T |
| `"affine"` | | $O(T^2 \cdot M \cdot n)$ | $O(n \cdot T^2 \cdot M)$ | Optimization, gradients |

---

## Withdrawal Support

Withdrawals `D` can be provided as:
- **Deterministic:** Shape `(T, M)` — same withdrawal schedule across all scenarios
- **Stochastic:** Shape `(n_sims, T, M)` — per-scenario withdrawals
- **None:** No withdrawals (default)

**Example:**
```python
# Deterministic: $500K from Housing account at month 12
D = np.zeros((T, M))
D[12, 1] = 500_000  # Account 1 = Housing

result = portfolio.simulate(A=A, R=R, X=X, D=D)
```

**In optimization:**
Withdrawals are treated as **parameters** (not decision variables), preserving convexity:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} \big(A_s x_s^m - D_s^m\big) F_{s,t}^m
$$

---

## Affine wealth representation

### Closed-form formula

The recursive dynamics admit a **closed-form solution**:

$$
\boxed{
W_t^m(X) = W_0^m \cdot F_{0,t}^m + \sum_{s=0}^{t-1} \big(A_s \, x_s^m - D_s^m\big) \, F_{s,t}^m
}
$$

where the **accumulation factor** from month $s$ to $t$ is:

$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m)
$$

**Convention:** $F_{s,s}^m = 1$ (no accumulation over empty interval).

### Mathematical properties

1. **Affine in $X$:**
   Since $D$ is a parameter (not decision variable), wealth remains affine in $X$:
   $$
   W_t^m(\alpha X + \beta Y) = \alpha W_t^m(X) + \beta W_t^m(Y) + \text{const}
   $$

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

**Memory estimates:**
- `n_sims=500, T=24, M=2`: ~115 MB
- `n_sims=500, T=120, M=5`: ~14 GB
- `n_sims=1000, T=240, M=10`: ~221 GB (infeasible)

**For $T > 100$, consider:**
- Using `method="recursive"` (no $F$ precomputation)
- Batching simulations (chunking `n_sims`)
- On-the-fly gradient computation

---

## Initial Wealth Override

The `initial_wealth` parameter allows overriding the initial wealth vector without creating temporary Account objects:

```python
# Accounts have initial_wealth=0, but optimization needs non-zero W0
accounts = [
    Account.from_annual("Emergency", 0.04, 0.05, initial_wealth=0),
    Account.from_annual("Housing", 0.07, 0.12, initial_wealth=0)
]
portfolio = Portfolio(accounts)

# Override W0 for optimization scenario
W0_scenario = np.array([5_000_000, 2_000_000])
result = portfolio.simulate(A=A, R=R, X=X, initial_wealth=W0_scenario)

# Verify override
assert np.allclose(result["wealth"][:, 0, :], W0_scenario)
```

---

## Integration with FinOpt pipeline

### Workflow

```python
from datetime import date
import numpy as np
from finopt.src.portfolio import Account, Portfolio
from finopt.src.returns import ReturnModel
from finopt.src.income import IncomeModel, FixedIncome

# 1. Define accounts (annual parameters + display names)
accounts = [
    Account.from_annual("EM", annual_return=0.04, annual_volatility=0.05,
                        display_name="Emergency Fund"),
    Account.from_annual("HS", annual_return=0.07, annual_volatility=0.12,
                        display_name="Housing Savings")
]

# 2. Create portfolio executor
portfolio = Portfolio(accounts)

# 3. Generate stochastic inputs externally
income = IncomeModel(fixed=FixedIncome(base=1_400_000, annual_growth=0.03))
returns = ReturnModel(accounts, default_correlation=np.eye(2))

T, n_sims = 24, 500
A = income.contributions(T, start=date(2025, 1, 1), n_sims=n_sims)  # (500, 24)
R = returns.generate(T, n_sims=n_sims, seed=42)                      # (500, 24, 2)

# 4. Define allocation policy
X = np.tile([0.6, 0.4], (T, 1))  # 60-40 split

# 5. Execute wealth dynamics (with optional withdrawals)
D = np.zeros((T, 2))
D[12, 0] = 500_000  # $500K withdrawal from Emergency at month 12

result = portfolio.simulate(A=A, R=R, X=X, D=D)
W = result["wealth"]              # (500, 25, 2)
W_total = result["total_wealth"]  # (500, 25)

# 6. Visualize with goals
from finopt.src.goals import TerminalGoal, IntermediateGoal

goals = [
    IntermediateGoal(date=date(2026, 1, 1), account="EM",
                     threshold=5_000_000, confidence=0.95),
    TerminalGoal(account="HS", threshold=20_000_000, confidence=0.90)
]

portfolio.plot(result=result, X=X, start=date(2025, 1, 1), goals=goals)
```

### Data flow

```
income.py          returns.py
    ↓                  ↓
    A                  R          (external generation)
    ↓                  ↓
    └──► portfolio.simulate(A, R, X, D) ──► W
                       ↑        ↑
                       X        D    (from user or optimizer)
```

---

## Visualization

**Method signature:**
```python
def plot(
    self,
    result: dict,           # from simulate()
    X: np.ndarray,          # allocation policy
    start: date = None,     # calendar start date
    goals: list = None,     # Goal objects to visualize
    figsize: tuple = (16, 10),
    title: str = None,
    save_path: str = None,
    return_fig_ax: bool = False,
    show_trajectories: bool = True,
    trajectory_alpha: float = 0.05,
    colors: dict = None,
    hist_bins: int = 30,
    hist_color: str = 'mediumseagreen'
)
```

**Panel layout:**
1. **Top-left:** Wealth per account (time series with Monte Carlo trajectories + goal markers)
2. **Top-right:** Total portfolio wealth + lateral histogram of final wealth distribution
3. **Bottom-left:** Portfolio composition (stacked area chart)
4. **Bottom-right:** Allocation policy (stacked bar chart)

**Goal visualization:**
- **TerminalGoal:** Horizontal dashed line at threshold across entire horizon
- **IntermediateGoal:** Dotted line up to goal month with diamond marker

**Calendar-aware x-axis:**
When `start` is provided, the x-axis shows calendar dates instead of month indices.

---

## Usage patterns

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
A = income.contributions(24, start=date(2025, 1, 1), n_sims=500)  # (500, 24)
R = returns.generate(T=24, n_sims=500, seed=42)                    # (500, 24, 2)
X = np.tile([0.7, 0.3], (24, 1))

result = portfolio.simulate(A, R, X)
```

### C) With withdrawals

```python
# $1M withdrawal from Housing at month 18
D = np.zeros((T, 2))
D[18, 1] = 1_000_000

result = portfolio.simulate(A, R, X, D=D)
```

### D) Time-varying allocation policy (glide path)

```python
T = 60
equity_fractions = np.linspace(0.8, 0.4, T)
X = np.column_stack([equity_fractions, 1 - equity_fractions])  # (60, 2)

result = portfolio.simulate(A, R, X)
```

### E) Optimization-ready gradient computation

```python
# Compute accumulation factors once
F = portfolio.compute_accumulation_factors(R)  # (n_sims, T+1, T+1, M)

# Gradient of E[W_24^0] w.r.t. X[10, 0]
t_goal, s_contrib, m_account = 24, 10, 0
A_val = A[:, 10].mean() if A.ndim == 2 else A[10]
grad = A_val * F[:, s_contrib, t_goal, m_account].mean()
```

### F) Method selection heuristic

**Use `method="recursive"` (default) when:**
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

## Exceptions

The module raises `AllocationConstraintError` when allocation policy violates constraints:

```python
from finopt.src.exceptions import AllocationConstraintError

try:
    result = portfolio.simulate(A, R, X)
except AllocationConstraintError as e:
    print(f"Invalid allocation: {e}")
    # Fix: ensure X[t, :].sum() == 1 and X >= 0
```

---

## Mathematical results

**Proposition 1 (Affine Wealth):**
For any allocation policy $X \in \mathcal{X}_T$, return realization $\{R_t^m\}$, and withdrawal schedule $\{D_t^m\}$:
$$
W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} (A_s x_s^m - D_s^m) F_{s,t}^m
$$
is affine in $X$ (since $D$ is a parameter).

**Corollary 1 (Linear Constraints):**
If goals are specified as $W_t^m(X) \geq b_t^m$ (deterministic), the feasible allocation set is a convex polytope.

**Proposition 2 (Gradient):**
The sensitivity of wealth to allocation at month $s$ is:
$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s F_{s,t}^m, \quad s < t
$$

**Corollary 2 (Monotonicity):**
If $F_{s,t}^m > 0$ (positive returns), then $W_t^m(X)$ is strictly increasing in $x_s^m$.

**Proposition 3 (Withdrawal Independence):**
Withdrawals $D$ do not affect gradients $\frac{\partial W_t^m}{\partial x_s^m}$ since they are parameters, not decision variables. This preserves convexity for CVaR optimization.
