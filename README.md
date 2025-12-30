# FinOpt

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Goal-based portfolio optimization under uncertainty using convex programming and stochastic simulation.**

FinOpt determines the minimum investment horizon to achieve multiple financial goals with probabilistic guarantees, combining Monte Carlo simulation with CVaR (Conditional Value-at-Risk) reformulation for globally optimal allocation policies.

---

## 🎯 Problem Statement

Traditional financial planning tools use either:
- **Heuristic rules** (e.g., 60/40 portfolio, constant rebalancing) → suboptimal
- **Gradient-based optimization** (local minima, no guarantees) → unreliable
- **Monte Carlo simulation** without optimization → what-if analysis only

**FinOpt solves the bilevel problem:**

```
min T                           [Find shortest horizon]
s.t. ∃ X ∈ F_T                  [Where policy X satisfies all goals]

where F_T = {X : P(W_t^m(X) ≥ b_t^m) ≥ 1-ε, ∀ goals}
```

Using **CVaR reformulation** (Rockafellar & Uryasev, 2000), we convert non-convex chance constraints into a **convex optimization problem** with global optimality guarantees.

---

## ✨ Key Features

### Mathematical Rigor
- **Affine wealth representation**: Closed-form dynamics expose linear structure for convex solvers
- **CVaR reformulation**: Transforms probabilistic constraints into tractable epigraphic form
- **Global optimality**: DCP-compliant objectives (CVXPY) eliminate local minima

### Practical Capabilities
- **Multiple goals**: Intermediate (fixed time) + terminal (optimized horizon)
- **Stochastic income**: Seasonality, noise, growth, floor/cap constraints
- **Correlated returns**: Multi-asset portfolios with lognormal shocks
- **Flexible objectives**: Terminal wealth, low turnover, risk-adjusted, balanced
- **Unified visualization**: 8 plotting modes with auto-simulation

---

## 📊 Visual Showcase

### Income Projection with Seasonality
![Income Projection](docs/images/income_projection.png)
*Stochastic variable income (gray trajectories) + deterministic fixed income with scheduled raises*

### Wealth Dynamics Under Optimal Policy
![Wealth Dynamics](docs/images/wealth_dynamics.png)
*Monte Carlo trajectories showing goal achievement: UF @ month 6, Conservative @ month 18, Terminal goals @ T=23*

### Allocation Analysis with Investment Gains
![Allocation Analysis](docs/images/allocation_analysis.png)
*Capital invested vs wealth decomposition: $23.3M contributions → $27.9M final wealth (+19.7%)*

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/finopt.git
cd finopt

# Create virtual environment (Python 3.11+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy`, `pandas`: Numerical computing
- `cvxpy`: Convex optimization (ECOS/SCS/CLARABEL solvers)
- `matplotlib`: Visualization
- `scipy`: Gradient-based optimization (SAA)

### Minimal Example

```python
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.goals import TerminalGoal
from finopt.optimization import CVaROptimizer
from datetime import date

# 1. Define income stream
income = IncomeModel(
    fixed=FixedIncome(
        base=1_500_000,  # CLP/month
        annual_growth=0.03,
        salary_raises={date(2026, 6, 1): 500_000}
    )
)

# 2. Configure accounts
accounts = [
    Account.from_annual(
        name="Conservative Fund",
        annual_return=0.08,
        annual_volatility=0.09,
        initial_wealth=1_000_000
    ),
    Account.from_annual(
        name="Aggressive Fund",
        annual_return=0.14,
        annual_volatility=0.15,
        initial_wealth=500_000
    )
]

# 3. Create model
model = FinancialModel(income, accounts)

# 4. Define goal
goals = [
    TerminalGoal(
        account="Aggressive Fund",
        threshold=5_000_000,
        confidence=0.80  # 80% success probability
    )
]

# 5. Optimize
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42
)

print(f"Minimum horizon: T*={result.T} months")
print(f"Objective value: {result.objective_value:.4f}")

# 6. Visualize
model.plot("wealth", result=result, show_trajectories=True)
```

**Output:**
```
=== GoalSeeker: BINARY search T ∈ [1, 120] ===
[Iter 1] Testing T=60... ✓ Feasible
[Iter 2] Testing T=30... ✓ Feasible  
[Iter 3] Testing T=15... ✗ Infeasible
[Iter 4] Testing T=23... ✓ Feasible
=== Optimal: T*=23 (converged in 7 iterations) ===
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           FinancialModel (Facade)               │
│  - Orchestration layer                          │
│  - Unified plotting interface                   │
│  - Simulation caching                           │
└───────┬─────────────────────────────────────────┘
        │
        ├──► IncomeModel ──┬──► FixedIncome
        │                  └──► VariableIncome
        │
        ├──► ReturnModel (Lognormal correlations)
        │
        ├──► Portfolio (Wealth dynamics executor)
        │       W_t = W_0·F_{0,t} + Σ A_s·x_s·F_{s,t}
        │
        └──► GoalSeeker ───┬──► AllocationOptimizer
                           │       ├─► SAAOptimizer (scipy)
                           │       └─► CVaROptimizer (CVXPY)
                           └──► GoalSet (validation)
```

**Module responsibilities:**

| Module | Purpose | Key Abstractions |
|--------|---------|------------------|
| `income.py` | Cash flow modeling | `FixedIncome`, `VariableIncome`, `IncomeModel` |
| `returns.py` | Stochastic shocks | `ReturnModel` (correlated lognormal) |
| `portfolio.py` | Wealth dynamics | `Account`, `Portfolio` (affine executor) |
| `goals.py` | Objective specification | `IntermediateGoal`, `TerminalGoal`, `GoalSet` |
| `optimization.py` | Convex solvers | `SAAOptimizer`, `CVaROptimizer`, `GoalSeeker` |
| `model.py` | Orchestration | `FinancialModel` (facade), `SimulationResult` |

---

## 📐 Mathematical Foundation

### Wealth Dynamics (Affine Representation)

For account $m$ with allocation policy $X = \{x_t^m\}$:

$$
W_t^m(X) = W_0^m \cdot F_{0,t}^m + \sum_{s=0}^{t-1} A_s \cdot x_s^m \cdot F_{s,t}^m
$$

where $F_{s,t}^m = \prod_{\tau=s}^{t-1}(1 + R_\tau^m)$ are accumulation factors.

**Key property**: $W_t^m$ is **affine in $X$** → convex constraints remain convex.

### CVaR Reformulation

Original chance constraint (non-convex):
$$
\mathbb{P}(W_t^m \geq b) \geq 1 - \varepsilon
$$

CVaR reformulation (convex):
$$
\text{CVaR}_\varepsilon(b - W_t^m) \leq 0
$$

where:
$$
\text{CVaR}_\varepsilon(L) = \min_{\gamma} \left\{ \gamma + \frac{1}{\varepsilon N}\sum_{i=1}^N [L_i - \gamma]_+ \right\}
$$

**Epigraphic form** (LP-compatible):
$$
\begin{align}
\gamma + \frac{1}{\varepsilon N}\sum_{i=1}^N z_i &\leq 0 \\
z_i &\geq b - W_t^m(\omega^{(i)}) - \gamma, \quad \forall i \\
z_i &\geq 0, \quad \forall i
\end{align}
$$

**Theorem (Rockafellar & Uryasev, 2000)**: CVaR ≤ 0 **implies** P(W ≥ b) ≥ 1-ε (conservative approximation with global optimality).

### Bilevel Optimization

**Outer problem** (horizon search):
```
min T ∈ ℕ
s.t. inner problem is feasible
```

**Inner problem** (allocation):
```
max f(X)                    [Objective: terminal_wealth, balanced, etc.]
s.t. Σ_m x_t^m = 1, ∀t      [Simplex constraint]
     x_t^m ≥ 0, ∀t,m         [Non-negativity]
     CVaR_ε(b_g - W_{t_g}) ≤ 0, ∀g  [Goal constraints]
```

**Solution method**: Binary search with warm-start (O(log T) iterations vs O(T) linear).

---

## 💼 Usage Examples

### Example 1: Multiple Goals with Different Horizons

```python
from finopt.goals import IntermediateGoal, TerminalGoal

goals = [
    # Emergency fund by month 6
    IntermediateGoal(
        month=6,
        account="Savings Account",
        threshold=3_000_000,
        confidence=0.95  # High confidence for safety net
    ),
    # House down payment by month 18
    IntermediateGoal(
        month=18,
        account="Conservative Fund",
        threshold=10_000_000,
        confidence=0.80
    ),
    # Retirement goal (horizon to be optimized)
    TerminalGoal(
        account="Aggressive Fund",
        threshold=50_000_000,
        confidence=0.70
    )
]

result = model.optimize(goals, optimizer, T_max=240)

# Verify goal satisfaction
verification = model.verify_goals(result, goals)
for goal_id, status in verification.items():
    print(f"{goal_id}: {status['status']} (margin: {status['margin']:.1%})")
```

### Example 2: Stochastic Income with Seasonality

```python
from finopt import VariableIncome

# Freelance income with summer peak
variable = VariableIncome(
    base=800_000,
    seasonality=[0.5, 0.5, 0.8, 1.0, 1.2, 1.5,  # Jan-Jun (summer high)
                 1.5, 1.2, 1.0, 0.8, 0.5, 0.5], # Jul-Dec (winter low)
    sigma=0.15,  # 15% monthly noise
    floor=0,     # Can have zero income months
    cap=2_000_000,
    annual_growth=0.05
)

income = IncomeModel(
    fixed=FixedIncome(base=2_000_000, annual_growth=0.03),
    variable=variable,
    monthly_contribution={
        "fixed": [0.4] * 12,    # Save 40% of fixed
        "variable": [1.0] * 12  # Save 100% of variable
    }
)
```

### Example 3: Risk-Adjusted Objectives

```python
from finopt.optimization import CVaROptimizer

# Compare different risk preferences
objectives = ["risky", "balanced", "conservative"]
results = {}

for obj in objectives:
    optimizer = CVaROptimizer(n_accounts=3, objective=obj)
    results[obj] = model.optimize(goals, optimizer, T_max=120, seed=42)
```

---

## 🔬 Extensions & Future Work

### Planned Features
- [ ] **Transaction costs**: Convex penalties for rebalancing (L1/L2 norms)
- [ ] **Taxes**: Capital gains modeling for Chilean tax regime
- [ ] **Multi-period rebalancing**: Dynamic policies vs pre-commitment
- [ ] **Robust optimization**: Uncertainty sets for (μ, σ) parameters
- [ ] **Real data backtesting**: Historical IPSA, Fintual fund returns

---

## 📚 References

**Optimization Theory**:
- Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk". *Journal of Risk*, 2(3), 21-41.
- Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2014). *Lectures on Stochastic Programming*. SIAM.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Maximiliano Lioi**  
M.Sc. Applied Mathematics | Data Scientist  
[GitHub](https://github.com/mfourier) | [LinkedIn](https://linkedin.com/in/mlioi)

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Not financial advice. Consult certified professionals before making investment decisions.
