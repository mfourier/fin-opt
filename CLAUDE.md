# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FinOpt is a goal-based portfolio optimization framework that solves the bilevel problem of finding the minimum investment horizon to achieve multiple financial goals with probabilistic guarantees. It combines Monte Carlo simulation with CVaR (Conditional Value-at-Risk) reformulation for globally optimal allocation policies.

**Core concept**: Instead of asking "what wealth can I achieve with horizon T?", FinOpt inverts the question to "what is the minimum horizon T* to achieve my goals?"

## Environment Setup

```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate finance

# Install CVXPY for convex optimization (required for CVaROptimizer)
pip install cvxpy

# Launch Jupyter for interactive development
jupyter lab
```

**Note**: There is no `requirements.txt` - dependencies are managed via `environment.yml`.

## Architecture: Six Independent Modules

The codebase follows strict separation of concerns with a unidirectional dependency flow:

```
income.py → portfolio.py → model.py → optimization.py
    ↓           ↓              ↓
returns.py ─────┘              │
                               │
goals.py ──────────────────────┘
```

### 1. `income.py` - Cash Flow Modeling

Generates contribution scenarios `A_t` (monthly contributions available for investment).

- **`FixedIncome`**: Deterministic salary with annual growth and scheduled raises
- **`VariableIncome`**: Stochastic income with seasonality (12-month factors), Gaussian noise, floor/cap constraints
- **`IncomeModel`**: Facade combining fixed + variable streams with configurable contribution rates

**Key pattern**: Calendar-aware outputs (indexed by first-of-month dates), reproducible randomness via explicit seeds.

### 2. `returns.py` - Stochastic Shocks

Generates correlated lognormal return scenarios `R_t^m` across multiple accounts.

- **`ReturnModel`**: Correlated lognormal generator with dual temporal representation (monthly/annual parameters)
- Covariance structure: Σ = D @ ρ @ D (diagonal volatilities × correlation matrix)
- Guarantees realistic returns: R_t > -1 (from lognormal distribution)

**Key insight**: No Portfolio dependency - loose coupling via Account interface.

### 3. `portfolio.py` - Wealth Dynamics Executor

Executes wealth evolution given contributions `A`, returns `R`, and allocation policy `X`.

- **`Account`**: Metadata container (name, initial wealth, return parameters). Always use `Account.from_annual()` for user-friendly API
- **`Portfolio`**: Vectorized wealth dynamics executor with two computation modes:
  - **Recursive**: W_{t+1} = (W_t + A_t·x_t)(1 + R_t)
  - **Affine** (closed-form): W_t = W_0·F_{0,t} + Σ A_s·x_s·F_{s,t}

**Affine representation is critical**: W_t is linear in allocation policy X, enabling convex optimization with analytical gradients.

### 4. `goals.py` - Objective Specification

Domain-level abstractions for financial goals as chance constraints.

- **`IntermediateGoal`**: Fixed calendar checkpoint (e.g., emergency fund by month 6)
  - ℙ(W_{t_fixed}^m ≥ b) ≥ 1-ε
- **`TerminalGoal`**: End-of-horizon target (e.g., retirement wealth at optimized horizon T*)
  - ℙ(W_T^m ≥ b) ≥ 1-ε
- **`GoalSet`**: Validator and metadata provider (T_min, account index resolution)

**Design principle**: Immutable frozen dataclasses with calendar-aware resolution.

### 5. `optimization.py` - Convex Solvers

Implements bilevel optimization: outer problem (minimize horizon T) + inner problem (convex allocation).

- **`CVaROptimizer`**: Convex reformulation via CVaR epigraphic form (uses CVXPY)
  - Transforms non-convex chance constraints into tractable LP/QP
  - Objectives: "risky", "balanced" (default), "conservative", "risky_turnover", or custom callable
- **`SAAOptimizer`**: Sample Average Approximation with gradient-based solvers (scipy)
- **`GoalSeeker`**: Binary/linear search over horizon T with warm-start

**CVaR Reformulation** (Rockafellar & Uryasev 2000):
```
Original (non-convex):  ℙ(W_t ≥ b) ≥ 1-ε
CVaR form (convex):     CVaR_ε(b - W_t) ≤ 0
Epigraph (LP):          γ + (1/εN)Σ z_i ≤ 0,  z_i ≥ b - W_t^i - γ,  z_i ≥ 0
```

This provides global optimality guarantees (vs local minima from gradient-based methods).

### 6. `model.py` - Orchestration Facade

**`FinancialModel`**: Unified entry point that coordinates income → returns → portfolio → optimization.

Key features:
- **Intelligent caching**: Simulation results cached by SHA256 hash of parameters (RAM-efficient)
- **Auto-simulation**: `plot()` method simulates internally when needed
- **Unified plotting**: 8 modes with single interface (wealth, allocation, income, etc.)

**`SimulationResult`**: Immutable dataclass containing wealth trajectories, contributions, returns, income breakdown, and metadata for reproducibility.

## Common Workflows

### Running the Main Notebook

The canonical example demonstrating the full pipeline:

```bash
jupyter lab notebooks/FinOpt-Workflow.ipynb
```

This notebook showcases:
- Multi-account setup with correlated returns
- Stochastic income with seasonality
- Multiple goals (intermediate + terminal)
- CVaR optimization with binary search
- Full visualization suite

### Typical Usage Pattern

```python
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.goals import TerminalGoal
from finopt.optimization import CVaROptimizer

# 1. Define income
income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))

# 2. Configure accounts (ALWAYS use .from_annual())
accounts = [
    Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09),
    Account.from_annual("Aggressive", annual_return=0.14, annual_volatility=0.15)
]

# 3. Create model
model = FinancialModel(income, accounts)

# 4. Define goals
goals = [TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.80)]

# 5. Optimize (finds minimum T* and optimal allocation policy X*)
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
result = model.optimize(goals=goals, optimizer=optimizer, T_max=120, n_sims=500, seed=42)

# 6. Visualize
model.plot("wealth", result=result, show_trajectories=True)
```

## Critical Implementation Details

### Allocation Policy Format

Allocation policy `X` must be:
- Shape: `(T, M)` where T=horizon, M=number of accounts
- Constraints: `x_t^m ≥ 0` and `Σ_m x_t^m = 1` (simplex constraint per time step)
- Interpretation: `X[t, m]` = fraction of month-t contribution to account m

### Seed Management

For reproducibility, seeds propagate through:
1. `VariableIncome(seed=...)` for stochastic income
2. `ReturnModel.generate(..., seed=...)` for return scenarios
3. `model.simulate(..., seed=...)` sets both income and returns

### Initial Wealth Override

Portfolio supports `W0_override` for optimization scenarios:
```python
# Accounts have initial_wealth=0, but optimization needs non-zero W0
W0_scenario = np.array([5_000_000, 2_000_000])
result = portfolio.simulate(A=A_sims, R=R_sims, X=X, W0_override=W0_scenario)
```

### Vectorization Requirements

All components process full Monte Carlo batches:
- Income: `(n_sims, T)` contribution arrays
- Returns: `(n_sims, T, M)` return arrays
- Wealth: `(n_sims, T+1, M)` trajectory arrays (includes t=0)

### Goal Resolution

- `IntermediateGoal.month`: integer offset from `start_date` OR `datetime.date` (auto-resolved)
- `account` parameter: integer index OR string name (resolved via `GoalSet`)
- `confidence`: 1-ε (e.g., 0.80 means 80% probability of success)

## Mathematical Foundation

### Affine Wealth Representation

The core optimization insight is that wealth is **affine in allocation policy**:

```
W_t^m(X) = W_0^m · F_{0,t}^m + Σ_{s=0}^{t-1} A_s · x_s^m · F_{s,t}^m
```

where `F_{s,t}^m = Π_{τ=s+1}^{t} (1 + R_τ^m)` is the accumulation factor.

**Consequences**:
- Convex constraints remain convex (enables CVaR reformulation)
- Analytical gradients: ∂W_t^m/∂x_s^m = A_s · F_{s,t}^m
- O(1) wealth evaluation (no recursion needed in optimization inner loop)

### CVaR Objectives

All built-in objectives maintain convexity:
- **"risky"**: Linear program, maximizes E[Σ_m W_T^m]
- **"balanced"**: Quadratic program, E[W_T] - λ·Σ|Δx| (turnover penalty)
- **"conservative"**: Quadratic program, E[W_T] - λ·Std(W_T) (risk-adjusted)

Custom objectives must be convex in X to maintain global optimality.

## Code Style and Patterns

### Docstring Format

All modules use comprehensive NumPy-style docstrings with:
- Purpose section explaining role in larger system
- Mathematical framework with LaTeX notation
- Design principles highlighting architectural decisions
- Example usage with imports and concrete values

### Type Hints

Extensive use of type hints with `TYPE_CHECKING` guards to avoid circular imports:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .optimization import AllocationOptimizer
```

### Frozen Dataclasses

Goals and results are immutable:
```python
@dataclass(frozen=True)
class TerminalGoal:
    ...
```

SimulationResult is mutable (`frozen=False`) to support field updates during optimization.

### Annual Parameters by Default

Always use `.from_annual()` constructors for user-facing APIs:
```python
# Good
Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09)

# Avoid (requires manual monthly conversion)
Account("Conservative", monthly_return=0.006434, monthly_volatility=0.02598)
```

## Project-Specific Conventions

### Currency Formatting

Use utility functions from `utils.py`:
- `millions_formatter()`: Matplotlib formatter for M CLP axis labels
- `format_currency(value)`: String formatting (e.g., "$1.50M CLP")

### Plotting Integration

`FinancialModel.plot()` supports 8 modes:
- `"wealth"`: Total wealth over time with goal achievement
- `"allocation"`: Heatmap of allocation policy X
- `"income"`: Fixed + variable income projections
- `"contributions"`: Capital deployed over time
- `"returns"`: Return distributions and correlations
- `"combined"`: Multi-panel dashboard
- Plus account-specific variants

All plots auto-simulate if `result` not provided, with caching enabled.

## Troubleshooting

### CVXPY Not Found

If `CVaROptimizer` raises ImportError:
```bash
pip install cvxpy
```

This is a lazy import - only required for convex optimization.

### Infeasible Optimization

If `GoalSeeker` reports all horizons infeasible:
1. Check goal thresholds aren't too high for available contributions
2. Verify confidence levels (lower ε = easier to satisfy)
3. Inspect intermediate goals: may force early wealth accumulation incompatible with terminal goals
4. Reduce number of scenarios (n_sims) for debugging - try n_sims=100

### Numerical Issues

For large wealth values (>100M), use relative thresholds in goals or scale all monetary values by 1e6 internally.

## References

The mathematical framework is based on:
- **CVaR reformulation**: Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
- **Affine wealth dynamics**: Exploits linearity for convex programming (standard MPC technique)
- **Bilevel optimization**: Outer binary search + inner convex program

For implementation details, see comprehensive docstrings in each module and the detailed markdown files in `docs/`.
