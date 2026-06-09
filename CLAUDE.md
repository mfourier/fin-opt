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

## Architecture: Nine Modules

The codebase follows strict separation of concerns with a unidirectional dependency flow:

```
Core Pipeline:
income.py → portfolio.py → model.py → optimization.py
    ↓           ↓              ↓
returns.py ─────┘              │
                               │
goals.py ──────────────────────┘

Supporting Modules:
config.py ─── serialization.py ─── cli.py
```

### 1. `income.py` - Cash Flow Modeling

Generates contribution scenarios `A_t` (monthly contributions available for investment).

- **`FixedIncome`**: Deterministic salary with annual growth and scheduled raises
- **`VariableIncome`**: Stochastic income with seasonality (12-month factors), Gaussian noise, floor/cap constraints
- **`IncomeModel`**: Facade combining fixed + variable streams with configurable contribution rates
  - `monthly_contribution`: dict with 12-element arrays per income type
  - **Defaults**: `{"fixed": [0.3]*12, "variable": [1.0]*12}` (30% of salary, 100% of bonuses)

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

- **`IntermediateGoal`**: Fixed calendar checkpoint (e.g., emergency fund by July 2025)
  - ℙ(W_{t_fixed}^m ≥ b) ≥ 1-ε
  - Params: `account`, `threshold`, `confidence`, `date`
- **`TerminalGoal`**: End-of-horizon target (e.g., retirement wealth at optimized horizon T*)
  - ℙ(W_T^m ≥ b) ≥ 1-ε
  - Params: `account`, `threshold`, `confidence`
- **`GoalSet`**: Validator and metadata provider (T_min, account index resolution)
  - `estimate_minimum_horizon(monthly_contribution, accounts)`: Deterministic T estimate for terminal goals

**Utility functions**:
- **`check_goals(result, goals, accounts, start_date)`**: Validates goal satisfaction, returns metrics dict with dual reporting:
  - **Legacy metrics**: satisfied, violation_rate, required_rate, margin, median_shortfall, n_violations
  - **CVaR transparency metrics** (new): empirical_probability, confidence_gap, note
  - Reports both specified confidence (CVaR guarantee) and empirical probability (observed success rate)
- **`goal_progress(result, goals, accounts, start_date)`**: Computes VaR-based progress toward each goal
- **`print_goal_status(result, goals, accounts, start_date)`**: Pretty-prints goal satisfaction status with dual metrics display

**Design principle**: Immutable frozen dataclasses with calendar-aware resolution.

**CVaR Transparency**: Due to CVaR conservatism (CVaR_ε ≤ 0 ⟹ ℙ(W≥b) ≥ 1-ε, one-way implication), empirical probability typically exceeds specified confidence. Both metrics are reported for intellectual honesty.

### 5. `optimization.py` - Convex Solvers

Implements bilevel optimization: outer problem (minimize horizon T) + inner problem (convex allocation).

- **`CVaROptimizer`**: Convex reformulation via CVaR epigraphic form (uses CVXPY)
  - Transforms non-convex chance constraints into tractable LP/QP
  - Objectives: "risky", "balanced" (default), "conservative", "risky_turnover", or custom callable
  - Solver selection via `solver` kwarg: "ECOS" (default), "SCS", "CLARABEL", "MOSEK"
- **`GoalSeeker`**: Binary/linear search over horizon T with warm-start
- **`OptimizationResult`**: Container for X*, T*, objective value, goal_set, and diagnostics
  - `validate_goals(result)`: Validates goal satisfaction using stored goal_set
  - `is_valid_allocation()`: Checks simplex constraints on X
  - `goal_metrics`: Optional field storing dual metrics from `check_goals()` (empirical probability + confidence gap)

**CVaR Reformulation** (Rockafellar & Uryasev 2000):
```
Original (non-convex):  ℙ(W_t ≥ b) ≥ 1-ε
CVaR form (convex):     CVaR_ε(b - W_t) ≤ 0
Epigraph (LP):          γ + (1/εN)Σ z_i ≤ 0,  z_i ≥ b - W_t^i - γ,  z_i ≥ 0
```

This provides global optimality guarantees (vs local minima from gradient-based methods).

**CVaR Conservatism**: The reformulation is conservative - satisfying CVaR constraint **guarantees** but does not **require** the probabilistic constraint. In practice:
- CVaR_ε(b - W) ≤ 0  ⟹  ℙ(W ≥ b) ≥ 1-ε  (one-way implication)
- Empirical probability typically exceeds specified confidence by 1-5%
- Both metrics reported via `check_goals()` for transparency

**Optimization Objectives**:

| Objective | Formula | Type | Use Case |
|-----------|---------|------|----------|
| `"risky"` | `E[Σ_m W_T^m]` | LP | Maximum wealth accumulation |
| `"balanced"` | `-Σ_{t,m}(Δx_{t,m})²` | QP | Stable allocations (turnover penalty only) |
| `"conservative"` | `E[W_T] - λ·Std(W_T)` | QP | Risk-averse mean-variance |
| `"risky_turnover"` | `E[W_T] - λ·Σ(Δx)²` | QP | Wealth + stability tradeoff |
| Custom callable | `f(W, X, T, M) → float` | - | User-defined (must be convex) |

### 6. `model.py` - Orchestration Facade

**`FinancialModel`**: Unified entry point that coordinates income → returns → portfolio → optimization.

Key features:
- **Intelligent caching**: Simulation results cached by SHA256 hash of parameters (RAM-efficient)
- **Auto-simulation**: `plot()` method simulates internally when needed
- **Unified plotting**: 8 modes with single interface (wealth, allocation, income, etc.)
- **`simulate_from_optimization()`**: Re-simulates using OptimizationResult's X* and goal_set

**`SimulationResult`**: Immutable dataclass containing wealth trajectories, contributions, returns, income breakdown, and metadata for reproducibility.

### 7. `config.py` - Configuration Management

Pydantic-based type-safe configuration for all model parameters.

- **`SimulationConfig`**: n_sims, seed, cache_enabled, verbose
- **`OptimizationConfig`**: T_max, T_min, solver, objective, search_strategy, tolerance
- **`IncomeConfig`**: Combines FixedIncomeConfig + VariableIncomeConfig with contribution rates
- **`AccountConfig`**: name, annual_return, annual_volatility, initial_wealth
- **`AppSettings`**: Environment variables (FINOPT_* prefix), .env file support

```python
from finopt.config import SimulationConfig, OptimizationConfig

sim_config = SimulationConfig(n_sims=1000, seed=42, cache_enabled=True)
opt_config = OptimizationConfig(T_max=120, solver="ECOS", objective="balanced")

# Serialize to JSON
config_dict = sim_config.model_dump()
```

### 8. `serialization.py` - Model Persistence

JSON serialization for model configurations and optimization results.

- **`save_model(model, path)`**: Saves FinancialModel to JSON (income, accounts, correlation)
- **`load_model(path)`**: Reconstructs FinancialModel from JSON
- **`save_optimization_result(result, path)`**: Saves X*, T, goals, diagnostics
- **`load_optimization_result(path)`**: Returns dict (requires context for full reconstruction)
- Schema versioning: Currently "0.2.0"

```python
from finopt.serialization import save_model, load_model
from pathlib import Path

save_model(model, Path("config.json"))
loaded_model = load_model(Path("config.json"))
```

### 9. `cli.py` - Command-Line Interface

Click-based CLI for running simulations without Python code.

```bash
# Run simulation from config file
finopt simulate --config config.json --output results/ --horizon 36

# Optimize allocation policy
finopt optimize --config config.json --goals goals.json --max-horizon 120

# Validate configuration
finopt config validate config.json

# Show version
finopt --version
```

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

### Wealth Array Indexing (Convention B: Start of Period)

The wealth array has shape `(n_sims, T+1, M)` using **start-of-period** semantics:
- `wealth[i, 0, m]` = W_0^m (initial wealth at **start** of period 0)
- `wealth[i, t, m]` = W_t^m (wealth at **start** of period t, after t periods of dynamics)
- `wealth[i, T, m]` = terminal wealth at start of period T (used for TerminalGoal evaluation)

**Wealth dynamics equation**:
```
W_{t+1} = (W_t + A_t·x_t - D_t)(1 + R_t)
```

Where during period t:
- W_t = wealth available at START of period t
- A_t = contribution received at start of period t (invested immediately)
- D_t = withdrawal during period t (taken from W_t, before contribution)
- R_t = return earned during period t
- W_{t+1} = resulting wealth at START of period t+1

**Timeline example** (start=January 2025, T=12):
```
Period:  0 (Jan)    1 (Feb)    ...    5 (Jun)    6 (Jul)    ...
         ↓          ↓                 ↓          ↓
W_t:     W_0 ────→ W_1 ────→ ... ──→ W_5 ────→ W_6 ────→ ... ──→ W_12
         Jan 1     Feb 1             Jun 1      Jul 1            Jan 1, 2026
```

### Probabilistic Constraints

**Goal constraint** (IntermediateGoal at API month m):
```
ℙ(W_m^k ≥ b^k) ≥ 1 - ε^k
```
Meaning: "Start period m with at least $b in account k with confidence 1-ε"

**Withdrawal feasibility constraint** (Conservative - Option 1):
```
ℙ(W_t^m ≥ D_t^m) ≥ 1 - ε
```
Meaning: "Have enough wealth at START of period t to cover withdrawal D_t"

Note: The withdrawal is taken from existing wealth W_t, NOT from W_t + A_t·x_t. This is conservative and ensures the withdrawal is feasible regardless of whether the contribution arrives on time.

### CVaR Reformulation

Transforms non-convex chance constraints into tractable convex form:
```
Original:   ℙ(W_t ≥ b) ≥ 1-ε        (non-convex)
CVaR form:  CVaR_ε(b - W_t) ≤ 0     (convex)
```

Based on Rockafellar & Uryasev (2000).

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

### Goal and Withdrawal Month Resolution

Both goals and withdrawals use **1-indexed API months** that map to 0-indexed array indices:

| Component | Date Example | `resolve_month()` | Array Index | Wealth/Variable |
|-----------|--------------|-------------------|-------------|-----------------|
| IntermediateGoal | July 1, 2025 | 6 | - | Checks W_6 |
| WithdrawalEvent | June 1, 2025 | 6 | 5 | Uses D_5, constraint on W_5 |

**Key insight**: Different dates can resolve to the same API month with different semantics:
- Goal `date=July 1` → "check wealth at start of July" → W_6
- Withdrawal `date=June 1` → "withdrawal during June" → D_5, affects W_5 → W_6

This ensures a withdrawal "on June 1" is reflected when checking a goal "by July 1".

**Parameter resolution**:
- `date`: datetime.date (auto-resolved to month offset via `resolve_month()`)
- `account`: integer index OR string name (resolved via `GoalSet`)
- `confidence`: 1-ε (e.g., 0.80 means 80% probability of success)

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

## API Layer (`api/`)

FastAPI service that exposes the `finopt` core library over HTTP, persists jobs and results in Supabase (JSONB), and runs optimization/simulation asynchronously.

### Directory structure

```
api/
├── main.py              # FastAPI app, endpoints, request/response models
├── config.py            # Settings (env vars via FINOPT_* prefix + .env)
├── supabase_client.py   # Supabase helpers: fetch, update_job, save_*_result
└── services/
    ├── _goal_metrics.py # Shared CVaR dual-metric helper
    ├── simulation.py    # Simulation service + goal status formatting
    ├── optimization.py  # Optimization service + goal status formatting
    └── reconstruction.py# Reconstructs FinancialModel from Supabase profile/scenario rows
```

### Endpoints (`api/main.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/simulate` | Enqueue simulation job, run async via `run_simulation()` |
| `POST` | `/optimize` | Enqueue optimization job, run async via `run_optimization()` |

Both `/simulate` and `/optimize` accept a `JobRequest` body (`scenario_id`, `job_id`) and return a `JobResponse`. The heavy work runs in a background task — callers poll Supabase for job status.

### Services

**`services/simulation.py`**
- `compute_goal_status(result, goals, accounts, start_date)` → `list[dict]` — evaluates each goal against the simulation wealth array; includes dual metric fields.
- `compute_summary_stats(sim_result, accounts, start_date)` → `SummaryStats` dict.
- `run_simulation(scenario_id, job_id)` — async entry point; fetches scenario, runs `FinancialModel.simulate()`, saves result to Supabase.

**`services/optimization.py`**
- `compute_goal_status_from_result(opt_result, model, sim_result, start_date)` → `list[dict]` — like the simulation variant but operates on `OptimizationResult`; handles `actual_prob = None` (goal month out of range) by setting all dual metric fields to `None`.
- `compute_cash_flow_stats(sim_result, accounts)` → cash flow dict.
- `compute_wealth_percentiles(sim_result, accounts)` → percentile dict.
- `run_optimization(scenario_id, job_id)` — async entry point; fetches scenario, runs `FinancialModel.optimize()` then `simulate_from_optimization()`, saves result to Supabase.

**`services/_goal_metrics.py`** — shared helper used by both services:
```python
compute_dual_metrics(empirical_probability: float, specified_confidence: float) -> dict
# Returns: {"empirical_probability": float, "confidence_gap": float, "note": str}
# note categories:
#   gap > 0.01  → "CVaR optimization yields conservative estimates..." (significant conservatism)
#   0 ≤ gap ≤ 0.01 → "CVaR constraint satisfied..." (mild conservatism)
#   gap < 0     → "Warning: Empirical probability ... below specified confidence..." (violation)
```

### `GoalStatus` output schema

Every `GoalStatus` dict in the API response contains:

```python
{
    "goal": str,                    # human-readable description
    "type": "terminal" | "intermediate",
    "account": str,
    "threshold": float,
    "required_confidence": float,   # specified confidence (CVaR guarantee)
    "actual_probability": float,    # empirical success rate (observed)
    "satisfied": bool,
    # CVaR transparency fields (v0.2+):
    "empirical_probability": float | None,   # same as actual_probability, explicit
    "confidence_gap": float | None,          # empirical - required (positive = conservative)
    "note": str | None,                      # human-readable explanation
}
```

These fields are mirrored in the TypeScript interface `GoalStatus` in `web/src/types/database.ts` as optional fields (`?:`).

### Supabase persistence

Results are stored as JSONB in the `results` table — no schema migration is needed when adding new optional fields to `goal_status` entries. The `web/` frontend reads `result.goal_status` and renders it via `GoalProgressCard`.

## References

The mathematical framework is based on:
- **CVaR reformulation**: Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
- **Affine wealth dynamics**: Exploits linearity for convex programming (standard MPC technique)
- **Bilevel optimization**: Outer binary search + inner convex program

For implementation details, see comprehensive docstrings in each module and the detailed markdown files in `docs/`.
