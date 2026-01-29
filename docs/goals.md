# `goals` — Probabilistic Goal Specification for FinOpt

> **Core idea:** translate **financial goals** into **chance constraints** that can be validated against Monte Carlo simulations and reformulated for convex optimization.
> `goals.py` connects what the user **wants to achieve** (e.g., "$20M in account X with 90% confidence") with the CVaR-based optimization framework that finds the **minimum horizon** and **optimal allocation policy** to satisfy these goals.

---

## Why a dedicated goals module?

- **Probabilistic semantics**: Goals are chance constraints ℙ(W_t^m ≥ b) ≥ 1-ε, not deterministic targets
- **Two temporal flavors**: Intermediate goals (fixed calendar date) vs. terminal goals (variable horizon T)
- **Optimization-ready**: Provides `GoalSet` abstraction consumed by `CVaROptimizer` and `GoalSeeker`
- **Validation utilities**: Post-simulation functions to check goal satisfaction with detailed metrics

---

## Design philosophy

1. **Immutable specifications**
   - Goals are `frozen` dataclasses — safe to hash, cache, and use as dict keys
   - No mutation after construction

2. **Dual temporal semantics**
   - `IntermediateGoal`: Fixed calendar checkpoint (independent of optimization horizon)
   - `TerminalGoal`: Evaluated at variable T (the horizon being optimized)

3. **Account resolution**
   - Goals reference accounts by name (`str`) or index (`int`)
   - `GoalSet` validates and resolves references at construction time

4. **Calendar-aware month resolution**
   - Dates converted to month offsets via `resolve_month(start_date)`
   - Results cached for O(1) lookup during optimization

---

## Key concepts

### Chance constraints

Goals are formulated as probabilistic requirements:

**IntermediateGoal** (fixed time t_fixed):
```
ℙ(W_{t_fixed}^m ≥ threshold) ≥ confidence
```

**TerminalGoal** (variable horizon T):
```
ℙ(W_T^m ≥ threshold) ≥ confidence
```

Where:
- `W_t^m` = wealth in account m at time t
- `threshold` = minimum required wealth (e.g., $5,500,000 CLP)
- `confidence` = 1 - ε (e.g., 0.90 means 90% chance of success)
- `ε` = violation tolerance (e.g., 0.10 allows 10% failure rate)

### CVaR reformulation

The optimization module transforms these non-convex chance constraints into tractable convex form:

```
Original (non-convex):  ℙ(W_t ≥ b) ≥ 1-ε
CVaR form (convex):     CVaR_ε(b - W_t) ≤ 0
```

This enables globally optimal allocation policies via convex programming.

---

## Main API

### 1) `IntermediateGoal` (dataclass)

Fixed-time financial checkpoint. Used for liquidity requirements, planned expenses, or milestone tracking.

```python
from datetime import date
from finopt.src.goals import IntermediateGoal

goal = IntermediateGoal(
    date=date(2025, 7, 1),       # Target date (required)
    account="Emergency",          # Account name or index
    threshold=5_500_000,          # Minimum wealth required
    confidence=0.90               # 90% chance of success
)

# Month resolution (relative to simulation start)
month = goal.resolve_month(date(2025, 1, 1))  # → 6

# Violation tolerance
epsilon = goal.epsilon  # → 0.10
```

**Parameters:**
- `date`: Target date for evaluation (converted to month offset)
- `account`: Account identifier (int index or str name)
- `threshold`: Minimum required wealth (must be > 0)
- `confidence`: Required probability of success ∈ (0, 1)

**Month resolution semantics:**
- `date=July 1, 2025` with `start_date=January 1, 2025` → month 6
- Checks wealth W_6 (wealth at start of period 6, i.e., July 1)

---

### 2) `TerminalGoal` (dataclass)

End-of-horizon target evaluated at variable T. Used for retirement targets, long-term savings, or final portfolio value.

```python
from finopt.src.goals import TerminalGoal

goal = TerminalGoal(
    account="Retirement",         # Account name or index
    threshold=20_000_000,         # Terminal wealth target
    confidence=0.90               # 90% chance of success
)
```

**Parameters:**
- `account`: Account identifier (int index or str name)
- `threshold`: Minimum required terminal wealth (must be > 0)
- `confidence`: Required probability of success ∈ (0, 1)

**Key difference from IntermediateGoal:**
- No fixed `date` — evaluated at horizon T (the optimization variable)
- Used by `GoalSeeker` to find minimum T* such that goal is feasible

---

### 3) `GoalSet` (class)

Validated collection of goals with account resolution and utilities for optimization.

```python
from datetime import date
from finopt.src.portfolio import Account
from finopt.src.goals import IntermediateGoal, TerminalGoal, GoalSet

# Define accounts
accounts = [
    Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
    Account.from_annual("Housing", annual_return=0.07, annual_volatility=0.12)
]

# Define goals
goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Emergency", threshold=20_000_000, confidence=0.90),
    TerminalGoal(account="Housing", threshold=7_000_000, confidence=0.90)
]

# Create validated collection
goal_set = GoalSet(goals, accounts, start_date=date(2025, 1, 1))

# Access properties
goal_set.T_min                    # → 6 (from intermediate goal)
goal_set.M                        # → 2 (number of accounts)
goal_set.intermediate_goals       # → [IntermediateGoal(...)]
goal_set.terminal_goals           # → [TerminalGoal(...), TerminalGoal(...)]

# Resolve account index for a goal
idx = goal_set.get_account_index(goals[0])  # → 0

# Get cached resolved month (O(1))
month = goal_set.get_resolved_month(goals[0])  # → 6
```

**Validation rules:**
- Goals list cannot be empty
- All account references must resolve to valid indices
- No duplicate IntermediateGoal for same (month, account) pair
- No duplicate TerminalGoal for same account

**Key methods:**
- `get_account_index(goal)`: Returns resolved 0-based account index
- `get_resolved_month(goal)`: Returns cached month offset for IntermediateGoal
- `estimate_minimum_horizon(...)`: Heuristic T estimate for terminal goals

---

### 4) Goal validation functions

#### `check_goals()` — Validate goal satisfaction

```python
from finopt.src.goals import check_goals

status = check_goals(
    result=simulation_result,
    goals=goals,
    accounts=accounts,
    start_date=date(2025, 1, 1)
)

for goal, metrics in status.items():
    print(f"{goal}: {'✓' if metrics['satisfied'] else '✗'}")
    print(f"  Violation rate: {metrics['violation_rate']:.1%}")
    print(f"  Required rate:  {metrics['required_rate']:.1%}")
    print(f"  Margin:         {metrics['margin']:+.1%}")
```

**Returns dict with metrics for each goal:**
- `satisfied`: bool — True if empirical violation rate ≤ ε
- `violation_rate`: float — Empirical ℙ(W_t^m < threshold)
- `required_rate`: float — Goal's ε = 1 - confidence
- `margin`: float — required_rate - violation_rate (positive = satisfied)
- `median_shortfall`: float — Median shortfall over violated scenarios
- `n_violations`: int — Count of scenarios violating threshold

---

#### `goal_progress()` — Track progress toward goals

```python
from finopt.src.goals import goal_progress

progress = goal_progress(
    result=simulation_result,
    goals=goals,
    accounts=accounts,
    start_date=date(2025, 1, 1)
)

for goal, pct in progress.items():
    account = goal.account
    print(f"{account}: {pct:.1%} progress")
```

**Progress metric:** `min(1, VaR_{1-ε}(W_t^m) / threshold)`

- 0.0: VaR is zero (far from goal)
- 0.5: VaR is 50% of threshold (halfway)
- 1.0: VaR ≥ threshold (goal achieved at confidence level)

---

#### `print_goal_status()` — Pretty-print goal satisfaction

```python
from finopt.src.goals import print_goal_status

print_goal_status(
    result=simulation_result,
    goals=goals,
    accounts=accounts,
    start_date=date(2025, 1, 1)
)
```

**Example output:**
```
=== Goal Status ===

[✓] IntermediateGoal: Emergency Fund @ month 6
    Target: $5,500,000 | Confidence: 90.0%
    Status: SATISFIED (margin: +2.3%)
    Violation rate: 7.7% (38 scenarios)

[✗] TerminalGoal: Emergency Fund @ T=24
    Target: $20,000,000 | Confidence: 90.0%
    Status: VIOLATED (margin: -3.1%)
    Violation rate: 13.1% (66 scenarios)
    Median shortfall: $1,234,567
```

---

## Month resolution semantics

Goals and the wealth array use **start-of-period** indexing:

| Date | `resolve_month()` | Wealth Index | Interpretation |
|------|-------------------|--------------|----------------|
| January 1, 2025 (start) | — | W_0 | Initial wealth |
| February 1, 2025 | 1 | W_1 | Wealth at start of month 1 |
| July 1, 2025 | 6 | W_6 | Wealth at start of month 6 |
| January 1, 2026 (T=12) | 12 | W_12 | Terminal wealth |

**Example with start_date = January 1, 2025:**
```python
goal = IntermediateGoal(date=date(2025, 7, 1), ...)
month = goal.resolve_month(date(2025, 1, 1))  # → 6
# Checks wealth[i, 6, m] which is W_6 (wealth at July 1)
```

---

## Integration with optimization

### Optimization workflow

```python
from finopt import FinancialModel
from finopt.src.goals import IntermediateGoal, TerminalGoal
from finopt.optimization import CVaROptimizer

# 1. Define goals
goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=5_500_000, confidence=0.90),
    TerminalGoal(account="Retirement", threshold=20_000_000, confidence=0.90)
]

# 2. Create optimizer
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

# 3. Optimize (finds minimum T* and optimal X*)
result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,
    n_sims=500,
    seed=42
)

# 4. Validate goals
result.validate_goals()  # Uses stored goal_set
```

### How goals flow through the system

1. **User defines goals** → `IntermediateGoal` / `TerminalGoal` objects
2. **`GoalSet` validates** → Resolves accounts, caches months, computes T_min
3. **`CVaROptimizer` reformulates** → Chance constraints → CVaR ≤ 0
4. **`GoalSeeker` searches** → Binary search over T with warm-start
5. **`check_goals` validates** → Post-simulation verification

---

## Complete example

```python
from datetime import date
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.src.goals import (
    IntermediateGoal, TerminalGoal, GoalSet,
    check_goals, goal_progress, print_goal_status
)
from finopt.optimization import CVaROptimizer

# Setup
income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
accounts = [
    Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
    Account.from_annual("Growth", annual_return=0.12, annual_volatility=0.15)
]
model = FinancialModel(income, accounts)

# Define goals
start_date = date(2025, 1, 1)
goals = [
    IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
                     threshold=5_000_000, confidence=0.95),
    TerminalGoal(account="Growth", threshold=50_000_000, confidence=0.80)
]

# Validate goal structure
goal_set = GoalSet(goals, accounts, start_date)
print(f"Minimum horizon from intermediate goals: T_min = {goal_set.T_min}")

# Optimize
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
result = model.optimize(goals=goals, optimizer=optimizer, T_max=120, n_sims=500)

# Check results
print_goal_status(result, goals, accounts, start_date)

# Get detailed metrics
status = check_goals(result, goals, accounts, start_date)
progress = goal_progress(result, goals, accounts, start_date)
```

---

## Comparison with legacy API

| Legacy (removed) | Current |
|------------------|---------|
| `Goal` dataclass | `IntermediateGoal`, `TerminalGoal` |
| `target_amount` | `threshold` |
| `target_date` / `target_month_index` | `date` (IntermediateGoal only) |
| `evaluate_goal()` | `check_goals()` |
| `evaluate_goals()` → DataFrame | `check_goals()` → Dict |
| `GoalEvaluation` | Dict with metrics |
| `allocate_contributions_proportional()` | Removed (use optimization) |
| `required_constant_contribution()` | `GoalSet.estimate_minimum_horizon()` |
| Deterministic success/failure | Probabilistic chance constraints |
