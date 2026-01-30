# `exceptions` — Error Handling for FinOpt

> **Core idea:** Provide a unified exception hierarchy for consistent error handling across all FinOpt modules.
> All exceptions inherit from `FinOptError`, enabling catch-all handling when needed while maintaining specific error types for precise debugging.

---

## Why a dedicated exceptions module?

- **Consistency:** All FinOpt errors share a common base class
- **Precision:** Specific exception types for different failure modes
- **Debugging:** Descriptive error messages with actionable suggestions
- **Integration:** Clean separation between FinOpt errors and Python built-ins

---

## Exception Hierarchy

```
FinOptError (base)
├── ConfigurationError      # Invalid configuration or parameters
├── ValidationError         # Data validation failures
│   ├── TimeIndexError          # Month/date indexing errors
│   └── AllocationConstraintError   # Allocation policy violations
├── OptimizationError       # Solver failures
│   └── InfeasibleError         # No feasible solution exists
└── MemoryLimitError        # Memory limit exceeded
```

---

## Exception Classes

### `FinOptError`

Base exception for all FinOpt errors. Use this to catch any FinOpt-specific error.

```python
from finopt.exceptions import FinOptError

try:
    result = model.optimize(goals)
except FinOptError as e:
    logger.error(f"FinOpt error: {e}")
```

---

### `ConfigurationError`

Invalid configuration or parameters.

**Raised when:**
- Invalid `annual_growth` values (must be > -1)
- Incompatible parameter combinations
- Missing required configuration fields

```python
from finopt.exceptions import ConfigurationError

# Example: Invalid annual growth
raise ConfigurationError(
    "annual_growth must be > -1, got -2.0. "
    "Value <= -1 would cause income to become zero or negative."
)
```

**Common causes:**
- `FixedIncome(annual_growth=-1.5)` — growth rate too negative
- `Account.from_annual(annual_return=-1.1, ...)` — return below -100%

---

### `ValidationError`

Data validation failures.

**Raised when:**
- Invalid array shapes
- Out-of-bounds values
- Constraint violations

```python
from finopt.exceptions import ValidationError

# Example: Invalid horizon
raise ValidationError(
    f"T must be positive, got {T}. "
    f"Use T >= 1 for valid return simulations."
)
```

**Common causes:**
- `model.simulate(T=-5)` — negative horizon
- `X.shape != (T, M)` — allocation policy shape mismatch
- `confidence > 1.0` — probability out of range

---

### `TimeIndexError`

Month/date indexing errors. Subclass of `ValidationError`.

**Raised when:**
- Month out of valid range
- Date before simulation start
- Inconsistent date/month specifications

```python
from finopt.exceptions import TimeIndexError

# Example: Withdrawal before simulation
raise TimeIndexError(
    f"Withdrawal month {month} is before simulation start (month 1). "
    f"Months are 1-indexed: month 1 = end of first simulation month."
)
```

**Common causes:**
- `IntermediateGoal(date=date(2024, 1, 1))` with `start=date(2025, 1, 1)` — goal before start
- `WithdrawalEvent(month=0)` — month must be ≥ 1
- Goal month exceeds horizon T

---

### `AllocationConstraintError`

Allocation policy constraint violations. Subclass of `ValidationError`.

**Raised when:**
- Negative allocations ($x_t^m < 0$)
- Simplex violations ($\sum_m x_t^m \neq 1$)
- Invalid allocation array shape

```python
from finopt.exceptions import AllocationConstraintError

# Example: Negative allocation
raise AllocationConstraintError(
    f"Allocation X has negative values at positions: {negative_locs}. "
    f"Min value: {X.min():.6f}. Allocations must be non-negative."
)
```

**Common causes:**
- Manual allocation with `X[t, m] = -0.1`
- Rounding errors after optimization (use tolerance)
- Row sums not equal to 1.0

---

### `OptimizationError`

Optimization solver failures.

**Raised when:**
- Solver convergence failure
- Numerical instability
- Unexpected solver status

```python
from finopt.exceptions import OptimizationError

# Example: Solver failure
raise OptimizationError(
    f"Solver returned unexpected status: {prob.status}. "
    f"Try adjusting solver settings or using a different solver."
)
```

**Common causes:**
- Ill-conditioned problem (extreme parameter values)
- Solver timeout
- Numerical precision issues with very large wealth values

---

### `InfeasibleError`

No feasible solution exists. Subclass of `OptimizationError`.

**Raised when:**
- Goals cannot be achieved within the horizon
- Withdrawal constraints cannot be satisfied
- Conflicting constraint requirements

```python
from finopt.exceptions import InfeasibleError

# Example: No feasible solution
raise InfeasibleError(
    f"No feasible solution found in T ∈ [{T_min}, {T_max}]. "
    f"Consider: (1) increasing T_max, (2) relaxing goal thresholds, "
    f"(3) reducing withdrawal amounts, (4) increasing epsilon tolerances."
)
```

**Common causes:**
- Goal threshold too high for available contributions
- Withdrawal exceeds projected wealth
- Intermediate goal conflicts with terminal goal
- `confidence` too high (e.g., 0.99 may be infeasible)

**Recovery strategies:**
1. Increase `T_max` to allow more time for wealth accumulation
2. Lower goal `threshold` values
3. Reduce `confidence` levels (e.g., 0.90 → 0.80)
4. Reduce or reschedule withdrawal amounts
5. Increase contribution rates

---

### `MemoryLimitError`

Memory limit exceeded.

**Raised when:**
- Large accumulation factor arrays
- Too many Monte Carlo scenarios
- Excessive horizon length

```python
from finopt.exceptions import MemoryLimitError

# Example: Memory exceeded
raise MemoryLimitError(
    f"Accumulation factors would require {memory_gb:.1f} GB. "
    f"Use LazyAccumulationFactors for large problems, or reduce: "
    f"n_sims={n_sims}, T={T}, M={M}"
)
```

**Common causes:**
- `n_sims=10000` with `T=240` and `M=5` — very large scenario matrix
- Pre-computing all accumulation factors for long horizons

**Recovery strategies:**
1. Reduce `n_sims` (e.g., 500 is often sufficient)
2. Use lazy evaluation for accumulation factors
3. Reduce horizon `T` if possible
4. Process in batches

---

## Usage Patterns

### A) Catch all FinOpt errors

```python
from finopt.exceptions import FinOptError

try:
    result = model.optimize(goals=goals, T_max=120)
except FinOptError as e:
    print(f"FinOpt encountered an error: {e}")
    # Log, notify user, or attempt recovery
```

---

### B) Handle specific error types

```python
from finopt.exceptions import (
    InfeasibleError,
    ValidationError,
    OptimizationError
)

try:
    result = model.optimize(goals=goals, T_max=60)
except InfeasibleError as e:
    print(f"Goals are not achievable: {e}")
    # Suggest relaxing constraints
except ValidationError as e:
    print(f"Invalid input: {e}")
    # Fix input data
except OptimizationError as e:
    print(f"Solver failed: {e}")
    # Try different solver
```

---

### C) Re-raise with context

```python
from finopt.exceptions import ValidationError

def validate_goals(goals, accounts):
    for goal in goals:
        if goal.account not in [a.name for a in accounts]:
            raise ValidationError(
                f"Goal references unknown account '{goal.account}'. "
                f"Available accounts: {[a.name for a in accounts]}"
            )
```

---

### D) Conditional recovery for infeasibility

```python
from finopt.exceptions import InfeasibleError

def optimize_with_fallback(model, goals, T_max=60):
    try:
        return model.optimize(goals=goals, T_max=T_max)
    except InfeasibleError:
        # Try with extended horizon
        print(f"Infeasible at T_max={T_max}, trying T_max={T_max*2}")
        return model.optimize(goals=goals, T_max=T_max*2)
```

---

## Module Usage in FinOpt

| Module | Exceptions Used |
|--------|-----------------|
| `income.py` | `ValidationError`, `ConfigurationError` |
| `portfolio.py` | `ValidationError`, `AllocationConstraintError` |
| `goals.py` | `ValidationError`, `TimeIndexError` |
| `withdrawal.py` | `ValidationError`, `TimeIndexError` |
| `optimization.py` | `OptimizationError`, `InfeasibleError`, `ValidationError` |
| `model.py` | `ValidationError`, `MemoryLimitError` |
| `returns.py` | `ValidationError` |

---

## API Summary

| Exception | Parent | Purpose |
|-----------|--------|---------|
| `FinOptError` | `Exception` | Base class for all FinOpt errors |
| `ConfigurationError` | `FinOptError` | Invalid configuration or parameters |
| `ValidationError` | `FinOptError` | Data validation failures |
| `TimeIndexError` | `ValidationError` | Month/date indexing errors |
| `AllocationConstraintError` | `ValidationError` | Allocation policy violations |
| `OptimizationError` | `FinOptError` | Solver failures |
| `InfeasibleError` | `OptimizationError` | No feasible solution exists |
| `MemoryLimitError` | `FinOptError` | Memory limit exceeded |
