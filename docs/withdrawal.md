# `withdrawal` — Scheduled Cash Outflows for FinOpt

> **Core idea:** Model planned **withdrawals** (retiros) from investment accounts as parameters in the wealth dynamics equation.
> `withdrawal.py` captures cash outflows such as purchases, emergency expenses, or periodic distributions, producing arrays that downstream modules (`portfolio`, `optimization`) consume to adjust wealth trajectories.

---

## Why a dedicated withdrawal module?

- **Explicit cash outflows:** Withdrawals are first-class citizens, not hidden adjustments
- **Optimization-compatible:** Withdrawals are **parameters** (not decision variables), preserving convexity
- **Calendar-aware:** Dates are resolved to month offsets, matching `goals.py` and `income.py` patterns
- **Dual mode:** Supports both deterministic schedules and stochastic (uncertain) withdrawals

---

## Design philosophy

1. **Immutable specifications**
   - `WithdrawalEvent` and `StochasticWithdrawal` are frozen dataclasses
   - Safe to hash, cache, and use as dict keys

2. **Calendar-aware resolution**
   - Dates converted to month offsets via `resolve_month(start_date)`
   - 1-indexed months (matching `IntermediateGoal`)

3. **Pattern matching**
   - `WithdrawalEvent` → analogous to `IntermediateGoal` (fixed date, single amount)
   - `StochasticWithdrawal` → analogous to `VariableIncome` (base + sigma + floor/cap)
   - `WithdrawalModel` → analogous to `IncomeModel` (facade combining deterministic + stochastic)

4. **Backward compatible**
   - `D=None` in `portfolio.simulate()` preserves existing behavior (no withdrawals)

---

## Mathematical Framework

### Wealth dynamics with withdrawals

$$
W_{t+1}^m = \big(W_t^m + A_t \cdot x_t^m - D_t^m\big)(1 + R_t^m)
$$

where:
- $W_t^m$ = wealth in account $m$ at start of month $t$
- $A_t$ = total contribution at month $t$
- $x_t^m$ = allocation fraction to account $m$
- $D_t^m$ = withdrawal from account $m$ during month $t$
- $R_t^m$ = return of account $m$ during month $t$

**Timing convention:** Withdrawal occurs at **start of month** (before returns applied). The withdrawn amount does not earn returns that month — this is the conservative assumption.

### Affine representation (critical for optimization)

$$
\boxed{
W_t^m(X) = W_0^m \cdot F_{0,t}^m + \sum_{s=0}^{t-1} \big(A_s \cdot x_s^m - D_s^m\big) \cdot F_{s,t}^m
}
$$

**Key insight:** $D$ is a **parameter** (not a decision variable), so wealth remains affine in $X$, preserving convexity for CVaR optimization.

---

## Key components

### 1) `WithdrawalEvent` (frozen dataclass)

Single scheduled withdrawal from an investment account.

```python
from datetime import date
from finopt.src.withdrawal import WithdrawalEvent

event = WithdrawalEvent(
    account="Conservador",       # Account name or index
    amount=400_000,              # Withdrawal amount (must be positive)
    date=date(2025, 6, 1),       # Calendar date
    description="Compra bicicleta"  # Optional description
)

# Month resolution (1-indexed)
month = event.resolve_month(date(2025, 1, 1))  # → 6
```

**Parameters:**
- `account`: Target account identifier (int index or str name)
- `amount`: Withdrawal amount (must be positive)
- `date`: Calendar date of the withdrawal
- `description`: Optional human-readable description

**Month resolution:** Same as `IntermediateGoal.resolve_month()` — returns 1-indexed month offset.

---

### 2) `WithdrawalSchedule` (dataclass)

Collection of scheduled withdrawals for portfolio simulation.

```python
from finopt.src.withdrawal import WithdrawalSchedule

schedule = WithdrawalSchedule(events=[
    WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1), "Bicicleta"),
    WithdrawalEvent("Agresivo", 2_000_000, date(2026, 12, 1), "Vacaciones")
])

# Convert to array for simulation
D = schedule.to_array(
    T=36,
    start_date=date(2025, 1, 1),
    accounts=accounts
)
# D.shape → (36, 2)
# D[5, 0] → 400000.0 (June withdrawal from account 0)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_array(T, start_date, accounts)` | `(T, M)` array | Convert events to numpy array |
| `total_by_account(accounts)` | `Dict[str, float]` | Sum of withdrawals per account |
| `get_events_for_account(account)` | `List[WithdrawalEvent]` | Filter events by account |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(payload)` | `WithdrawalSchedule` | Deserialize from dictionary |

**Behavior:**
- Events outside simulation horizon are ignored with a warning
- Multiple events on the same month/account are summed
- Empty events list is valid (returns zeros)

---

### 3) `StochasticWithdrawal` (frozen dataclass)

Withdrawal with variability/uncertainty. Models withdrawals that have a base expected amount but may vary across scenarios (e.g., variable medical expenses, emergency costs).

```python
from finopt.src.withdrawal import StochasticWithdrawal

withdrawal = StochasticWithdrawal(
    account="Conservador",
    base_amount=300_000,      # Expected amount (mean)
    sigma=50_000,             # Standard deviation
    date=date(2025, 9, 1),    # Calendar date (or use month=9)
    floor=200_000,            # Minimum amount
    cap=500_000,              # Maximum amount
    seed=42                   # Random seed
)

# Generate samples
samples = withdrawal.sample(n_sims=1000, start_date=date(2025, 1, 1))
# samples.shape → (1000,)
# All samples in [200_000, 500_000]
```

**Parameters:**
- `account`: Target account identifier
- `base_amount`: Expected withdrawal amount (mean of distribution)
- `sigma`: Standard deviation
- `month` or `date`: Timing (mutually exclusive, one required)
- `floor`: Minimum withdrawal (default 0.0)
- `cap`: Maximum withdrawal (None = no cap)
- `seed`: Random seed for reproducibility

**Sampling:** Truncated Gaussian distribution $\mathcal{N}(\text{base}, \sigma^2)$ clamped to $[\text{floor}, \text{cap}]$.

---

### 4) `WithdrawalModel` (dataclass)

Unified facade combining scheduled and stochastic withdrawals.

```python
from finopt.src.withdrawal import WithdrawalModel, WithdrawalSchedule, StochasticWithdrawal

model = WithdrawalModel(
    scheduled=WithdrawalSchedule(events=[
        WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
    ]),
    stochastic=[
        StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            date=date(2025, 9, 1),
            seed=42
        )
    ]
)

# Generate combined withdrawal array
D = model.to_array(
    T=36,
    start_date=date(2025, 1, 1),
    accounts=accounts,
    n_sims=500,
    seed=42
)
# D.shape → (500, 36, 2)

# Check expected totals
model.total_expected(accounts)
# → {'Conservador': 700000.0, 'Agresivo': 0.0}
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_array(T, start_date, accounts, n_sims, seed)` | `(n_sims, T, M)` array | Combined withdrawal scenarios |
| `total_expected(accounts)` | `Dict[str, float]` | Expected total per account |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(payload)` | `WithdrawalModel` | Deserialize from dictionary |

**Behavior:**
- Scheduled withdrawals: broadcast to all scenarios (same values)
- Stochastic withdrawals: independent sampling per scenario
- Empty model returns zeros

---

## Month Resolution

Withdrawals use **1-indexed** months, matching `IntermediateGoal`:

| Date | `resolve_month()` | Array Index | Interpretation |
|------|-------------------|-------------|----------------|
| January 1, 2025 (start) | 1 | 0 | Withdrawal during month 0 |
| February 1, 2025 | 2 | 1 | Withdrawal during month 1 |
| June 1, 2025 | 6 | 5 | Withdrawal during month 5 |
| December 1, 2026 | 24 | 23 | Withdrawal during month 23 |

**Example:**
```python
event = WithdrawalEvent("Account", 100_000, date(2025, 6, 1))
month = event.resolve_month(date(2025, 1, 1))  # → 6
# Array index = month - 1 = 5
# D[5, m] receives the withdrawal
```

---

## Integration with Portfolio

### Basic usage

```python
from datetime import date
from finopt.src.portfolio import Account, Portfolio
from finopt.src.withdrawal import WithdrawalSchedule, WithdrawalEvent

# Define accounts
accounts = [
    Account.from_annual("Conservador", 0.06, 0.08),
    Account.from_annual("Agresivo", 0.12, 0.15)
]
portfolio = Portfolio(accounts)

# Define withdrawals
schedule = WithdrawalSchedule(events=[
    WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
    WithdrawalEvent("Agresivo", 2_000_000, date(2026, 12, 1))
])

# Convert to array
D = schedule.to_array(T=36, start_date=date(2025, 1, 1), accounts=accounts)

# Simulate with withdrawals
result = portfolio.simulate(A=A, R=R, X=X, D=D)
```

### With stochastic withdrawals

```python
from finopt.src.withdrawal import WithdrawalModel, StochasticWithdrawal

model = WithdrawalModel(
    scheduled=schedule,
    stochastic=[
        StochasticWithdrawal(
            account="Conservador",
            base_amount=200_000,
            sigma=30_000,
            date=date(2025, 9, 1),
            floor=100_000,
            cap=400_000
        )
    ]
)

# Generate stochastic withdrawal scenarios
D = model.to_array(
    T=36,
    start_date=date(2025, 1, 1),
    accounts=accounts,
    n_sims=500,
    seed=42
)
# D.shape → (500, 36, 2)

# Simulate
result = portfolio.simulate(A=A, R=R, X=X, D=D)
```

---

## Integration with Optimization

### CVaROptimizer with withdrawals

```python
from finopt.src.optimization import CVaROptimizer, GoalSeeker

optimizer = CVaROptimizer(n_accounts=2, objective='balanced')

# Define D_generator for GoalSeeker
def D_gen(T, n_sims, seed):
    return model.to_array(
        T=T,
        start_date=date(2025, 1, 1),
        accounts=accounts,
        n_sims=n_sims,
        seed=seed
    )

seeker = GoalSeeker(optimizer, T_max=120)
result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=initial_wealth,
    accounts=accounts,
    start_date=date(2025, 1, 1),
    n_sims=500,
    seed=42,
    D_generator=D_gen,
    withdrawal_epsilon=0.05  # 95% confidence for withdrawal feasibility
)
```

### Withdrawal feasibility constraints

The optimizer adds CVaR constraints to ensure sufficient wealth before each withdrawal:

$$
\mathbb{P}(W_t^m \geq D_t^m) \geq 1 - \epsilon
$$

CVaR reformulation:
$$
\text{CVaR}_\epsilon(D_t^m - W_t^m) \leq 0
$$

**Default:** `withdrawal_epsilon=0.05` (95% confidence of meeting withdrawals)

---

## Complete Example

```python
from datetime import date
import numpy as np
from finopt.src.portfolio import Account, Portfolio
from finopt.src.withdrawal import (
    WithdrawalEvent, WithdrawalSchedule,
    StochasticWithdrawal, WithdrawalModel
)
from finopt.src.goals import TerminalGoal
from finopt.src.optimization import CVaROptimizer, GoalSeeker

# 1. Define accounts
accounts = [
    Account.from_annual("Conservador", 0.06, 0.08,
                        display_name="Fondo Conservador"),
    Account.from_annual("Agresivo", 0.12, 0.15,
                        display_name="Fondo Agresivo")
]

# 2. Define withdrawals
withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule(events=[
        WithdrawalEvent(
            account="Conservador",
            amount=400_000,
            date=date(2025, 6, 1),
            description="Compra bicicleta"
        ),
        WithdrawalEvent(
            account="Agresivo",
            amount=5_000_000,
            date=date(2027, 1, 1),
            description="Pie departamento"
        )
    ]),
    stochastic=[
        StochasticWithdrawal(
            account="Conservador",
            base_amount=200_000,
            sigma=50_000,
            date=date(2025, 12, 1),
            floor=100_000,
            cap=400_000
        )
    ]
)

# 3. Define goals
goals = [
    TerminalGoal(account="Conservador", threshold=10_000_000, confidence=0.90),
    TerminalGoal(account="Agresivo", threshold=30_000_000, confidence=0.85)
]

# 4. Setup optimization
start_date = date(2025, 1, 1)
initial_wealth = np.array([2_000_000, 5_000_000])

def A_gen(T, n_sims, seed):
    return np.full((n_sims, T), 500_000)

def R_gen(T, n_sims, seed):
    np.random.seed(seed)
    return np.random.normal(0.005, 0.02, (n_sims, T, 2))

def D_gen(T, n_sims, seed):
    return withdrawals.to_array(T, start_date, accounts, n_sims, seed)

# 5. Optimize
optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
seeker = GoalSeeker(optimizer, T_max=60, verbose=True)

result = seeker.seek(
    goals=goals,
    A_generator=A_gen,
    R_generator=R_gen,
    initial_wealth=initial_wealth,
    accounts=accounts,
    start_date=start_date,
    n_sims=500,
    seed=42,
    D_generator=D_gen,
    withdrawal_epsilon=0.05
)

print(f"Optimal horizon: T*={result.T} months")
print(f"Expected withdrawals: {withdrawals.total_expected(accounts)}")
```

---

## Serialization

Both `WithdrawalSchedule` and `WithdrawalModel` support JSON serialization:

```python
# Serialize
data = model.to_dict()
# {
#     "scheduled": {"events": [...]},
#     "stochastic": [...]
# }

# Save to file
import json
with open("withdrawals.json", "w") as f:
    json.dump(data, f, indent=2)

# Load from file
with open("withdrawals.json", "r") as f:
    data = json.load(f)
model = WithdrawalModel.from_dict(data)
```

---

## Comparison with Other Modules

| Module | Concept | Pattern |
|--------|---------|---------|
| `income.py` | `FixedIncome` | Deterministic contributions |
| `withdrawal.py` | `WithdrawalEvent` | Deterministic withdrawals |
| `income.py` | `VariableIncome` | Stochastic contributions |
| `withdrawal.py` | `StochasticWithdrawal` | Stochastic withdrawals |
| `income.py` | `IncomeModel` | Facade (fixed + variable) |
| `withdrawal.py` | `WithdrawalModel` | Facade (scheduled + stochastic) |
| `goals.py` | `IntermediateGoal` | Fixed-date constraint |
| `withdrawal.py` | `WithdrawalEvent` | Fixed-date withdrawal |

---

## API Summary

| Class | Type | Purpose |
|-------|------|---------|
| `WithdrawalEvent` | frozen dataclass | Single scheduled withdrawal |
| `WithdrawalSchedule` | dataclass | Collection of scheduled withdrawals |
| `StochasticWithdrawal` | frozen dataclass | Withdrawal with uncertainty |
| `WithdrawalModel` | dataclass | Unified facade |

**Key methods:**
- `resolve_month(start_date)` — Convert date to 1-indexed month offset
- `to_array(T, start_date, accounts, ...)` — Generate numpy array for simulation
- `total_by_account(accounts)` / `total_expected(accounts)` — Summarize withdrawals
- `to_dict()` / `from_dict(payload)` — Serialization
