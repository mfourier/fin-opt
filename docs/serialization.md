# `serialization` — Model Persistence for FinOpt

> **Core idea:** Provide JSON serialization and deserialization for FinOpt models, enabling configuration persistence, sharing, version control, and reproducibility.
> `serialization.py` bridges the gap between Python objects and human-readable files, allowing models to be saved, loaded, and shared across sessions.

---

## Why a dedicated serialization module?

- **Persistence:** Save model configurations and optimization results to disk
- **Reproducibility:** Capture all parameters needed to recreate experiments
- **Sharing:** Exchange configurations between users or systems
- **Version control:** Track model changes with git-friendly JSON format
- **Validation:** Type-safe loading via Pydantic configs

---

## Design principles

1. **Type-safe serialization**
   - Uses Pydantic configs from `config.py` for validation during loading
   - Catches malformed or invalid configurations early

2. **Human-readable format**
   - JSON output with pretty-printing (2-space indentation)
   - Date fields stored as ISO format strings (`"2025-06-01"`)

3. **Reproducibility first**
   - Includes seeds, all parameters, and contribution rates
   - Schema versioning for backward compatibility tracking

4. **Modular architecture**
   - Individual serializers for each component (accounts, income, goals, withdrawals)
   - Composable into full model or scenario serialization

5. **Backward compatible**
   - Schema version checking with warnings
   - Supports legacy formats (e.g., `month` → `date` conversion for goals)

---

## Schema Version

```python
SCHEMA_VERSION = "0.2.0"
```

All serialized files include a `schema_version` field. When loading files with different versions, a warning is issued but loading proceeds.

---

## Core Functions

### Model Serialization

#### `save_model(model, path, include_correlation=True)`

Saves `FinancialModel` configuration to JSON file.

```python
from pathlib import Path
from finopt.serialization import save_model

save_model(model, Path("config.json"))
```

**Parameters:**
- `model`: `FinancialModel` instance to save
- `path`: Output file path (creates parent directories if needed)
- `include_correlation`: Whether to include return correlation matrix (default: `True`)

**Output format:**
```json
{
  "schema_version": "0.2.0",
  "income": {
    "fixed": {
      "base": 1400000.0,
      "annual_growth": 0.03,
      "salary_raises": {
        "2025-07-01": 200000.0
      }
    },
    "variable": {
      "base": 200000.0,
      "sigma": 0.15,
      "annual_growth": 0.02,
      "seasonality": [1.0, 0.95, 1.05, ...],
      "floor": 50000.0,
      "cap": 400000.0,
      "seed": 42
    },
    "contribution_rate_fixed": 0.3,
    "contribution_rate_variable": 1.0
  },
  "accounts": [
    {
      "name": "Conservador",
      "annual_return": 0.06,
      "annual_volatility": 0.08,
      "initial_wealth": 1000000.0,
      "display_name": "Fondo Conservador"
    },
    {
      "name": "Agresivo",
      "annual_return": 0.12,
      "annual_volatility": 0.15,
      "initial_wealth": 500000.0
    }
  ],
  "correlation": [
    [1.0, 0.3],
    [0.3, 1.0]
  ]
}
```

---

#### `load_model(path) -> FinancialModel`

Reconstructs `FinancialModel` from JSON configuration file.

```python
from pathlib import Path
from finopt.serialization import load_model

model = load_model(Path("config.json"))
```

**Parameters:**
- `path`: Input file path

**Returns:**
- `FinancialModel`: Fully reconstructed model instance

**Notes:**
- Validates configuration using Pydantic configs
- Issues warning if schema version differs from current
- Automatically sets correlation matrix if present

---

### Optimization Result Serialization

#### `save_optimization_result(result, path, include_policy=True)`

Saves `OptimizationResult` to JSON file.

```python
from pathlib import Path
from finopt.serialization import save_optimization_result

save_optimization_result(result, Path("optimal_policy.json"))
```

**Parameters:**
- `result`: `OptimizationResult` instance
- `path`: Output file path
- `include_policy`: Whether to include full allocation policy matrix `X` (default: `True`)

**Output format:**
```json
{
  "schema_version": "0.2.0",
  "T": 36,
  "objective_value": 0.0023,
  "feasible": true,
  "solve_time": 1.234,
  "n_iterations": 5,
  "X": [
    [0.4, 0.6],
    [0.35, 0.65],
    ...
  ],
  "goals": [
    {
      "type": "intermediate",
      "threshold": 5000000.0,
      "confidence": 0.9,
      "account": "Conservador",
      "date": "2025-07-01"
    },
    {
      "type": "terminal",
      "threshold": 30000000.0,
      "confidence": 0.85,
      "account": "Agresivo",
      "date": null
    }
  ]
}
```

---

#### `load_optimization_result(path) -> Dict[str, Any]`

Loads optimization result from JSON file.

```python
from pathlib import Path
import numpy as np
from finopt.serialization import load_optimization_result

result_data = load_optimization_result(Path("optimal_policy.json"))
X = result_data["X"]  # Already converted to np.ndarray
T = result_data["T"]
```

**Parameters:**
- `path`: Input file path

**Returns:**
- `dict`: Dictionary with optimization result data (not full `OptimizationResult` object)

**Note:** Returns dictionary instead of `OptimizationResult` because full reconstruction requires `SimulationResult` context.

---

### Scenario Serialization

A **scenario** captures everything needed to reproduce an optimization: model configuration (or reference), goals, withdrawals, and simulation/optimization parameters.

#### `save_scenario(...)`

Saves a complete optimization scenario to JSON file.

```python
from pathlib import Path
from datetime import date
from finopt.serialization import save_scenario
from finopt.goals import TerminalGoal, IntermediateGoal

goals = [
    IntermediateGoal(account="Conservador", threshold=5_000_000,
                     confidence=0.9, date=date(2025, 7, 1)),
    TerminalGoal(account="Agresivo", threshold=30_000_000, confidence=0.85)
]

# Option 1: Embed model in scenario
save_scenario(
    scenario_name="Plan de Retiro",
    goals=goals,
    path=Path("scenarios/retirement.json"),
    model=my_model,
    withdrawals=my_withdrawals,
    start_date=date(2025, 1, 1),
    description="Escenario base para jubilación",
    n_sims=1000,
    seed=42,
    T_max=120,
    solver="CLARABEL",
    objective="balanced"
)

# Option 2: Reference external model file
save_scenario(
    scenario_name="Plan de Retiro",
    goals=goals,
    path=Path("scenarios/retirement.json"),
    model_path="profiles/my_profile.json",  # relative to scenario file
    start_date=date(2025, 1, 1)
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `scenario_name` | `str` | Human-readable scenario name |
| `goals` | `List[Goal]` | Financial goals for the scenario |
| `path` | `Path` | Output file path |
| `model` | `FinancialModel` | Embed full model (mutually exclusive with `model_path`) |
| `model_path` | `str` | Reference external model file (mutually exclusive with `model`) |
| `withdrawals` | `WithdrawalModel` | Scheduled and stochastic withdrawals |
| `start_date` | `date` | Simulation start date (default: today) |
| `description` | `str` | Optional scenario description |
| `n_sims` | `int` | Number of Monte Carlo simulations (default: 500) |
| `seed` | `int` | Random seed for reproducibility |
| `T_max` | `int` | Maximum optimization horizon (default: 240) |
| `solver` | `str` | CVXPY solver backend (default: `"ECOS"`) |
| `objective` | `str` | Optimization objective (default: `"balanced"`) |

**Output format:**
```json
{
  "schema_version": "0.2.0",
  "name": "Plan de Retiro",
  "description": "Escenario base para jubilación",
  "start_date": "2025-01-01",
  "model": {
    "income": {...},
    "accounts": [...],
    "correlation": [...]
  },
  "intermediate_goals": [
    {"account": "Conservador", "threshold": 5000000.0, "confidence": 0.9, "date": "2025-07-01"}
  ],
  "terminal_goals": [
    {"account": "Agresivo", "threshold": 30000000.0, "confidence": 0.85}
  ],
  "withdrawals": {
    "scheduled": [...],
    "stochastic": [...]
  },
  "simulation": {
    "n_sims": 1000,
    "seed": 42,
    "cache_enabled": true,
    "verbose": true
  },
  "optimization": {
    "T_max": 120,
    "solver": "CLARABEL",
    "objective": "balanced"
  }
}
```

---

#### `load_scenario(path, load_model_from_path=True) -> Dict[str, Any]`

Loads a scenario from JSON file.

```python
from pathlib import Path
from finopt.serialization import load_scenario

scenario = load_scenario(Path("scenarios/retirement.json"))

# Access components
model = scenario["model"]           # FinancialModel
goals = scenario["goals"]           # List[IntermediateGoal | TerminalGoal]
withdrawals = scenario["withdrawals"]  # WithdrawalModel or None
start_date = scenario["start_date"]    # date
sim_config = scenario["simulation"]    # SimulationConfig
opt_config = scenario["optimization"]  # OptimizationConfig
```

**Parameters:**
- `path`: Input file path
- `load_model_from_path`: If scenario has `model_path`, load the model from that file (default: `True`)

**Returns:**
- `dict` with keys:
  - `name`: `str`
  - `description`: `str`
  - `start_date`: `date`
  - `model`: `FinancialModel` (if embedded or loaded from path)
  - `model_path`: `str` (if referenced)
  - `goals`: `List[IntermediateGoal | TerminalGoal]`
  - `withdrawals`: `WithdrawalModel` or `None`
  - `simulation`: `SimulationConfig`
  - `optimization`: `OptimizationConfig`

---

### Component Serializers

#### Account Serialization

```python
from finopt.serialization import account_to_dict, account_from_dict

# Serialize
data = account_to_dict(account)
# {"name": "Conservador", "annual_return": 0.06, "annual_volatility": 0.08, ...}

# Deserialize
account = account_from_dict(data)
```

---

#### Income Serialization

```python
from finopt.serialization import income_to_dict, income_from_dict

# Serialize
data = income_to_dict(income_model)

# Deserialize
income = income_from_dict(data)
```

**Features:**
- Handles `None` components (fixed-only or variable-only)
- Converts `salary_raises` dates to ISO strings
- Supports both scalar and 12-element array contribution rates

---

#### Withdrawal Serialization

```python
from finopt.serialization import withdrawal_to_dict, withdrawal_from_dict

# Serialize
data = withdrawal_to_dict(withdrawal_model)
# {
#   "scheduled": [{"account": "...", "amount": 100000, "date": "2025-06-01"}],
#   "stochastic": [{"account": "...", "base_amount": 200000, "sigma": 50000, ...}]
# }

# Deserialize
withdrawals = withdrawal_from_dict(data)
```

**Features:**
- Handles both scheduled (`WithdrawalEvent`) and stochastic (`StochasticWithdrawal`)
- Supports `month` or `date` timing for stochastic withdrawals

---

#### Goal Serialization

```python
from finopt.serialization import goals_to_dict, goals_from_dict

# Serialize
data = goals_to_dict(goals)
# {
#   "intermediate": [{"account": "...", "threshold": 5000000, "confidence": 0.9, "date": "2025-07-01"}],
#   "terminal": [{"account": "...", "threshold": 30000000, "confidence": 0.85}]
# }

# Deserialize
goals = goals_from_dict(data, start_date=date(2025, 1, 1))
```

**Features:**
- Separates intermediate and terminal goals
- Backward compatible: converts legacy `month` format to `date` with deprecation warning

---

## Integration with Config Module

The serialization module relies on Pydantic configs from `config.py` for validation:

| Serializer | Config Class |
|------------|--------------|
| `account_from_dict` | `AccountConfig` |
| `income_from_dict` | `IncomeConfig`, `FixedIncomeConfig`, `VariableIncomeConfig` |
| `withdrawal_from_dict` | `WithdrawalConfig`, `WithdrawalEventConfig`, `StochasticWithdrawalConfig` |
| `goals_from_dict` | `IntermediateGoalConfig`, `TerminalGoalConfig` |
| `load_scenario` | `SimulationConfig`, `OptimizationConfig` |

This ensures all loaded configurations are validated against defined schemas before being used to construct Python objects.

---

## Usage Patterns

### A) Save and load model configuration

```python
from pathlib import Path
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.serialization import save_model, load_model

# Create model
income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
accounts = [
    Account.from_annual("Conservador", 0.06, 0.08),
    Account.from_annual("Agresivo", 0.12, 0.15)
]
model = FinancialModel(income, accounts)

# Save
save_model(model, Path("configs/my_profile.json"))

# Load in another session
loaded_model = load_model(Path("configs/my_profile.json"))
```

---

### B) Save optimization results for later analysis

```python
from pathlib import Path
from finopt.serialization import save_optimization_result, load_optimization_result

# After optimization
result = model.optimize(goals=goals, ...)

# Save
save_optimization_result(result, Path("results/optimal_policy.json"))

# Load later
data = load_optimization_result(Path("results/optimal_policy.json"))
X = data["X"]  # np.ndarray
T = data["T"]  # int
```

---

### C) Create reproducible experiment scenarios

```python
from pathlib import Path
from datetime import date
from finopt.serialization import save_scenario, load_scenario
from finopt.goals import TerminalGoal
from finopt.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent

# Define scenario
goals = [TerminalGoal(account="Agresivo", threshold=50_000_000, confidence=0.85)]
withdrawals = WithdrawalModel(
    scheduled=WithdrawalSchedule([
        WithdrawalEvent("Conservador", 5_000_000, date(2027, 1, 1), "Pie departamento")
    ])
)

# Save complete scenario
save_scenario(
    scenario_name="Casa + Jubilación",
    goals=goals,
    path=Path("scenarios/casa_jubilacion.json"),
    model=model,
    withdrawals=withdrawals,
    start_date=date(2025, 1, 1),
    n_sims=1000,
    seed=42,
    T_max=120
)

# Load and run
scenario = load_scenario(Path("scenarios/casa_jubilacion.json"))
result = scenario["model"].optimize(
    goals=scenario["goals"],
    start=scenario["start_date"],
    withdrawals=scenario["withdrawals"],
    n_sims=scenario["simulation"].n_sims,
    seed=scenario["simulation"].seed,
    T_max=scenario["optimization"].T_max
)
```

---

### D) Reference external model files

```python
from pathlib import Path
from finopt.serialization import save_scenario, load_scenario

# Save scenario referencing external model
save_scenario(
    scenario_name="Variante Conservadora",
    goals=goals,
    path=Path("scenarios/conservative.json"),
    model_path="../profiles/my_profile.json"  # relative path
)

# Load resolves the reference automatically
scenario = load_scenario(Path("scenarios/conservative.json"))
model = scenario["model"]  # Loaded from referenced file
```

---

## Backward Compatibility

### Schema versioning

```python
# When loading files with different schema versions
>>> model = load_model(Path("old_config.json"))
UserWarning: Config schema version 0.1.0 differs from current version 0.2.0.
May encounter compatibility issues.
```

### Legacy goal format

The `month` field in intermediate goals is deprecated in favor of `date`:

```python
# Old format (deprecated)
{"account": "Savings", "threshold": 1000000, "confidence": 0.9, "month": 6}

# New format
{"account": "Savings", "threshold": 1000000, "confidence": 0.9, "date": "2025-07-01"}
```

When loading old format, a deprecation warning is issued and the month is converted to a date.

---

## API Summary

| Function | Purpose |
|----------|---------|
| `save_model(model, path)` | Save `FinancialModel` to JSON |
| `load_model(path)` | Load `FinancialModel` from JSON |
| `save_optimization_result(result, path)` | Save `OptimizationResult` to JSON |
| `load_optimization_result(path)` | Load optimization result as dict |
| `save_scenario(...)` | Save complete scenario with all parameters |
| `load_scenario(path)` | Load scenario with reconstructed objects |
| `account_to_dict(account)` | Serialize single `Account` |
| `account_from_dict(data)` | Deserialize single `Account` |
| `income_to_dict(income_model)` | Serialize `IncomeModel` |
| `income_from_dict(data)` | Deserialize `IncomeModel` |
| `withdrawal_to_dict(withdrawal_model)` | Serialize `WithdrawalModel` |
| `withdrawal_from_dict(data)` | Deserialize `WithdrawalModel` |
| `goals_to_dict(goals)` | Serialize list of goals |
| `goals_from_dict(data)` | Deserialize list of goals |

**Constants:**
- `SCHEMA_VERSION = "0.2.0"` — Current schema version for all serialized files
