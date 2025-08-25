# `scenario` — Philosophy and Role in FinOpt

> **Core idea:** orchestrate investment simulations by connecting **income generation** (`income.py`), **allocation rules** (`investment.py`), and **return paths**, producing wealth trajectories and performance metrics.  
> `scenario.py` acts as the **scenario engine**: it builds contribution paths, allocates them across accounts, normalizes return assumptions, and runs simulations (single-asset or multi-account).

---

## Why a dedicated scenario engine?

- **Separation of concerns**  
  - `income.py` → how much you can invest (cash inflows).  
  - `investment.py` → how those contributions grow (compounding).  
  - `scenario.py` → ties both together in *scenarios*.  

- **Backwards compatibility**  
  Provides the original single-asset API (used in `optimization.py`) while adding modern multi-account support.

- **Flexibility**  
  Supports constant weights or time-varying allocation matrices `C[t,k]`, scalar or path-based returns, and both deterministic and stochastic scenarios.

- **Reproducibility**  
  All randomness is controlled with explicit seeds (`set_random_seed`).

- **Scalability**  
  Handles both MVP cases (single aggregate account, three deterministic paths) and extensions (multi-account, Monte Carlo, custom allocation matrices).

---

## Design philosophy

1. **Deterministic by default**  
   Unless Monte Carlo is explicitly requested, outputs are fixed and repeatable.

2. **Configuration-driven**  
   Parameters (horizon, start date, contribution ratios, return assumptions, MC settings) are encapsulated in `ScenarioConfig`.

3. **Dual API surface**  
   - **Legacy API**: single-asset (`ScenarioConfig`, `ScenarioResult`, `SimulationEngine.run_three_cases`, `run_monte_carlo`).  
   - **Multi-account API**: general runner with account-level contributions, return paths, and wealth series.

4. **Structured outputs**  
   - Single-asset: `ScenarioResult` with series (`contributions`, `returns`, `wealth`) and `PortfolioMetrics`.  
   - Multi-account: `MultiScenarioResult` with per-account DataFrames for contributions, returns, wealth (plus total), and metrics on total wealth.

---

## Core components

### 1) `ScenarioConfig`
Centralizes configuration:
- Horizon and calendar: `months`, `start`.  
- Contribution rule: `alpha_fixed`, `beta_variable`.  
- Deterministic returns: `base_r`, `optimistic_r`, `pessimistic_r`.  
- Monte Carlo: `mc_mu`, `mc_sigma`, `mc_paths`, `seed`.

---

### 2) Results dataclasses
- **`ScenarioResult`**: legacy single-asset results.  
- **`MultiScenarioResult`**: modern multi-account results with:
  - `contributions_total` (Series)  
  - `contributions_by_account` (DataFrame, T×K)  
  - `returns_by_account` (DataFrame, T×K)  
  - `wealth_by_account` (DataFrame, T×K plus `"total"`)  
  - `metrics` (PortfolioMetrics on total wealth)

---

### 3) `SimulationEngine`
The orchestrator, supporting both APIs:

- **Single-asset API (legacy)**  
  - `run_case(name, r)` → one deterministic case.  
  - `run_three_cases()` → base/optimistic/pessimistic.  
  - `run_monte_carlo()` → IID lognormal MC paths.

- **Multi-account API (new)**  
  - `allocate_by_weights(...)` → derive per-account contributions via constant weights or time-varying `C[t,k]`.  
  - `build_returns_by_account(...)` → normalize return specs to a `(T,K)` DataFrame.  
  - `run_case_named(...)` → full simulation producing `MultiScenarioResult`.

---

### 4) Plot helpers
- `plot_scenario(result)` → plots wealth trajectories and contributions by account.  
- `plot_scenarios(results)` → compares total wealth across multiple scenarios.

---

## Integration in FinOpt workflow

1. **Income generation** (`income.py`) → aggregate contributions.  
2. **Allocation** (`investment.py`) → split contributions across accounts using weights or `C[t,k]`.  
3. **Capital accumulation** (`investment.py`) → simulate wealth paths.  
4. **Scenario orchestration** (`scenario.py`) → coordinate the above into structured outputs for analysis and optimization.

---

## Typical usage

### A) Single-asset (legacy)
```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.scenario import ScenarioConfig, SimulationEngine

income = IncomeModel(
    fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
    variable=VariableIncome(base=200_000.0, sigma=0.00),
)

cfg = ScenarioConfig(
    months=36,
    start=date(2025, 9, 1),
    alpha_fixed=0.35,
    beta_variable=1.0,
    base_r=0.004,
    optimistic_r=0.007,
    pessimistic_r=0.001,
)

sim = SimulationEngine(income, cfg)
results = sim.run_three_cases()
print(results["base"].metrics)
```
### B) Multi-account with weights
```python
accounts = ["housing", "emergency"]
sim = SimulationEngine(income, cfg, accounts=accounts)

result = sim.run_case_named(
    "housing_vs_emergency",
    weights_by_account={"housing": 0.6, "emergency": 0.4},
    returns_by_account={"housing": 0.004, "emergency": 0.002},
)

print(result.metrics)
```
### C) Multi-account with time-varying C[t,k]
```python
T = cfg.months
idx = pd.date_range(start=cfg.start, periods=T, freq="MS")
C = np.zeros((T, 2))
C[:, 0] = np.linspace(1.0, 0.5, T)   # shift weight from housing...
C[:, 1] = 1.0 - C[:, 0]              # ...to emergency
C_df = pd.DataFrame(C, index=idx, columns=accounts)

result = sim.run_case_named(
    "glidepath_allocation",
    C_matrix=C_df,
    returns_by_account=0.004,
)

```