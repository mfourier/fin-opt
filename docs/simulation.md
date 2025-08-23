# `simulation` — Philosophy and Role in FinOpt

> **Core idea:** orchestrate the entire simulation flow by connecting **income generation** (`income.py`) with **capital accumulation** (`investment.py`).  
> `simulation.py` acts as the **scenario engine**: it builds contribution paths, generates return scenarios, and produces wealth trajectories with performance metrics.

---

## Why a dedicated orchestrator?

- **Separation of concerns**  
  - `income.py` → how much you can invest (cash inflows).  
  - `investment.py` → how those investments grow (compounding).  
  - `simulation.py` → ties both together in **scenarios**.  

- **Reproducibility**  
  All randomness is controlled with explicit seeds (`set_random_seed`), ensuring that simulations can be replicated.

- **Flexibility**  
  Enables both simple **three-case deterministic setups** (base, optimistic, pessimistic) and more advanced **Monte Carlo simulations**.

- **Scalability**  
  Built as a façade that can later support multi-asset portfolios, expense models, and optimization routines.

---

## Design philosophy

1. **Deterministic by default**  
   Unless Monte Carlo is explicitly enabled, results are fixed and repeatable.

2. **Configuration-driven**  
   All inputs (horizon, start date, return assumptions, contribution ratios, Monte Carlo settings) are encapsulated in `ScenarioConfig`.

3. **Simple but extensible**  
   - MVP: single aggregate asset, three deterministic return paths.  
   - Extension: multiple goals, multi-asset portfolios, advanced return generators.

4. **Structured outputs**  
   Each run yields a `ScenarioResult` containing:
   - `contributions` (Series)
   - `returns` (Series)
   - `wealth` (Series)
   - `metrics` (`PortfolioMetrics` with final wealth, CAGR, volatility, drawdown, etc.)

---

## Core components

### 1) `ScenarioConfig`
A dataclass that centralizes simulation parameters:
- `months`, `start`: horizon and calendar alignment.  
- `alpha_fixed`, `beta_variable`: proportions of fixed/variable income invested.  
- Deterministic returns: `base_r`, `optimistic_r`, `pessimistic_r`.  
- Monte Carlo parameters: `mc_mu`, `mc_sigma`, `mc_paths`, `seed`.

---

### 2) `ScenarioResult`
A dataclass that stores the outcome of a single simulation run:
- `name`: identifier (e.g., `"base"`, `"optimistic"`, `"mc_001"`).  
- `contributions`, `returns`, `wealth`: time series aligned to the calendar.  
- `metrics`: summary performance metrics.

---

### 3) `SimulationEngine`
The high-level orchestrator:
- **`build_contributions()`**: creates monthly contribution paths based on income proportions.  
- **`run_case(name, r)`**: runs a single deterministic scenario with fixed return `r`.  
- **`run_three_cases()`**: executes base/optimistic/pessimistic scenarios in one call.  
- **`run_monte_carlo()`**: generates multiple stochastic paths with IID lognormal returns, each with its own seed, and evaluates wealth outcomes.

---

## Integration in FinOpt workflow

1. **Income generation**  
   Contributions are computed via `IncomeModel` (`income.py`).  
2. **Capital accumulation**  
   Wealth is simulated via `simulate_capital` (`investment.py`) using contributions + return paths.  
3. **Scenario orchestration**  
   `SimulationEngine` coordinates the above, producing consistent results for analysis and optimization.

---

## Typical usage

```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel
from finopt.src.simulation import ScenarioConfig, SimulationEngine

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
