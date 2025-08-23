# FinOpt
*Smart financial planning through simulation and optimization.*

---

## 🚀 Overview
**FinOpt** is a modular Python toolkit to **simulate, plan, and optimize personal investments**.  
It connects **income generation**, **capital growth**, **scenario simulation**, and **goal evaluation** into a unified framework.

---

## ✨ Features

- **Income modeling (`income.py`)**  
  - `FixedIncome`: deterministic salary with optional annual growth.  
  - `VariableIncome`: seasonal + noisy streams (e.g., tutoring, bonuses).  
  - `IncomeModel`: combines streams and derives monthly contributions.  

- **Investment dynamics (`investment.py`)**  
  - `simulate_capital`: aggregate wealth under contributions + returns.  
  - `simulate_portfolio`: multi-asset portfolios with weights/rebalancing.  
  - Scenario generators: `fixed_rate_path`, `lognormal_iid`.  
  - Metrics: CAGR, volatility, max drawdown, final wealth.  

- **Scenario orchestration (`simulation.py`)**  
  - Deterministic cases: base / optimistic / pessimistic returns.  
  - Stochastic Monte Carlo with lognormal returns.  
  - `SimulationEngine`: integrates income + investment into coherent scenarios.  

- **Goal evaluation (`goals.py`)**  
  - Define goals (target + deadline).  
  - Evaluate success, shortfall, attainment ratio.  
  - Allocate contributions across multiple goals.  

- **Optimization solvers (`optimization.py`)**  
  - Minimum constant contribution to reach a target.  
  - Minimum time given a fixed contribution.  
  - Success probabilities across goals (chance constraints).  
  - Extensible solver registry (future LP/QP/MILP).  

- **Utilities (`utils.py`)**  
  - Rate conversions, validation, index builders.  
  - Finance helpers (drawdown, CAGR).  
  - Scenario helpers (rescaling, bootstrapping).  
  - Reporting (`summary_metrics`).  

---

## 📊 Example Usage

### 1. Income projection
```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel

income = IncomeModel(
    fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
    variable=VariableIncome(base=200_000.0, sigma=0.00),
)
df = income.project_monthly(months=12, start=date(2025, 9, 1), as_dataframe=True)
print(df.head())
```
### 2. Simulate scenarios
```python
from finopt.src.simulation import ScenarioConfig, SimulationEngine

cfg = ScenarioConfig(
    months=24, start=date(2025, 9, 1),
    alpha_fixed=0.35, beta_variable=1.0,
    base_r=0.004, optimistic_r=0.007, pessimistic_r=0.001,
)
sim = SimulationEngine(income, cfg)
results = sim.run_three_cases()
print(results["base"].metrics)

```
### 3. Optimize contributions
```python
from finopt.src.optimization import MinContributionInput, min_constant_contribution
from finopt.src.investment import fixed_rate_path

r = fixed_rate_path(24, 0.004)
inp = MinContributionInput(target_amount=20_000_000.0, start_wealth=0.0, returns_path=r)
res = min_constant_contribution(inp)
print("Required contribution:", res.a_star)
```
---

## 📚 Documentation
Detailed documentation for each module is available in the [`docs/`](docs) folder:

- [**income.md**](docs/income.md) — Income modeling (`FixedIncome`, `VariableIncome`, `IncomeModel`).
- [**investment.md**](docs/investment.md) — Capital accumulation, portfolio simulation, and performance metrics.
- [**simulation.md**](docs/simulation.md) — Scenario orchestration (deterministic cases, Monte Carlo).
- [**goals.md**](docs/goals.md) — Goal definition, evaluation, and contribution allocation.
- [**optimization.md**](docs/optimization.md) — Optimization problems (minimum contribution, minimum time, chance constraints).
- [**framework.md**](docs/framework.md) — Theoretical and technical framework of FinOpt.

## 🛠️ Project Structure
```bash
fin-opt/
├── data/                  # Data (raw/processed)
├── docs/                  # Documentation and guides
├── notebooks/             # Exploratory Jupyter notebooks
├── src/                   # Core source code
│   ├── income.py
│   ├── investment.py
│   ├── simulation.py
│   ├── goals.py
│   ├── optimization.py
│   └── utils.py
├── environment.yml        # Conda environment definition
└── README.md              # Project overview
```