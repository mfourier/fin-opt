# Quick Start Guide

This guide will help you get started with FinOpt in under 5 minutes.

## Installation

=== "Conda (Recommended)"

    ```bash
    conda env create -f environment.yml
    conda activate finance
    pip install cvxpy
    ```

=== "Pip"

    ```bash
    pip install numpy pandas cvxpy matplotlib
    ```

## Step 1: Define Your Income

FinOpt models your cash flows—both fixed salary and variable bonuses.

```python
from finopt import IncomeModel, FixedIncome, VariableIncome
from datetime import date

# Fixed salary with 3% annual growth
fixed = FixedIncome(
    base=1_500_000,        # Monthly base salary (CLP)
    annual_growth=0.03,    # 3% annual raise
    start_date=date(2025, 1, 1)
)

# Variable income (bonuses) with seasonality
variable = VariableIncome(
    expected=500_000,      # Expected monthly bonus
    volatility=0.3,        # 30% standard deviation
    seasonality=[0.5, 0.5, 1.0, 0.5, 0.5, 2.0,   # Jan-Jun
                 0.5, 0.5, 1.0, 0.5, 0.5, 3.0],  # Jul-Dec (Dec = 3x)
    seed=42
)

income = IncomeModel(fixed=fixed, variable=variable)
```

## Step 2: Set Up Investment Accounts

Define your portfolio accounts with expected returns and volatility.

```python
from finopt import Account

accounts = [
    Account.from_annual(
        name="Conservative",
        annual_return=0.08,      # 8% expected annual return
        annual_volatility=0.09,  # 9% volatility
        initial_wealth=1_000_000
    ),
    Account.from_annual(
        name="Aggressive",
        annual_return=0.14,      # 14% expected annual return
        annual_volatility=0.18,  # 18% volatility
        initial_wealth=500_000
    )
]
```

!!! tip "Always use `from_annual()`"
    The `from_annual()` constructor automatically converts annual parameters to monthly equivalents, which is required for simulation.

## Step 3: Create the Model

Combine income and accounts into a unified model.

```python
from finopt import FinancialModel

model = FinancialModel(
    income=income,
    accounts=accounts,
    correlation=0.3  # Correlation between account returns
)
```

## Step 4: Define Your Goals

Specify what you want to achieve and by when.

```python
from finopt.goals import TerminalGoal, IntermediateGoal
from datetime import date

goals = [
    # Emergency fund: $3M in conservative account by month 12
    IntermediateGoal(
        account="Conservative",
        threshold=3_000_000,
        confidence=0.90,        # 90% probability of success
        date=date(2026, 1, 1)
    ),
    # Long-term wealth: $10M at end of planning horizon
    TerminalGoal(
        account="Aggressive",
        threshold=10_000_000,
        confidence=0.80         # 80% probability of success
    )
]
```

## Step 5: Optimize

Find the minimum time horizon and optimal allocation policy.

```python
from finopt.optimization import CVaROptimizer

optimizer = CVaROptimizer(
    n_accounts=2,
    objective="balanced"  # Minimizes allocation changes
)

result = model.optimize(
    goals=goals,
    optimizer=optimizer,
    T_max=120,      # Maximum 10-year horizon
    n_sims=500,     # Monte Carlo scenarios
    seed=42
)

print(f"Minimum horizon: {result.T} months")
print(f"Goal satisfaction: {result.validate_goals()}")
```

## Step 6: Visualize Results

```python
# Wealth trajectories
model.plot("wealth", result=result, show_trajectories=True)

# Allocation policy heatmap
model.plot("allocation", result=result)

# Combined dashboard
model.plot("combined", result=result)
```

## What's Next?

- **[Framework Overview](framework.md)**: Understand the mathematical foundations
- **[Optimization Guide](optimization.md)**: Learn about CVaR reformulation and different objectives
- **[Goals Framework](goals.md)**: Advanced goal configuration with chance constraints
