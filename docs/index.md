# Welcome to FinOpt

**FinOpt** is a modular Python framework designed for intelligent financial planning. It combines stochastic simulation of income and investment returns with convex optimization to help users achieve their financial goals under uncertainty.

## Key Features

- **Stochastic Income Modeling**: Simulate fixed and variable income with growth, seasonality, and noise.
- **Wealth Dynamics**: Model the evolution of investment accounts using affine wealth representations.
- **Goal-Oriented Optimization**: Find the minimum time to achieve multiple financial goals (e.g., emergency funds, housing) with a specified level of confidence.
- **Bilevel Optimization**: Solve complex problems that minimize time while maximizing terminal wealth.
- **Extensible Architecture**: Modular design allows for custom return models, optimizers, and goal types.

## Documentation Roadmap

Explore the technical components of the FinOpt framework:

### Core Components
- [Stochastic Returns](returns.md): Generating correlated lognormal returns for simulation.
- [Income Module](income.md): Modeling fixed and variable cash flows with growth and seasonality.
- [Portfolio Dynamics](portfolio.md): The mathematics of wealth evolution and affine representations.
- [Scheduled Withdrawals](withdrawal.md): Integration of planned cash outflows into the wealth equation.

### Optimization & Logic
- [Goals Framework](goals.md): Defining financial milestones as probabilistic chance constraints.
- [Optimization](optimization.md): Technical details on CVaR reformulation and Sample Average Approximation (SAA) solvers.

### Integration & Architecture
- [Unified Model](model.md): The `FinancialModel` facade that orchestrates all system components.
- [Technical Framework](framework.md): A deep dive into the system architecture, design principles, and mathematical foundations.

### Advanced & Utilities
- [Exceptions](exceptions.md): Error handling and custom exception classes.
- [Serialization](serialization.md): Saving and loading models and scenarios.
- [Shared Utilities](utils.md): Common helper functions and numerical routines.

## Installation

```bash
pip install -r requirements.txt
```

*(Refer to the repository for more details on setup and usage)*
