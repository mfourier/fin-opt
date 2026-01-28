# Welcome to FinOpt

**FinOpt** is a modular Python framework designed for intelligent financial planning. It combines stochastic simulation of income and investment returns with convex optimization to help users achieve their financial goals under uncertainty.

## Key Features

- **Stochastic Income Modeling**: Simulate fixed and variable income with growth, seasonality, and noise.
- **Wealth Dynamics**: Model the evolution of investment accounts using affine wealth representations.
- **Goal-Oriented Optimization**: Find the minimum time to achieve multiple financial goals (e.g., emergency funds, housing) with a specified level of confidence.
- **Bilevel Optimization**: Solve complex problems that minimize time while maximizing terminal wealth.
- **Extensible Architecture**: Modular design allows for custom return models, optimizers, and goal types.

## Getting Started

To explore the technical details of the framework, check out the following sections:

- [Technical Framework](framework.md): An overview of the system architecture and design principles.
- [Income Module](income.md): How we model cash flows.
- [Portfolio Dynamics](portfolio.md): The math behind wealth evolution.
- [Optimization](optimization.md): Details on the chance-constrained optimization and SAA solvers.

## Installation

```bash
pip install -r requirements.txt
```

*(Refer to the repository for more details on setup and usage)*
