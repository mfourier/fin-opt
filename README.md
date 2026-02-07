# FinOpt

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **Minimum-horizon portfolio optimization with probabilistic goal guarantees**

Find the shortest investment horizon to achieve your financial goals using CVaR-reformulated convex programming and Monte Carlo simulation.

**Why FinOpt?** Traditional tools either use heuristics (suboptimal), gradient descent (local minima), or simulation without optimization (no decisions). FinOpt combines stochastic simulation with convex optimization for **globally optimal allocation policies**.

![Wealth Dynamics](docs/images/wealth_dynamics.png)
*Monte Carlo wealth trajectories under optimal allocation policy with goal achievement markers*

---

## What is FinOpt?

FinOpt is a **full-stack goal-based portfolio optimization platform** that solves the bilevel problem of finding the minimum investment horizon to achieve multiple financial goals with probabilistic guarantees.

### The Core Question

**Traditional approach**: *"Given horizon T, what wealth can I achieve?"*

**FinOpt's approach**: *"What is the minimum horizon T* to achieve my goals with high probability?"*

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FinOpt Platform                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Frontend   │◄──►│   Backend    │◄──►│      Database        │  │
│  │  React/Vite  │    │   FastAPI    │    │      Supabase        │  │
│  │              │    │              │    │                      │  │
│  │ - Profiles   │    │ - /simulate  │    │ - profiles           │  │
│  │ - Scenarios  │    │ - /optimize  │    │ - scenarios          │  │
│  │ - Results    │    │ - /health    │    │ - jobs               │  │
│  │ - Dashboard  │    │              │    │ - results            │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 Mathematical Core (finopt)                    │  │
│  │                                                               │  │
│  │  income.py → portfolio.py → model.py → optimization.py       │  │
│  │      ↓           ↓              ↓                            │  │
│  │  returns.py ─────┘              │                            │  │
│  │                                 │                            │  │
│  │  goals.py ──────────────────────┘                            │  │
│  │                                                               │  │
│  │  Supporting: config.py, serialization.py, cli.py             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Mathematical Core
- **CVaR Reformulation**: Transforms non-convex chance constraints into tractable convex form
- **Affine Wealth Dynamics**: Closed-form representation enabling analytical gradients
- **Global Optimality**: DCP-compliant objectives via CVXPY eliminate local minima
- **Multiple Goals**: Intermediate (fixed time) + terminal (optimized horizon)
- **Scheduled & Stochastic Withdrawals**: Calendar-based cash outflows with feasibility constraints
- **Stochastic Income**: Seasonality, noise, growth, floor/cap constraints
- **Correlated Returns**: Multi-asset portfolios with lognormal shocks

### Web Application
- **User Authentication**: Secure login via Supabase Auth
- **Profile Management**: Configure income sources, accounts, correlations
- **Scenario Builder**: Define goals, withdrawals, optimization parameters
- **Real-time Job Tracking**: Monitor simulation/optimization progress
- **Results Visualization**: Allocation heatmaps, wealth trajectories, goal status

### API
- **Background Job Processing**: Async simulation and optimization
- **Supabase Integration**: CRUD via PostgREST, compute via FastAPI
- **Health Monitoring**: Ready for production deployment

---

## Quick Start

### Option 1: Use as Python Library

For researchers and developers who want to use the mathematical core directly:

```bash
# Clone and setup
git clone https://github.com/mfourier/finopt.git
cd finopt

# Using Conda (recommended)
conda env create -f environment.yml
conda activate finance
pip install -e .
```

```python
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.goals import TerminalGoal
from finopt.optimization import CVaROptimizer
from datetime import date

# 1. Define income stream
income = IncomeModel(
    fixed=FixedIncome(
        base=1_500_000,
        annual_growth=0.03,
        salary_raises={date(2026, 6, 1): 500_000}
    )
)

# 2. Configure accounts
accounts = [
    Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09),
    Account.from_annual("Aggressive", annual_return=0.14, annual_volatility=0.15)
]

# 3. Create model and define goals
model = FinancialModel(income, accounts)
goals = [TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.80)]

# 4. Optimize
optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
result = model.optimize(goals=goals, optimizer=optimizer, T_max=120, n_sims=500, seed=42)

print(f"Minimum horizon: T*={result.T} months")
model.plot("wealth", result=result, show_trajectories=True)
```

### Option 2: Run Full Application

For end-users who want the complete web experience:

```bash
# Clone repository
git clone https://github.com/mfourier/finopt.git
cd finopt

# Setup environment variables
cp .env.example .env
# Edit .env with your Supabase credentials

# Start with Docker Compose
docker-compose up -d

# Or run services individually:
# Backend
cd api && uvicorn main:app --reload

# Frontend
cd web && npm install && npm run dev
```

Access the application at `http://localhost:5173`

---

## Project Structure

```
finopt/
├── src/finopt/              # Mathematical core library
│   ├── income.py            # Cash flow modeling (fixed + variable)
│   ├── returns.py           # Correlated lognormal returns
│   ├── portfolio.py         # Wealth dynamics executor
│   ├── goals.py             # Goal specifications (intermediate + terminal)
│   ├── optimization.py      # CVaR optimizer + GoalSeeker
│   ├── model.py             # FinancialModel facade
│   ├── withdrawal.py        # Scheduled + stochastic withdrawals
│   ├── config.py            # Pydantic configuration
│   ├── serialization.py     # JSON persistence
│   └── cli.py               # Command-line interface
│
├── api/                     # FastAPI backend
│   ├── main.py              # API endpoints (/simulate, /optimize, /health)
│   ├── config.py            # Settings management
│   ├── supabase_client.py   # Database client
│   └── services/            # Business logic
│       ├── simulation.py    # Monte Carlo simulation service
│       ├── optimization.py  # CVaR optimization service
│       └── reconstruction.py # Model reconstruction from DB
│
├── web/                     # React frontend
│   ├── src/
│   │   ├── pages/           # LoginPage, Dashboard, Profiles, Scenarios, Results
│   │   ├── components/      # Layout, AllocationHeatmap
│   │   ├── lib/             # Supabase client, API calls, store
│   │   └── types/           # TypeScript definitions
│   └── package.json
│
├── supabase/                # Database schema
│   └── migrations/          # SQL migrations
│
├── docker/                  # Docker configurations
├── docs/                    # MkDocs documentation
├── notebooks/               # Jupyter notebooks
└── tests/                   # Unit and integration tests
```

---

## Mathematical Foundation

### Bilevel Optimization Problem

```
Outer: min T                              # Find shortest horizon
Inner: max f(X)                           # Objective (wealth, turnover, risk-adjusted)
       s.t. X ∈ Simplex                   # Allocation policy
            P(W_t ≥ goal_t) ≥ 1-ε, ∀ goals
            P(W_t ≥ D_t) ≥ 1-δ, ∀ withdrawals
```

### CVaR Reformulation

Original (non-convex): P(W_t ≥ b) ≥ 1-ε

CVaR form (convex): CVaR_ε(b - W_t) ≤ 0

Based on Rockafellar & Uryasev (2000), this provides global optimality guarantees.

### Affine Wealth Representation

```
W_t^m(X) = W_0^m · F_{0,t}^m + Σ_{s=0}^{t-1} (A_s · x_s^m - D_s^m) · F_{s,t}^m
```

Where F_{s,t}^m are accumulation factors. This linearity in X enables efficient convex optimization.

---

## Development

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Integration tests only
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/ api/

# Type checking
mypy src/finopt/

# Linting
ruff check src/ tests/
```

### Documentation

```bash
# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

---

## Roadmap

### Phase 1: Core Improvements (In Progress)

- [ ] **Historical Backtesting**: Validate with real fund return data
- [ ] **Robust Optimization**: Uncertainty sets for (μ, σ) parameters
- [ ] **Tax-Aware Optimization**: After-tax wealth maximization
- [ ] **Rebalancing Constraints**: Transaction costs, minimum trade sizes

### Phase 2: Platform Features

- [ ] **Multi-Currency Support**: Handle portfolios across currencies
- [ ] **PDF Reports**: Exportable optimization reports
- [ ] **Scenario Comparison**: Side-by-side analysis of multiple scenarios
- [ ] **What-If Analysis**: Sensitivity to parameter changes
- [ ] **Goal Templates**: Pre-configured goals (retirement, house, education)

### Phase 3: Advanced Capabilities

- [ ] **Real-Time Data Integration**: Live market data feeds
- [ ] **Portfolio Monitoring**: Track actual vs planned allocation
- [ ] **Alerts & Notifications**: Goal progress, rebalancing reminders
- [ ] **API for External Tools**: REST API for third-party integrations
- [ ] **Mobile App**: React Native companion app

### Phase 4: Enterprise Features

- [ ] **Multi-User Collaboration**: Shared scenarios, team workspaces
- [ ] **Advisor Dashboard**: Financial advisor tools
- [ ] **Audit Trail**: Compliance and regulatory logging
- [ ] **Custom Objectives**: User-defined optimization objectives
- [ ] **White-Label Solution**: Customizable branding

---

## Deployment

### Render (Current)

The application is configured for Render deployment via `render.yaml`:
- **API**: Python web service
- **Frontend**: Static site with Vite build
- **Database**: Supabase (external)

### Docker

```bash
# Build and run all services
docker-compose up -d

# Build individual images
docker build -f docker/Dockerfile.api -t finopt-api .
docker build -f docker/Dockerfile.web -t finopt-web .
```

---

## References

**Optimization Theory**:
- Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk". *Journal of Risk*, 2(3), 21-41.
- Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2014). *Lectures on Stochastic Programming*. SIAM.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure tests pass (`pytest tests/`)
4. Format code (`black src/ tests/`)
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Maximiliano Lioi**
M.Sc. Applied Mathematics | Quantitative Finance | Data Science

[GitHub](https://github.com/mfourier) | [LinkedIn](https://linkedin.com/in/mlioi)

---

*This software is for educational and research purposes. Not financial advice. Consult certified professionals before making investment decisions.*
