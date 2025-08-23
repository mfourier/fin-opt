# `investment.py` — Philosophy and Role in FinOpt

> **Core idea:** model **how capital grows** when you make monthly contributions and face returns (constant, scenario-based, or simulated).  
> `investment.py` is the **accumulation engine**: it takes monthly contributions (from `income.py` or contribution rules) and return paths, and produces the **wealth trajectory** and **key metrics**.

---

## Why a dedicated module?

Separating *cash-in* from *capital growth* allows for:
- **Clarity:** `income.py` focuses on generating contributions; `investment.py` focuses on how those contributions compound.
- **Traceability:** given a return path and a contribution series, the wealth equation is fully determined.
- **Composability:** integrates with `simulation.py` (scenarios) and optimization modules (e.g., “what contribution do I need?” or “what portfolio weights are optimal?”).

---

## Design Philosophy

1. **Deterministic by default**  
   Nothing is random unless you explicitly request stochastic simulation with a `seed`. This makes comparisons reproducible.

2. **Simple first, extensible later**  
   - Core: `simulate_capital` for an aggregate asset.  
   - Clean extension: `simulate_portfolio` for multi-asset setups with weights and rebalancing.

3. **Consistent calendar indices**  
   Outputs are `pd.Series` aligned to simulation calendars, so results can be plotted and compared seamlessly.

4. **Practical metrics**  
   `compute_metrics` returns a minimal but useful set: **final_wealth**, **total_contributions**, **CAGR**, **vol** (approximate), and **max_drawdown**.

---

## Main Components

### 1) Capital accumulation (single asset)
**Function:** `simulate_capital(contributions, returns, start_value=0.0, ...)`

Monthly dynamics:
\[
W_{t+1} = (W_t + a_t)\,(1 + r_t)
\]
- `contributions`: vector \(a_t\) (negatives allowed for withdrawals).
- `returns`: either a scalar (constant rate) or a path \((r_t)\) (arithmetic returns).
- Optional safeguard: `clip_negative_wealth=True` to prevent negative wealth paths.

**When to use:** when modeling the portfolio as an “aggregate asset” (e.g., a balanced fund or benchmark).

---

### 2) Multi-asset portfolios
**Function:** `simulate_portfolio(contributions, asset_returns, weights, rebalance=True, ...)`

- `asset_returns`: \(T \times N\) matrix of arithmetic returns.
- `weights`:
  - Vector \(N\): constant target weights.  
  - Matrix \(T \times N\): **schedule** of weights (implicit monthly rebalancing).
- `rebalance=True` with constant weights ⇒ realign monthly to targets.  
- `rebalance=False` ⇒ buy-and-hold style: weights drift with returns; contributions added **pro-rata** to avoid cash bias.

**Conceptual focus:** separate **allocation decisions** (weights) from **return paths** (scenarios), and be explicit about **when** rebalancing happens.

---

### 3) Return scenario generators
- `fixed_rate_path(months, r)`: constant return path (useful for “base/optimistic/pessimistic” cases).
- `lognormal_iid(months, mu, sigma, seed, drift_in_logs=False)`: arithmetic returns generated from a lognormal process on \(1+r\).  
  - By default, `mu` and `sigma` approximate monthly **arithmetic** mean and volatility.

**Intention:** provide reproducible *building blocks* for `simulation.py` without locking the engine to a single distribution.

---

### 4) Performance metrics
**Class/Function:** `PortfolioMetrics` and `compute_metrics(wealth, contributions=None, periods_per_year=12)`

Returns:
- `final_wealth`: \(W_T\)  
- `total_contributions`: \(\sum_t a_t\)  
- `cagr`: robust CAGR (avoids division by zero if \(W_0=0\))  
- `vol`: standard deviation of simple returns on wealth (approximate)  
- `max_drawdown`: minimum of drawdown series \((W_t - \max_{u\le t} W_u)/\max_{u\le t} W_u\)

**Interpretation:**  
- `cagr` contextualizes annualized growth.  
- `vol` and `max_drawdown` approximate *risk/downsides* observed in simulated wealth, not in raw asset returns.

---

## Integration in the FinOpt workflow

1. **`income.py`** produces monthly contributions \(a_t\) (e.g., \(\alpha\) of fixed income + \(\beta\) of variable).  
2. **`investment.py`** combines \(a_t\) with **returns** (constant, scenario-based, or Monte Carlo) to produce the **wealth path**.  
3. **`simulation.py`** orchestrates both (builds return paths, runs scenarios/Monte Carlo, computes metrics, returns results for reporting/optimization).

---

## Modeling choices and trade-offs

- **Arithmetic returns**  
  Easier to interpret month by month. If higher accuracy is needed, use log returns in scenarios and convert to arithmetic for simulation.

- **Explicit monthly rebalancing**  
  Defines clear operational behavior (transaction costs out of scope in MVP).  

- **Wealth volatility**  
  Approximation of variability in investor outcomes (mixing contributions and returns). For pure asset risk, use volatility/CVaR at `asset_returns` level.

- **No shorting**  
  Weights must be ≥ 0 and sum to 1; suitable for retail-style portfolios and keeps interpretability.

---

## Recommended usage patterns

### A) Single asset, constant rate
```python
import numpy as np
from finopt.src.investment import simulate_capital, fixed_rate_path, compute_metrics

T = 24
a = np.full(T, 700_000.0)         # monthly contribution
r = fixed_rate_path(T, 0.005)     # 0.5% monthly
wealth = simulate_capital(a, r)
metrics = compute_metrics(wealth, a)
```
### B) 60/40 portfolio with monthly rebalancing
```python
import numpy as np
from finopt.src.investment import simulate_portfolio, compute_metrics

T, N = 36, 2
rng = np.random.default_rng(42)
R = np.column_stack([
    rng.normal(0.006, 0.02, size=T),  # asset 1
    rng.normal(0.003, 0.01, size=T),  # asset 2
])
a = np.full(T, 700_000.0)
w = np.array([0.6, 0.4])
wealth = simulate_portfolio(a, R, w, rebalance=True)
metrics = compute_metrics(wealth, a)
```
### C) Dynamic weights (glidepath)
```python
# weights change from 80/20 to 50/50 over horizon
w_schedule = np.column_stack([
    np.linspace(0.8, 0.5, T),
    np.linspace(0.2, 0.5, T),
])
wealth = simulate_portfolio(a, R, w_schedule, rebalance=True)
```