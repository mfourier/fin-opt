# `utils` — Shared Utilities for FinOpt

> **Core idea:** Provide common helper functions used across all FinOpt modules.
> `utils.py` centralizes validation, rate conversions, array handling, financial metrics, and formatting utilities to avoid code duplication and ensure consistency.

---

## Why a dedicated utils module?

- **DRY principle:** Common operations defined once, used everywhere
- **Consistency:** Standardized rate conversions and formatting across modules
- **Robustness:** Edge case handling (division by zero, empty arrays) in one place
- **Testability:** Isolated functions that are easy to unit test

---

## Contents Overview

| Category | Functions |
|----------|-----------|
| **Validation** | `check_non_negative` |
| **Rate conversions** | `annual_to_monthly`, `monthly_to_annual` |
| **Array/Series helpers** | `ensure_1d`, `to_series`, `month_index`, `align_index_like`, `normalize_start_month` |
| **Finance helpers** | `drawdown`, `compute_cagr` |
| **Scenario helpers** | `set_random_seed`, `rescale_returns`, `bootstrap_returns` |
| **Reporting** | `summary_metrics` |
| **Matplotlib formatters** | `millions_formatter`, `format_currency` |
| **Return generators** | `fixed_rate_path`, `lognormal_iid` |
| **Metrics** | `PortfolioMetrics`, `compute_metrics` |

---

## Validation Helpers

### `check_non_negative(name, value)`

Raises `ValueError` if value is negative.

```python
from finopt.utils import check_non_negative

check_non_negative("base", 1_000_000)  # OK
check_non_negative("sigma", -0.1)      # Raises ValueError
```

**Used by:** `FixedIncome`, `VariableIncome`, `Account`

---

## Rate Conversions

### `annual_to_monthly(r_annual) -> float`

Convert nominal annual rate to equivalent compounded monthly rate.

$$
r_m = (1 + r_a)^{1/12} - 1
$$

```python
from finopt.utils import annual_to_monthly

monthly = annual_to_monthly(0.12)  # 0.12 annual → 0.00949 monthly
```

**Note:** Accepts negative values (for modeling deflation or decay).

---

### `monthly_to_annual(r_monthly) -> float`

Convert nominal monthly rate to equivalent compounded annual rate.

$$
r_a = (1 + r_m)^{12} - 1
$$

```python
from finopt.utils import monthly_to_annual

annual = monthly_to_annual(0.01)  # 0.01 monthly → 0.1268 annual (12.68%)
```

---

## Array / Series Helpers

### `ensure_1d(a, name="array") -> np.ndarray`

Convert input to a 1-D float NumPy array with validation.

```python
from finopt.utils import ensure_1d

arr = ensure_1d([1.0, 2.0, 3.0], name="returns")
# Returns: array([1., 2., 3.])

ensure_1d([[1, 2], [3, 4]])  # Raises ValueError: must be 1-D
ensure_1d([1.0, np.nan])     # Raises ValueError: must contain only finite values
```

**Checks:**
- Shape is 1-D
- All values are finite (no NaN or Inf)

---

### `to_series(a, index, name="value") -> pd.Series`

Create pandas Series from array-like with optional index.

```python
from finopt.utils import to_series

series = to_series([100, 200, 300], index=pd.date_range("2025-01-01", periods=3, freq="MS"))
```

---

### `month_index(start, months) -> pd.DatetimeIndex`

Construct a first-of-month DatetimeIndex for given number of periods.

```python
from datetime import date
from finopt.utils import month_index

idx = month_index(start=date(2025, 1, 1), months=6)
# DatetimeIndex(['2025-01-01', '2025-02-01', '2025-03-01',
#                '2025-04-01', '2025-05-01', '2025-06-01'], freq='MS')

idx = month_index(start=None, months=6)  # Uses current month as start
```

**Used by:** `IncomeModel.project()`, `IncomeModel.contributions()`, plotting methods

---

### `align_index_like(months, like) -> pd.DatetimeIndex`

Infer DatetimeIndex from an existing Series/DataFrame if possible.

```python
from finopt.utils import align_index_like

# Reuse index from existing data
idx = align_index_like(months=12, like=existing_series)

# Falls back to month_index(None, months) if like is None or incompatible
```

---

### `normalize_start_month(start) -> int`

Map start date or month integer to 0-indexed offset (0=Jan, 11=Dec).

```python
from datetime import date
from finopt.utils import normalize_start_month

normalize_start_month(date(2025, 3, 15))  # → 2 (March)
normalize_start_month(9)                   # → 8 (September)
normalize_start_month(None)                # → 0 (January default)
```

**Used by:** Seasonality rotation in `VariableIncome`, contribution fraction rotation

---

## Finance Helpers

### `drawdown(series) -> pd.Series`

Compute drawdown series: $(W - \text{cummax}(W)) / \text{cummax}(W)$.

```python
from finopt.utils import drawdown

wealth = pd.Series([100, 120, 110, 130, 115])
dd = drawdown(wealth)
# Returns: [0.0, 0.0, -0.0833, 0.0, -0.1154]
```

**Features:**
- Returns zeros for non-positive running maxima (avoids division by zero)
- Preserves Series name

---

### `compute_cagr(wealth, periods_per_year=12) -> float`

Compute Compound Annual Growth Rate from a wealth series.

$$
\text{CAGR} = \left(\frac{W_T}{W_0}\right)^{1/\text{years}} - 1
$$

```python
from finopt.utils import compute_cagr

wealth = pd.Series([1_000_000, 1_050_000, 1_100_000, ..., 1_500_000])  # 24 months
cagr = compute_cagr(wealth, periods_per_year=12)  # ~22.5% annual
```

**Features:**
- Uses first strictly-positive observation as starting base
- Handles $W_0 = 0$ gracefully (contribution-driven processes)
- Returns 0.0 for empty or invalid series

---

## Scenario Helpers

### `set_random_seed(seed)`

Set NumPy and Python random seeds for reproducibility.

```python
from finopt.utils import set_random_seed

set_random_seed(42)  # Sets both np.random and random module
set_random_seed(None)  # No-op (non-deterministic)
```

---

### `rescale_returns(path, target_mean, target_vol) -> np.ndarray`

Rescale an arithmetic-returns path to match target mean and volatility.

```python
from finopt.utils import rescale_returns

historical = np.array([0.01, -0.02, 0.03, 0.015, -0.01])
rescaled = rescale_returns(historical, target_mean=0.008, target_vol=0.04)
```

**Use case:** Normalize historical returns to match model assumptions.

---

### `bootstrap_returns(history, months, seed=None) -> np.ndarray`

Simple IID bootstrap of arithmetic returns from historical sample.

```python
from finopt.utils import bootstrap_returns

historical = np.array([0.01, -0.02, 0.03, 0.015, -0.01, 0.02])
bootstrapped = bootstrap_returns(historical, months=24, seed=42)
# Returns: 24 randomly sampled returns (with replacement)
```

**Use case:** Generate synthetic return scenarios from historical data.

---

## Reporting Helpers

### `summary_metrics(results) -> pd.DataFrame`

Build a metrics table from a dict of ScenarioResult-like objects.

```python
from finopt.utils import summary_metrics

results = {
    "Conservative": scenario_a,
    "Aggressive": scenario_b,
}
df = summary_metrics(results)
#                    final_wealth  total_contributions    cagr     vol  max_drawdown
# Conservative       25_000_000            15_000_000  0.085   0.045        -0.12
# Aggressive         35_000_000            15_000_000  0.125   0.095        -0.25
```

**Duck-typing:** Each value must have `.metrics` with attributes: `final_wealth`, `total_contributions`, `cagr`, `vol`, `max_drawdown`.

---

## Matplotlib Formatters

### `millions_formatter(x, pos) -> str`

Format axis values as millions for matplotlib FuncFormatter.

```python
from matplotlib.ticker import FuncFormatter
from finopt.utils import millions_formatter

ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
# 25_000_000 → "25M"
# 12_500_000 → "12.5M"
```

---

### `format_currency(value, decimals=1, symbol='$', unit='M') -> str`

Format currency values for text annotations and labels.

```python
from finopt.utils import format_currency

format_currency(25_000_000)              # → '$25.0M'
format_currency(25_000_000, decimals=0)  # → '$25M'
format_currency(5_500_000, decimals=2)   # → '$5.50M'
```

**Use case:** Text annotations, legends, titles where FuncFormatter cannot be applied.

---

## Return-Path Generators

### `fixed_rate_path(months, r_monthly) -> np.ndarray`

Return a constant arithmetic monthly return path.

```python
from finopt.utils import fixed_rate_path

path = fixed_rate_path(months=12, r_monthly=0.005)
# Returns: array([0.005, 0.005, ..., 0.005])  # 12 elements
```

**Use case:** Deterministic scenarios for testing or benchmarking.

---

### `lognormal_iid(months, mu, sigma, seed=None) -> np.ndarray`

IID arithmetic returns derived from lognormal gross returns.

$$
G_t \sim \text{LogNormal}(\mu, \sigma), \quad r_t = G_t - 1
$$

```python
from finopt.utils import lognormal_iid

returns = lognormal_iid(months=24, mu=0.005, sigma=0.04, seed=42)
# Returns: 24 arithmetic returns, all > -1 (guaranteed)
```

**Guarantee:** $r_t > -1$ always (no impossible bankruptcies).

**Note:** `mu` and `sigma` are parameters of the normal distribution in log-gross space, not arithmetic mean/volatility.

---

## Metrics

### `PortfolioMetrics` (frozen dataclass)

Container for portfolio performance metrics.

```python
@dataclass(frozen=True)
class PortfolioMetrics:
    final_wealth: float
    total_contributions: float
    cagr: float
    vol: float
    max_drawdown: float
```

---

### `compute_metrics(wealth, contributions=None, periods_per_year=12) -> PortfolioMetrics`

Compute key metrics for a simulated wealth path.

```python
from finopt.utils import compute_metrics

metrics = compute_metrics(wealth_series, contributions=contrib_series)
print(f"Final wealth: {metrics.final_wealth:,.0f}")
print(f"CAGR: {metrics.cagr:.2%}")
print(f"Max drawdown: {metrics.max_drawdown:.2%}")
```

**Computed metrics:**
| Metric | Description |
|--------|-------------|
| `final_wealth` | $W_T$ (terminal wealth) |
| `total_contributions` | $\sum A_t$ (if provided) |
| `cagr` | Compound annual growth rate |
| `vol` | Standard deviation of approximate monthly returns |
| `max_drawdown` | Minimum of drawdown series |

**Note:** When wealth includes contributions, `vol` based on $\Delta W / W_{t-1}$ is approximate. For pure risk analysis, use exogenous returns instead.

---

## Usage Examples

### A) Rate conversion for account setup

```python
from finopt.utils import annual_to_monthly, monthly_to_annual

# User specifies annual, internally we use monthly
annual_return = 0.08
monthly_return = annual_to_monthly(annual_return)  # 0.00643

# Verify round-trip
assert abs(monthly_to_annual(monthly_return) - annual_return) < 1e-10
```

### B) Calendar-aware index construction

```python
from datetime import date
from finopt.utils import month_index

# Generate 24-month projection index starting September 2025
idx = month_index(start=date(2025, 9, 1), months=24)
# ['2025-09-01', '2025-10-01', ..., '2027-08-01']
```

### C) Portfolio metrics after simulation

```python
from finopt.utils import compute_metrics, format_currency

result = model.simulate(T=36, n_sims=1)
metrics = compute_metrics(
    wealth=result.wealth.iloc[:, 0],  # First simulation, total wealth
    contributions=result.contributions
)

print(f"Final: {format_currency(metrics.final_wealth)}")
print(f"CAGR: {metrics.cagr:.1%}")
print(f"Max DD: {metrics.max_drawdown:.1%}")
```

### D) Consistent plot formatting

```python
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from finopt.utils import millions_formatter, format_currency

fig, ax = plt.subplots()
ax.plot(dates, wealth)
ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
ax.set_title(f"Wealth projection (Final: {format_currency(wealth[-1])})")
```

---

## Module Usage in FinOpt

| Module | Utils Used |
|--------|------------|
| `income.py` | `check_non_negative`, `annual_to_monthly`, `month_index`, `normalize_start_month`, `millions_formatter`, `format_currency` |
| `portfolio.py` | `check_non_negative`, `annual_to_monthly`, `monthly_to_annual` |
| `returns.py` | `annual_to_monthly`, `monthly_to_annual` |
| `model.py` | `month_index`, `millions_formatter`, `format_currency`, `compute_metrics` |
| `goals.py` | `month_index` |
| `withdrawal.py` | `month_index` |

---

## API Summary

| Function | Purpose |
|----------|---------|
| `check_non_negative(name, value)` | Validate non-negative values |
| `annual_to_monthly(r)` | Annual → monthly rate conversion |
| `monthly_to_annual(r)` | Monthly → annual rate conversion |
| `ensure_1d(a, name)` | Convert to validated 1-D array |
| `to_series(a, index, name)` | Create pandas Series |
| `month_index(start, months)` | Build first-of-month DatetimeIndex |
| `align_index_like(months, like)` | Infer index from existing data |
| `normalize_start_month(start)` | Map date/int to 0-11 offset |
| `drawdown(series)` | Compute drawdown series |
| `compute_cagr(wealth)` | Compute CAGR from wealth path |
| `set_random_seed(seed)` | Set RNG seeds for reproducibility |
| `rescale_returns(path, mean, vol)` | Normalize returns to target stats |
| `bootstrap_returns(history, months)` | IID bootstrap from historical data |
| `summary_metrics(results)` | Build comparison DataFrame |
| `millions_formatter(x, pos)` | Matplotlib axis formatter |
| `format_currency(value)` | Text currency formatting |
| `fixed_rate_path(months, r)` | Constant return path |
| `lognormal_iid(months, mu, sigma)` | Lognormal IID returns |
| `compute_metrics(wealth)` | Compute PortfolioMetrics |
