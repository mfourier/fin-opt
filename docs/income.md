# `income` – Philosophy & Role in FinOpt

> **Purpose:** Model **where the money comes from** (and how it evolves) so the rest of FinOpt can decide **how to allocate and invest it**.  
> In FinOpt's pipeline, `income.py` is the **entry point of cash flows**: it turns assumptions about salary and variable earnings into a clean, reproducible **monthly series** that downstream modules consume for contributions and simulations.

---

## Why a dedicated income module?

Financial planning hinges on **cash availability per period**. Any optimizer or simulator that ignores timing or volatility of income will produce plans that are hard to execute. `income.py` separates **cash generation** (what you earn) from **capital dynamics** (what you invest), giving you:

- **Clarity:** Incomes are modeled explicitly (fixed vs. variable).  
- **Composability:** Outputs plug directly into `simulation.py` and `investment.py`.  
- **Reproducibility:** Deterministic by default; any randomness is controlled by an explicit seed.  
- **Extensibility:** Easy to add expenses, taxes, or more streams without touching portfolio code.

---

## Design principles

1. **Deterministic baseline, controlled randomness**
   - Everything is deterministic unless you explicitly add noise to `VariableIncome` via `sigma` and `seed`.
2. **Calendar-first outputs**
   - Returns **pandas Series/DataFrames** indexed by the **first day of each month** (friendly for reporting and plotting).
3. **Minimal but expressive**
   - Two stream types cover common cases:
     - `FixedIncome`: predictable salary with optional **annual growth** and **scheduled raises**.
     - `VariableIncome`: irregular stream with **seasonality** and **Gaussian noise**.
4. **Single responsibility**
   - `income.py` does **not** simulate returns or portfolios. It only **projects income** and derives **contribution series** from it.

---

## The three core surfaces

### 1) `FixedIncome`

A deterministic monthly base with **compounded annual growth** and optional **salary raises**:

**Parameters:**
- `base`: monthly base income at t=0
- `annual_growth`: nominal annual rate (converted internally to monthly compounding)
- `salary_raises`: `Optional[Dict[date, float]]` – absolute raise amounts at specific dates
- `name`: identifier for labeling outputs

**Method signature:**
```python
def project(
    self, 
    months: int, 
    *, 
    start: Optional[date] = None,
    output: Literal["array", "series"] = "array"
) -> np.ndarray | pd.Series
```
- `start`: required when `salary_raises` is specified; used for calendar alignment
- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` with calendar index

**Key behaviors:**
- Monthly projection uses the equivalent monthly rate: `m = (1 + annual_growth)^(1/12) - 1`
- Salary raises are applied permanently from the month containing the specified date
- Growth compounds on the updated base after each raise
- Requires `start` date in `.project()` when `salary_raises` is specified
- Guarantees **non-negativity** and well-formed arrays

**Interpretation:** models a salary with contractual raises and inflation adjustments; simple and transparent.

---

### 2) `VariableIncome`

A variable stream with optional **seasonality**, **noise**, **floor/cap**, and **annual growth**:

**Parameters:**
- `base`: baseline monthly income before transformations
- `seasonality`: 12 multiplicative factors (Jan–Dec), must have length 12
- `sigma`: standard deviation of noise as **a fraction of the month mean**
- `floor` / `cap`: guardrails applied after noise (e.g., minimum expected side income)
- `annual_growth`: nominal annual rate applied before seasonality
- `seed`: RNG seed for reproducible noise (can be overridden in `.project()`)
- `name`: identifier for labeling outputs

**Method signature:**
```python
def project(
    self, 
    months: int, 
    *, 
    start: Optional[date | int] = None, 
    seed: Optional[int] = None,
    output: Literal["array", "series"] = "array"
) -> np.ndarray | pd.Series
```
- `start`: can be `date` or `int` (month 1-12); determines seasonality rotation
- `seed` in `project()` overrides instance-level seed if provided
- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` with calendar index

**Interpretation:** models tutoring, bonuses, or freelancing income whose level changes across the year and fluctuates each month.

---

### 3) `IncomeModel`

A façade that **combines streams** and produces projections, contributions, metrics, and visualizations.

#### Core projection methods

**`project(months, start=None, output="series", seed=None)`**
- Returns total income (optionally as DataFrame with [fixed, variable, total] columns)
- Aligned to calendar via `start` date
- `output`: `"series"` returns total as Series, `"dataframe"` returns breakdown with components
- `seed`: controls reproducibility of variable income (overrides instance seed)
- Example:
  ```python
  # Total income as Series (default)
  total = income.project(months=24, start=date(2025, 9, 1))
  
  # Breakdown as DataFrame
  df = income.project(months=24, start=date(2025, 9, 1), output="dataframe")
  ```

**`contributions(months, start=None, seed=None, output="series")`**
- Computes monthly contributions using **12-month fractional arrays** that rotate based on `start`:
  $$
  \text{contrib}_t = \alpha^{\text{fixed}}_{(t+\text{offset})\bmod 12} \cdot y^{\text{fixed}}_t + \alpha^{\text{variable}}_{(t+\text{offset})\bmod 12} \cdot y^{\text{variable}}_t
  $$
  where `offset = normalize_start_month(start)`.

- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` with calendar index
- `seed`: controls reproducibility of variable income (overrides instance seed)

- **Default fractions** (if `monthly_contribution` is `None`):
  - Fixed: 30% each month
  - Variable: 100% each month

- **Custom fractions** via attribute:
  ```python
  income.monthly_contribution = {
      "fixed": [0.35]*12,      # Jan-Dec fractions
      "variable": [1.0]*12
  }
  contrib = income.contributions(months=24, start=date(2025, 9, 1))
  ```

- Contributions are floored at zero (no negative values)
- The 12-month arrays repeat cyclically for horizons > 12 months

#### Statistical methods

**`income_metrics(months, start=None, variable_threshold=None)`**
- Returns `IncomeMetrics` dataclass with:
  ```python
  @dataclass
  class IncomeMetrics:
      months: int
      total_fixed, total_variable, total_income: float
      mean_fixed, mean_variable, mean_total: float
      std_variable, coefvar_variable: float
      fixed_share, variable_share: float
      min_variable, max_variable: float
      pct_variable_below_threshold: float  # NaN if threshold not provided
  ```

**`summary(months, start=None, variable_threshold=None, round_digits=2)`**
- Convenience wrapper that returns `income_metrics()` as a rounded pandas Series

#### Visualization methods

**`plot_income(months, start=None, ...)`**
- Plots fixed, variable, and total income streams
- **Dual-axis support**: automatic when scales differ by `dual_axis_ratio` (default 3.0)
- **Monte Carlo trajectories**: shows individual stochastic paths when `show_trajectories=True` and `sigma > 0`
- **Confidence bands**: statistical intervals when `show_confidence_band=True` and `sigma > 0`
- Key parameters:
  - `ax`, `figsize`, `title`, `legend`, `grid`
  - `ylabel_left`, `ylabel_right`: axis labels
  - `dual_axis`: `"auto"` | `True` | `False`
  - `show_trajectories=True`, `trajectory_alpha=0.08`
  - `show_confidence_band=False`, `confidence=0.9`, `n_simulations=500`
  - `colors`: `{"fixed": "blue", "variable": "orange", "total": "black"}`
  - `save_path`, `return_fig_ax`
- Displays cumulative totals annotation

**`plot_contributions(months, start=None, ...)`**
- Plots total monthly contributions with optional Monte Carlo trajectories and confidence bands
- Single y-axis (no dual-axis)
- Key parameters:
  - `ax`, `figsize`, `title`, `legend`, `grid`, `ylabel`
  - `show_trajectories=True`, `trajectory_alpha=0.08`
  - `show_confidence_band=False`, `confidence=0.9`, `n_simulations=500`
  - `colors`: `{"total": "blue", "ci": "orange"}`
  - `save_path`, `return_fig_ax`
- Displays total contributions annotation

**`plot(mode="income"|"contributions", ...)`**
- Unified wrapper that dispatches to `plot_income()` or `plot_contributions()`
- All parameters are forwarded to the appropriate method

#### Serialization

**`to_dict()` / `from_dict(payload)`**
- Serialize/deserialize model configuration for persistence
- Handles `salary_raises` date conversion (ISO format strings)

---

## How `income.py` powers the rest of FinOpt

- **`simulation.py`**  
  Uses `IncomeModel.contributions(...)` to generate the **contribution series** aligned to the simulation calendar, then combines it with deterministic or Monte Carlo **returns** to simulate wealth.

- **`investment.py`**  
  Receives the contributions from `income.py` and applies **capital accumulation**:
  $$
  W_{t+1}=(W_t+A_t)(1+R_t).
  $$
  Metrics (CAGR, drawdown, volatility) are computed downstream on the resulting wealth path.

- **`utils.py`**  
  Provides shared helpers used by `income.py` (e.g., **rate conversions** annual↔monthly, **month index** construction, **validation**).

---

## Recommended usage patterns

### A) Baseline projection (deterministic)
```python
from datetime import date
from finopt.src.income import FixedIncome, VariableIncome, IncomeModel

income = IncomeModel(
    fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
    variable=VariableIncome(base=200_000.0, sigma=0.00)  # no noise
)

# 24-month calendar-aligned totals (and components if needed)
df = income.project(months=24, start=date(2025, 9, 1), output="dataframe")
```

### B) From income to contributions

**Option 1: Use defaults (30% fixed, 100% variable)**
```python
contrib = income.contributions(months=24, start=date(2025, 9, 1))
```

**Option 2: Custom monthly fractions**
```python
income.monthly_contribution = {
    "fixed": [0.35]*12,      # 35% each month (Jan-Dec)
    "variable": [1.0]*12     # 100% each month
}
contrib = income.contributions(months=24, start=date(2025, 9, 1))
```

**Option 3: Seasonal contribution patterns**
```python
# Higher contributions in bonus months
income.monthly_contribution = {
    "fixed": [0.30]*12,
    "variable": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]  # save 50% of Dec bonus
}
```

### C) Fixed income with salary raises
```python
fixed = FixedIncome(
    base=1_400_000.0,
    annual_growth=0.03,
    salary_raises={
        date(2025, 7, 1): 200_000,   # raise in July
        date(2026, 1, 1): 150_000    # raise in January
    }
)
# Requires start date when projecting
path = fixed.project(months=24, start=date(2025, 1, 1))  # returns array by default
series = fixed.project(months=24, start=date(2025, 1, 1), output="series")  # with calendar index
```

### D) Variable income with seasonality + noise
```python
seasonality = [1.00, 0.95, 1.05, 1.10, 1.15, 1.10, 
               1.00, 0.90, 0.95, 1.05, 1.10, 1.20]

income_var = VariableIncome(
    base=200_000.0,
    seasonality=seasonality,
    sigma=0.15,
    floor=50_000.0,
    cap=400_000.0,
    annual_growth=0.02,
    seed=123
)

# Project as array (default)
path = income_var.project(months=12)

# Project as Series with calendar alignment
series = income_var.project(months=12, start=date(2025, 1, 1), output="series")
```

### E) Statistical summary
```python
# Detailed metrics as dataclass
metrics = income.income_metrics(
    months=24, 
    start=date(2025, 1, 1),
    variable_threshold=150_000.0
)

# Compact summary as Series
summary = income.summary(months=24, start=date(2025, 1, 1), round_digits=2)
print(summary)
```

### F) Visualization with Monte Carlo trajectories
```python
# Income streams with stochastic trajectories (default if sigma > 0)
income.plot_income(
    months=24, 
    start=date(2025, 1, 1),
    show_trajectories=True,
    n_simulations=500,
    dual_axis="auto",
    save_path="income_projection.png"
)

# Legacy mode: confidence bands only
income.plot_income(
    months=24,
    start=date(2025, 1, 1),
    show_trajectories=False,
    show_confidence_band=True,
    confidence=0.95
)

# Hybrid mode: trajectories + bands
income.plot_income(
    months=24,
    start=date(2025, 1, 1),
    show_trajectories=True,
    show_confidence_band=True,
    n_simulations=150,
    trajectory_alpha=0.03
)

# Contributions with trajectories
income.plot_contributions(
    months=24, 
    start=date(2025, 1, 1),
    show_trajectories=True,
    title="Monthly Investment Contributions"
)

# Using unified wrapper
income.plot(
    mode="income",  # or "contributions"
    months=24,
    start=date(2025, 1, 1)
)
```

---

## Key design decisions

### 1. Monthly fractions are 12-element arrays, not scalars
This allows **seasonal contribution strategies**: save more during high-income months, less during lean months. The arrays rotate based on `start` and repeat cyclically.

### 2. Salary raises are date-based, not month-offset-based
You specify `{date(2025, 7, 1): 200_000}`, not `{6: 200_000}`. The conversion to month offsets happens internally relative to the projection `start` date, making the model calendar-aware.

### 3. Flexible output formats via `output` parameter
All projection methods (`FixedIncome.project()`, `VariableIncome.project()`, `IncomeModel.project()`, `IncomeModel.contributions()`) support an `output` parameter to control return type:
- `"array"`: returns `np.ndarray` (default for streams, no calendar overhead)
- `"series"`: returns `pd.Series` with calendar index (default for `IncomeModel`, user-friendly)
- `"dataframe"`: returns `pd.DataFrame` with component breakdown (only `IncomeModel.project()`)

### 4. Reproducibility via explicit `seed` parameter
Variable income randomness is controlled at two levels:
- Instance-level: `VariableIncome(seed=42)` sets default seed
- Method-level: `project(..., seed=123)` overrides for specific realizations
This enables reproducible stochastic projections without mutating the model state.

### 5. Monte Carlo trajectories as primary visualization
When `sigma > 0`, plotting methods default to showing individual trajectories (`show_trajectories=True`) instead of confidence bands, providing more intuitive visualization of stochastic dynamics. Confidence bands remain available via `show_confidence_band=True`.

### 6. Dual-axis activation is automatic by default
When fixed and variable incomes differ by a factor > `dual_axis_ratio`, the plot automatically uses separate y-axes to avoid visual compression. Override with `dual_axis=True|False`.

---

## Implementation notes

- **Rate conversion**: `annual_to_monthly(g) = (1+g)^(1/12) - 1` ensures geometric compounding
- **Calendar alignment**: `month_index(start, months)` generates `pd.DatetimeIndex` of first-of-month dates
- **Seasonality rotation**: `normalize_start_month(start)` returns 0-indexed month offset (0=Jan, 11=Dec)
- **Non-negativity**: all income and contribution values are floored at zero after transformations
- **Seed propagation**: `seed=None` in methods uses instance seed; both `None` generates non-deterministic noise