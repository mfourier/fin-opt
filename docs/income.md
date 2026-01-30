# `income` — Cash Flow Modeling for FinOpt

> **Core idea:** Model **where the money comes from** (and how it evolves) so the rest of FinOpt can decide **how to allocate and invest it**.
> In FinOpt's pipeline, `income.py` is the **entry point of cash flows**: it turns assumptions about salary and variable earnings into a clean, reproducible **monthly series** that downstream modules consume for contributions and simulations.

---

## Why a dedicated income module?

Financial planning hinges on **cash availability per period**. Any optimizer or simulator that ignores timing or volatility of income will produce plans that are hard to execute. `income.py` separates **cash generation** (what you earn) from **capital dynamics** (what you invest), giving you:

- **Clarity:** Incomes are modeled explicitly (fixed vs. variable).
- **Composability:** Outputs plug directly into `simulation.py` and `investment.py`.
- **Reproducibility:** Deterministic by default; any randomness is controlled by an explicit seed.
- **Extensibility:** Easy to add expenses, taxes, or more streams without touching portfolio code.
- **Performance:** Vectorized Monte Carlo generation via `n_sims` parameter.

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
   - Either component can be `None` in `IncomeModel` (at least one required).

4. **Single responsibility**
   - `income.py` does **not** simulate returns or portfolios. It only **projects income** and derives **contribution series** from it.

5. **Vectorized Monte Carlo**
   - All projection methods support `n_sims` parameter for efficient batch generation.
   - ~100x speedup vs. sequential calls for typical Monte Carlo workloads.

---

## The three core surfaces

### 1) `FixedIncome`

A deterministic monthly base with **compounded annual growth** and optional **salary raises**:

**Parameters:**
- `base`: monthly base income at t=0 (must be non-negative)
- `annual_growth`: nominal annual rate (converted internally to monthly compounding)
- `salary_raises`: `Optional[Dict[date, float]]` — absolute raise amounts at specific dates
- `name`: identifier for labeling outputs (default: `"fixed"`)

**Method signature:**
```python
def project(
    self,
    months: int,
    *,
    start: Optional[date] = None,
    output: Literal["array", "series"] = "array",
    n_sims: int = 1,
) -> np.ndarray | pd.Series
```

**Parameters:**
- `months`: Number of months to project (≥ 0)
- `start`: Required when `salary_raises` is specified; used for calendar alignment
- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` with calendar index
- `n_sims`: Number of simulations (deterministic replication for API consistency)

**Returns:**
- If `n_sims=1` and `output="array"`: `np.ndarray` of shape `(months,)`
- If `n_sims>1` and `output="array"`: `np.ndarray` of shape `(n_sims, months)` (all rows identical)
- If `n_sims=1` and `output="series"`: `pd.Series` indexed by first-of-month dates
- If `n_sims>1` and `output="series"`: raises `ValueError`

**Key behaviors:**
- Monthly projection uses the equivalent monthly rate: `m = (1 + annual_growth)^(1/12) - 1`
- Salary raises are applied permanently from the month containing the specified date
- Growth compounds on the updated base after each raise
- Guarantees **non-negativity** and well-formed arrays
- When `n_sims > 1`, the deterministic projection is replicated across all simulations

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
- `name`: identifier for labeling outputs (default: `"variable"`)

**Method signature:**
```python
def project(
    self,
    months: int,
    *,
    start: Optional[date | int] = None,
    seed: Optional[int] = None,
    output: Literal["array", "series"] = "array",
    n_sims: int = 1,
) -> np.ndarray | pd.Series
```

**Parameters:**
- `months`: Number of months to project (≥ 0)
- `start`: Can be `date` or `int` (month 1-12); determines seasonality rotation
- `seed`: Overrides instance-level seed if provided; controls reproducibility of all simulations
- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` with calendar index
- `n_sims`: Number of independent simulations to generate

**Returns:**
- If `n_sims=1` and `output="array"`: `np.ndarray` of shape `(months,)`
- If `n_sims>1` and `output="array"`: `np.ndarray` of shape `(n_sims, months)`
- If `n_sims=1` and `output="series"`: `pd.Series` indexed by first-of-month dates
- If `n_sims>1` and `output="series"`: raises `ValueError`

**Performance:**
```python
# Vectorized: ~100x faster than sequential calls
sims = vi.project(months=240, n_sims=500)  # shape: (500, 240)

# vs. sequential (avoid this)
sims = np.array([vi.project(240) for _ in range(500)])  # slow!
```

**Interpretation:** models tutoring, bonuses, or freelancing income whose level changes across the year and fluctuates each month.

---

### 3) `IncomeModel`

A façade that **combines streams** and produces projections, contributions, metrics, and visualizations.

**Parameters:**
- `fixed`: `Optional[FixedIncome]` — deterministic income stream (can be `None`)
- `variable`: `Optional[VariableIncome]` — stochastic income stream (can be `None`)
- `name_fixed`: label for fixed component in outputs (default: `"fixed"`)
- `name_variable`: label for variable component in outputs (default: `"variable"`)
- `monthly_contribution`: `Optional[MonthlyContributionDict]` — 12-month fractional arrays per stream

**Constraint:** At least one of `fixed` or `variable` must be provided.

#### Core projection methods

**`project(months, start=None, output="series", seed=None, n_sims=1)`**

Returns total income (optionally as DataFrame with component breakdown or dict of arrays).

**Parameters:**
- `months`: Number of months to project
- `start`: Start date for calendar alignment
- `output`: Output format:
  - `"series"`: total as `pd.Series` (default, n_sims=1 only)
  - `"dataframe"`: breakdown `pd.DataFrame` with [fixed, variable, total] (n_sims=1 only)
  - `"array"`: `dict` with `{name_fixed: array, name_variable: array, "total": array}`
- `seed`: Controls reproducibility of variable income
- `n_sims`: Number of independent simulations (only `output="array"` supports n_sims > 1)

**Returns:**
- `n_sims=1`, `output="series"`: `pd.Series` of total income
- `n_sims=1`, `output="dataframe"`: `pd.DataFrame` with component columns
- `n_sims=1`, `output="array"`: `dict` with shape `(months,)` arrays
- `n_sims>1`, `output="array"`: `dict` with shape `(n_sims, months)` arrays

**Example:**
```python
# Total income as Series (default)
total = income.project(months=24, start=date(2025, 9, 1))

# Breakdown as DataFrame
df = income.project(months=24, start=date(2025, 9, 1), output="dataframe")

# Multiple simulations (vectorized)
result = income.project(months=24, n_sims=500, output="array")
# result["total"].shape → (500, 24)
```

---

**`contributions(months, start=None, seed=None, output="series", n_sims=1)`**

Computes monthly contributions using **12-month fractional arrays** that rotate based on `start`:

$$
\text{contrib}_t = \alpha^{\text{fixed}}_{(t+\text{offset})\bmod 12} \cdot y^{\text{fixed}}_t + \alpha^{\text{variable}}_{(t+\text{offset})\bmod 12} \cdot y^{\text{variable}}_t
$$

where `offset = normalize_start_month(start)`.

**Parameters:**
- `months`: Number of months to compute contributions
- `start`: Calendar start date for fraction rotation
- `seed`: Controls reproducibility of variable income
- `output`: `"array"` returns `np.ndarray`, `"series"` returns `pd.Series` (default)
- `n_sims`: Number of independent simulations (only `output="array"` supports n_sims > 1)

**Returns:**
- `n_sims=1`, `output="array"`: `np.ndarray` of shape `(months,)`
- `n_sims>1`, `output="array"`: `np.ndarray` of shape `(n_sims, months)`
- `n_sims=1`, `output="series"`: `pd.Series` indexed by first-of-month dates

**Default fractions** (if `monthly_contribution` is `None`):
- Fixed: 30% each month
- Variable: 100% each month

**Custom fractions** via attribute:
```python
income.monthly_contribution = {
    "fixed": [0.35]*12,      # Jan-Dec fractions
    "variable": [1.0]*12
}
contrib = income.contributions(months=24, start=date(2025, 9, 1))
```

- Contributions are floored at zero (no negative values)
- The 12-month arrays repeat cyclically for horizons > 12 months

---

#### Statistical methods

**`income_metrics(months, start=None, variable_threshold=None)`**

Returns `IncomeMetrics` frozen dataclass with:
```python
@dataclass(frozen=True)
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

Convenience wrapper that returns `income_metrics()` as a rounded pandas Series.

---

#### Visualization methods

**`plot_income(months, start=None, ...)`**

Plots fixed, variable, and total income streams.

**Key parameters:**
- `ax`, `figsize`, `title`, `legend`, `grid`
- `ylabel_left`, `ylabel_right`: axis labels
- `dual_axis`: `"auto"` | `True` | `False` (default: `"auto"`)
- `dual_axis_ratio`: threshold for automatic dual-axis activation (default: 3.0)
- `show_trajectories`: show individual Monte Carlo paths (default: `True`)
- `show_confidence_band`: show statistical intervals (default: `False`)
- `trajectory_alpha`: transparency for trajectories (default: 0.07)
- `confidence`: confidence level for bands (default: 0.9)
- `n_simulations`: number of Monte Carlo simulations (default: 500)
- `colors`: `{"fixed": "black", "variable": "gray", "total": "blue"}`
- `save_path`, `return_fig_ax`

**Dual-axis support:** automatic when scales differ by `dual_axis_ratio`.

**`plot_contributions(months, start=None, ...)`**

Plots total monthly contributions with optional Monte Carlo trajectories and confidence bands.

**Key parameters:**
- `ax`, `figsize`, `title`, `legend`, `grid`, `ylabel`
- `show_trajectories`: show individual Monte Carlo paths (default: `True`)
- `show_confidence_band`: show statistical intervals (default: `False`)
- `trajectory_alpha`, `confidence`, `n_simulations`
- `colors`: `{"total": "blue", "ci": "orange"}`
- `save_path`, `return_fig_ax`

**`plot(mode="income"|"contributions", ...)`**

Unified wrapper that dispatches to `plot_income()` or `plot_contributions()`.

---

#### Serialization

**`to_dict()` / `from_dict(payload)`**

Serialize/deserialize model configuration for persistence.
- Handles `salary_raises` date conversion (ISO format strings)
- Supports `None` components (fixed-only or variable-only models)

```python
# Serialize
data = income.to_dict()
# {
#     "fixed": {"base": 1400000.0, "annual_growth": 0.03, ...},
#     "variable": {"base": 200000.0, "sigma": 0.15, ...}
# }

# Deserialize
income = IncomeModel.from_dict(data)
```

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

### B) Fixed-only or variable-only models
```python
# Fixed income only (no variable component)
fixed_only = IncomeModel(
    fixed=FixedIncome(base=1_400_000.0, annual_growth=0.03),
    variable=None
)

# Variable income only (no fixed component)
variable_only = IncomeModel(
    fixed=None,
    variable=VariableIncome(base=500_000.0, sigma=0.20, seed=42)
)

# Both work seamlessly
total_fixed = fixed_only.project(months=12, start=date(2025, 1, 1))
total_var = variable_only.project(months=12, start=date(2025, 1, 1))
```

### C) From income to contributions

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

### D) Fixed income with salary raises
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

### E) Variable income with seasonality + noise
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

### F) Vectorized Monte Carlo simulations
```python
# Generate 500 independent income scenarios in one call
sims = income_var.project(months=24, n_sims=500)
# sims.shape → (500, 24)

# Compute statistics
mean_income = sims.mean(axis=0)      # shape: (24,)
std_income = sims.std(axis=0)        # shape: (24,)
percentile_5 = np.percentile(sims, 5, axis=0)

# For contributions
contrib_sims = income.contributions(months=24, n_sims=500, output="array")
# contrib_sims.shape → (500, 24)

# Full model projection with multiple sims
result = income.project(months=24, n_sims=500, output="array")
# result["total"].shape → (500, 24)
# result["fixed"].shape → (500, 24)
# result["variable"].shape → (500, 24)
```

### G) Statistical summary
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

### H) Visualization with Monte Carlo trajectories
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
All projection methods support an `output` parameter to control return type:
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

### 7. Vectorized `n_sims` for Monte Carlo efficiency
The `n_sims` parameter enables batch generation of multiple independent simulations:
- Single memory allocation and NumPy vectorization throughout
- ~100x speedup vs. sequential calls for typical workloads
- Essential for Monte Carlo simulations in optimization

### 8. Optional components (fixed-only or variable-only)
`IncomeModel` supports partial configurations:
- `IncomeModel(fixed=FixedIncome(...), variable=None)` — salary-only model
- `IncomeModel(fixed=None, variable=VariableIncome(...))` — freelance-only model
- At least one component must be provided (validation in `__post_init__`)

---

## Implementation notes

- **Rate conversion**: `annual_to_monthly(g) = (1+g)^(1/12) - 1` ensures geometric compounding
- **Calendar alignment**: `month_index(start, months)` generates `pd.DatetimeIndex` of first-of-month dates
- **Seasonality rotation**: `normalize_start_month(start)` returns 0-indexed month offset (0=Jan, 11=Dec)
- **Non-negativity**: all income and contribution values are floored at zero after transformations
- **Seed propagation**: `seed=None` in methods uses instance seed; both `None` generates non-deterministic noise
- **Validation**: Uses `ValidationError` from `exceptions` module for input validation
- **Type hints**: Uses `MonthlyContributionDict` and `PlotColorsDict` from `types` module

---

## API Summary

| Class | Type | Purpose |
|-------|------|---------|
| `FixedIncome` | frozen dataclass | Deterministic income with growth and raises |
| `VariableIncome` | frozen dataclass | Stochastic income with seasonality and noise |
| `IncomeModel` | dataclass | Unified facade combining streams |
| `IncomeMetrics` | frozen dataclass | Statistical summary container |

**Key methods:**

| Method | Class | Returns |
|--------|-------|---------|
| `project(months, start, output, n_sims)` | All | Array, Series, or DataFrame |
| `contributions(months, start, seed, output, n_sims)` | `IncomeModel` | Array or Series |
| `income_metrics(months, start, variable_threshold)` | `IncomeModel` | `IncomeMetrics` |
| `summary(months, start, variable_threshold, round_digits)` | `IncomeModel` | `pd.Series` |
| `plot_income(...)` | `IncomeModel` | Plot or (fig, ax) |
| `plot_contributions(...)` | `IncomeModel` | Plot or (fig, ax) |
| `plot(mode, ...)` | `IncomeModel` | Plot or (fig, ax) |
| `to_dict()` / `from_dict(payload)` | `IncomeModel` | Serialization |
