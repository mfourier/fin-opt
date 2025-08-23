# `income.py` — Philosophy & Role in FinOpt

> **Purpose:** Model **where the money comes from** (and how it evolves) so the rest of FinOpt can decide **how to allocate and invest it**.  
> In FinOpt’s pipeline, `income.py` is the **entry point of cash flows**: it turns assumptions about salary and variable earnings into a clean, reproducible **monthly series** that downstream modules consume for contributions and simulations.

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
     - `FixedIncome`: predictable salary with optional **annual growth**.
     - `VariableIncome`: irregular stream with **seasonality** and **Gaussian noise**.
4. **Single responsibility**
   - `income.py` does **not** simulate returns or portfolios. It only **projects income** and derives **contribution series** from it.

---

## The three core surfaces

### 1) `FixedIncome`
A deterministic monthly base with **compounded annual growth**:
- Parameters: `base`, `annual_growth`, `name`.
- Monthly projection uses the equivalent monthly rate from `annual_growth`.
- Guarantees **non-negativity** and well-formed arrays.

**Interpretation:** models a salary that may receive annual adjustments; simple and transparent.

---

### 2) `VariableIncome`
A variable stream with optional **seasonality**, **noise**, **floor/cap**, and **annual growth**:
- `seasonality`: 12 multiplicative factors (Jan–Dec).
- `sigma`: standard deviation of noise as **a fraction of the month mean**.
- `floor` / `cap`: guardrails after noise (e.g., minimum expected side income).
- `seed`: RNG seed to make runs reproducible.
- Ensures **non-negative** results after transformations.

**Interpretation:** models tutoring, bonuses, or freelancing income whose level changes across the year and fluctuates each month.

---

### 3) `IncomeModel`
A façade that **combines streams** and produces:
- `project_monthly(months, start, as_dataframe)`: total income (and optionally the fixed/variable breakdown) for the horizon.
- `contributions_from_proportions(months, alpha_fixed, beta_variable, start)`: turns income into **monthly investment contributions**:
  \[
  a_t = \alpha \cdot y^{\text{fixed}}_t + \beta \cdot y^{\text{variable}}_t,\quad \alpha,\beta \in [0,1].
  \]
  Any negative values are **floored at zero** to keep contributions feasible.

**Interpretation:** a clean bridge from “what I earn” to “what I can invest every month”.

---

## How `income.py` powers the rest of FinOpt

- **`simulation.py`**  
  Uses `IncomeModel.contributions_from_proportions(...)` to generate the **contribution series** aligned to the simulation calendar, then combines it with deterministic or Monte Carlo **returns** to simulate wealth.
- **`investment.py`**  
  Receives the contributions from `income.py` and applies **capital accumulation**:
  \[
  W_{t+1}=(W_t+a_t)(1+R_t).
  \]
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
df = income.project_monthly(months=24, start=date(2025, 9, 1), as_dataframe=True)
```

### B) From income to contributions
```python
# Contribute 35% of fixed income and 100% of variable income
contrib = income.contributions_from_proportions(
    months=24, alpha_fixed=0.35, beta_variable=1.0, start=date(2025, 9, 1)
)
```

### C) Variable income with seasonality + noise
```python
seasonality = [1.00, 0.95, 1.05, 1.10, 1.15, 1.10, 1.00, 0.90, 0.95, 1.05, 1.10, 1.20]
income_var = VariableIncome(
    base=200_000.0,
    seasonality=seasonality,
    sigma=0.15,
    floor=50_000.0,
    cap=400_000.0,
    annual_growth=0.02,
    seed=123
)
```