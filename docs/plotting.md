# `plotting` — Visualization Suite

> **Core idea:** One method, many views. `plotting.py` provides
> `ModelPlottingMixin`, mixed into `FinancialModel`, so every visualization is
> reached through a single `model.plot(mode, ...)` call that auto-simulates and
> caches when needed.

---

## Design

- **Unified entry point:** `FinancialModel.plot(mode, ...)` dispatches to the right specialized plot by `mode` string.
- **Auto-simulation:** Simulation-based modes run `simulate()` internally (cached) unless a pre-computed `result` is passed.
- **Mixin, not standalone:** `ModelPlottingMixin` is only meant to be mixed into `FinancialModel` — do not instantiate it directly.

```python
from datetime import date

# Auto-simulates internally (cached)
model.plot("wealth", T=24, X=X, n_sims=500, seed=42, start=date(2025, 1, 1))

# Reuse an existing optimization/simulation result
model.plot("wealth", result=result, show_trajectories=True)
```

---

## Modes

### Pre-simulation (no wealth simulation needed)

| Mode | Shows |
|------|-------|
| `"income"` | Income streams (fixed, variable, total) |
| `"contributions"` | Monthly contribution schedule |
| `"returns"` | Return distributions and trajectories |
| `"returns_cumulative"` | Cumulative return evolution |
| `"returns_horizon"` | Risk–return by investment horizon |

### Simulation-based (auto-simulates if `result` not provided)

| Mode | Shows |
|------|-------|
| `"wealth"` | Portfolio dynamics (4-panel) |
| `"allocation"` | Allocation analysis with investment gains (4-panel) |
| `"comparison"` | Compare multiple strategies |

Modes `"wealth"` and `"comparison"` require either a `result` **or** the `T` and `X` parameters so a simulation can be run.

---

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | *(required)* | One of the modes above |
| `T` | int | `None` | Horizon (simulation-based modes) |
| `X` | ndarray `(T, M)` | `None` | Allocation policy (simulation-based modes) |
| `n_sims` | int | `500` | Monte Carlo paths |
| `start` | date | `None` | Calendar start (for seasonality alignment) |
| `seed` | int | `None` | Reproducibility |
| `result` | `SimulationResult` | `None` | Bypass simulation with a precomputed result |
| `figsize` | tuple | `None` | Figure size |
| `title` | str | `None` | Override the title |
| `save_path` | str | `None` | Save the figure to disk |
| `return_fig_ax` | bool | `False` | Return `(fig, ax)` instead of showing |
| `use_cache` | bool | `True` | Reuse cached simulations |

Mode-specific options (e.g. `show_trajectories` for `"wealth"`) are passed through `**kwargs`.

---

## Examples

```python
# Pre-simulation views
model.plot("income")
model.plot("returns_horizon")

# Wealth dynamics under an optimized policy
result = model.optimize(goals=goals, optimizer=optimizer, T_max=120)
model.plot("wealth", result=result, show_trajectories=True)

# Save instead of show
model.plot("allocation", result=result, save_path="docs/images/allocation_analysis.png")

# Grab the axes for further customization
fig, ax = model.plot("wealth", result=result, return_fig_ax=True)
```

---

## See also

- [Integration Model](model.md) — `FinancialModel`, which mixes this in
- [Utilities](utils.md) — `millions_formatter`, `format_currency` used by the plots
