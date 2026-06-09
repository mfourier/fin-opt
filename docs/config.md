# `config` — Type-Safe Configuration

> **Core idea:** Validate every model parameter *before* a simulation runs.
> `config.py` defines [Pydantic](https://docs.pydantic.dev/) models that enforce
> types, ranges, and cross-field constraints, and that serialize cleanly to JSON
> for the [CLI](cli.md) and [API](serialization.md).

---

## Why a configuration layer?

- **Fail fast:** Out-of-range or misspelled parameters raise `ValidationError` at construction, not deep inside a solver.
- **Self-documenting:** Each field carries a description and explicit bounds.
- **Serializable:** `model.model_dump()` / `model_dump_json()` round-trip to the JSON consumed by `finopt simulate` and the API.
- **Immutable:** Core configs are `frozen=True` with `extra="forbid"` — no silent typo'd keys.

```python
from finopt.config import SimulationConfig, OptimizationConfig

sim = SimulationConfig(n_sims=1000, seed=42, cache_enabled=True)
opt = OptimizationConfig(T_max=120, solver="ECOS", objective="balanced")

config_dict = sim.model_dump()   # → plain dict, JSON-ready
```

---

## Models Overview

| Model | Role |
|-------|------|
| `SimulationConfig` | Monte Carlo parameters |
| `OptimizationConfig` | Solver, objective, horizon search |
| `FixedIncomeConfig` / `VariableIncomeConfig` | Income streams |
| `IncomeConfig` | Combines fixed + variable with contribution rates |
| `AccountConfig` | A single investment account |
| `WithdrawalEventConfig` / `StochasticWithdrawalConfig` / `WithdrawalConfig` | Cash outflows |
| `IntermediateGoalConfig` / `TerminalGoalConfig` | Goals |
| `ScenarioConfig` | Full scenario bundling all of the above |
| `AppSettings` | Environment-variable settings (`FINOPT_*`) |

---

## `SimulationConfig`

| Field | Type | Default | Bounds |
|-------|------|---------|--------|
| `n_sims` | int | `500` | 100–10,000 |
| `seed` | int \| None | `None` | — |
| `cache_enabled` | bool | `True` | — |
| `verbose` | bool | `True` | — |

---

## `OptimizationConfig`

| Field | Type | Default | Bounds / Choices |
|-------|------|---------|------------------|
| `T_max` | int | `240` | 12–600 |
| `T_min` | int | `12` | ≥ 1, must be ≤ `T_max` |
| `solver` | str | `"ECOS"` | `ECOS`, `SCS`, `CLARABEL`, `MOSEK` |
| `objective` | str | `"balanced"` | `risky`, `balanced`, `conservative`, `risky_turnover` |
| `search_strategy` | str | `"bracketed"` | `bracketed`, `binary`, `linear` |
| `tolerance` | float | `1e-4` | 1e-6–1e-2 |
| `verbose` | bool | `True` | — |
| `warm_start` | bool | `True` | — |
| `max_iterations` | int | `1000` | 10–10,000 |

A `field_validator` enforces `T_min ≤ T_max`.

!!! note "Where the search strategy is applied"
    `search_strategy` mirrors the system default (`bracketed`, see
    [Optimization](optimization.md#search-strategies)). At runtime the strategy is
    selected by the `search_method` argument of `model.optimize()` /
    `GoalSeeker.seek()` (which defaults to `bracketed`); `OptimizationConfig`
    records the intended strategy as part of the serializable config schema.

---

## Income Configuration

### `FixedIncomeConfig`

| Field | Type | Default | Bounds |
|-------|------|---------|--------|
| `base` | float | *(required)* | ≥ 0 |
| `annual_growth` | float | `0.0` | -0.5–0.5 |
| `salary_raises` | dict \| None | `None` | date → raise amount |

### `VariableIncomeConfig`

| Field | Type | Default | Bounds |
|-------|------|---------|--------|
| `base` | float | *(required)* | ≥ 0 |
| `sigma` | float | `0.0` | 0–2.0 (fraction of base) |
| `annual_growth` | float | `0.0` | -0.5–0.5 |
| `seasonality` | list[float] \| None | `None` | 12 factors, sum to 12 |
| `floor` / `cap` | float \| None | `None` | ≥ 0 |
| `seed` | int \| None | `None` | — |

### `IncomeConfig`

Combines `fixed` and `variable` streams with `contribution_rate_fixed` and `contribution_rate_variable` (each a scalar or a 12-element list).

---

## `AccountConfig`

Maps directly to `Account.from_annual()`:

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Account identifier |
| `annual_return` | float | Expected annual return |
| `annual_volatility` | float | Annual volatility |
| `initial_wealth` | float | Starting balance |
| `display_name` | str \| None | Optional UI label |

---

## `AppSettings` — Environment Variables

`AppSettings` (a `BaseSettings`) reads `FINOPT_`-prefixed environment variables and an optional `.env` file:

```bash
FINOPT_DEBUG=true
FINOPT_CACHE_DIR=/tmp/finopt-cache
```

```python
from finopt.config import AppSettings

settings = AppSettings(_env_file=".env")
```

---

## See also

- [Serialization](serialization.md) — persisting configs and results to JSON
- [CLI](cli.md) — commands that consume these config files
- [Optimization](optimization.md) — what `OptimizationConfig` drives
