# FinOpt — `optimization` Module (Design & Philosophy)

> **Tagline:** *Simulation- and optimization‑driven personal finance planning.*

This document explains the **philosophy**, **problem‑first API design**, and **mathematical formulations** implemented in `optimization.py`. The module provides small, composable solvers that integrate tightly with FinOpt’s simulation stack to answer practical questions such as:

- *What is the **minimum constant monthly contribution** I need to reach a target by a fixed date?*
- *Given a constant contribution, **how many months** do I need to reach a target?*
- *Under uncertainty, **what is the probability** of achieving one or more goals (chance‑constraints)?*

The module favors **closed‑form solutions**, **search**, and **Monte Carlo wrappers** over heavy external solver dependencies, keeping the MVP simple, fast, and deterministic by default.

---

## 1) Design Principles

- **Problem‑first API**: Each optimization task has explicit `@dataclass` inputs and outputs (clarity, testability).
- **Deterministic by default**: Stochasticity only appears when you provide random paths or seeds.
- **Tight integration**: Solvers compose with:
  - `income.py` (`IncomeModel` → contribution paths),
  - `investment.py` (`simulate_capital`, `fixed_rate_path`, `lognormal_iid`, `compute_metrics`),
  - `simulation.py` (`ScenarioConfig`, `SimulationEngine`),
  - `goals.py` (`Goal`, `evaluate_goals`),
  - `utils.py` (`ensure_1d`, `check_non_negative`).
- **Extensible**: A lightweight **solver registry** allows registering alternatives (e.g., MILP/QP versions later).

---

## 2) Core Optimization Problems

### 2.1 Minimum Constant Contribution for a Fixed Horizon

**Question.** Given:
- target \( B \) at month \( T \),
- initial wealth \( W_0 \),
- monthly (arithmetic) returns path \( r_0,\dots,r_{T-1} \),

find the smallest **constant** contribution \( a \) such that final wealth \( W_T \ge B \).

**Dynamics.**
$$
W_{t+1} = (W_t + a)\,(1 + r_t),\quad t = 0,\dots,T-1.
$$

Define the backward growth factors
$$
G_t = \prod_{u=t}^{T-1} (1 + r_u),\quad G_T = 1.
$$

Then terminal wealth is
$$
W_T = W_0\,G_0 + a \sum_{t=0}^{T-1} G_{t+1}.
$$

Let the **annuity factor** be
$$
\mathrm{AF} = \sum_{t=0}^{T-1} G_{t+1}.
$$

The **closed‑form solution** is

$$
a^* = \max\!\Big(0,\; \frac{B - W_0\,G_0}{\mathrm{AF}}\Big) \quad \text{(when AF \(> 0\)).}
$$

The solver `_solve_min_constant_contribution_closed_form` computes $ G_t $, returns $ a^* $, $ \mathrm{AF} $, $ W_0G_0 $, and $ T $. A non‑negativity clamp is applied by default (configurable).

---

### 2.2 Minimum Time Given a Constant Contribution

**Question.** Given a **fixed contribution** \( a \), find the **smallest horizon** \( T \) such that \( W_T \ge B \).

We keep the same dynamics
$$
W_{t+1} = (W_t + a)\,(1 + r_t),
$$
but now we **search** over \( T \) using a **binary search** on the prefix of the provided return path \( r_{0:(T-1)} \). For a candidate \( T \), we simulate wealth via `simulate_capital` with length \( T \); if the target is met, we try a smaller \( T \); otherwise, we try a larger one. The solver returns the minimal feasible \( T_{\hat{}} \) and its wealth path. If no \( T \) within the allowed range meets \( B \), the solver returns the upper bound and its path (useful diagnostics).

---

### 2.3 Chance‑Constraints (Success Probabilities Across Goals)

**Question.** With **uncertain returns** and **income variability**, what is the probability of meeting **each** goal and **all jointly** by their deadlines?

We use `SimulationEngine` to generate **Monte Carlo** scenarios (IID lognormal monthly returns in the MVP). For each simulated wealth path, we evaluate a set of `Goal`s via `evaluate_goals` and count successes.

Let \( \mathcal{G} = \{g_1,\dots,g_M\} \) be the goals set and simulate \( K \) wealth paths \( \{W^{(k)}\}_{k=1}^K \). Define indicators
$$
I_m^{(k)} = \mathbf{1}\{W^{(k)}_{T_m} \ge B_m\},
$$
then the **per‑goal success probability** estimator is
$$
\hat{p}_m = \frac{1}{K}\sum_{k=1}^K I_m^{(k)},
$$
and the **joint success probability** estimator is
$$
\hat{p}_{\text{joint}} = \frac{1}{K}\sum_{k=1}^K \prod_{m=1}^M I_m^{(k)}.
$$

The solver returns a `pd.Series` with \( \hat{p}_m \) (indexed by goal name) and a `summary` dict with `{"joint_success": \hat{p}_{joint}, "paths": K}`.

---

## 3) Public Facade

- `min_constant_contribution(inp: MinContributionInput) -> MinContributionResult`  
- `min_time_given_contribution(inp: MinTimeInput) -> MinTimeResult`  
- `chance_constraints(inp: ChanceConstraintsInput) -> ChanceConstraintsResult`

Each facade dispatches to a registered solver (default names below).

```python
register_solver("min_contribution.closed_form", _solve_min_constant_contribution_closed_form)
register_solver("min_time.binary_search", _solve_min_time_given_contribution)
register_solver("chance_constraints.monte_carlo", _solve_chance_constraints)
```

You can plug in alternatives by registering your own function under a new key.

---

## 4) Types (Inputs/Outputs)

### 4.1 Minimum Constant Contribution
- **Input** `MinContributionInput`  
  `target_amount: float`, `start_wealth: float`, `returns_path: Sequence[float]`, `non_negative: bool = True`
- **Output** `MinContributionResult`  
  `a_star: float`, `annuity_factor: float`, `growth_W0: float`, `T: int`

### 4.2 Minimum Time
- **Input** `MinTimeInput`  
  `contribution: float`, `start_wealth: float`, `returns_path: Sequence[float]`, `success_threshold: float`, `search_lo_hi: Optional[Tuple[int,int]]`
- **Output** `MinTimeResult`  
  `T_hat: int`, `wealth_path: pd.Series`

### 4.3 Chance‑Constraints
- **Input** `ChanceConstraintsInput`  
  `goals: Iterable[Goal]`, `income_model: IncomeModel`, `scen_cfg: ScenarioConfig`, `mc_paths: int = 1000`
- **Output** `ChanceConstraintsResult`  
  `success_prob_by_goal: pd.Series`, `summary: Dict[str, float]`

---

## 5) Integration with the Simulation Stack

- **Income**: `IncomeModel.contributions_from_proportions(...)` turns fixed/variable income into contribution paths, parameterized by `alpha_fixed` and `beta_variable`.
- **Returns**: Deterministic cases use `fixed_rate_path`. Stochastic cases use `lognormal_iid` configured by `ScenarioConfig (mc_mu, mc_sigma, mc_paths, seed)`.
- **Wealth**: `simulate_capital` implements the fundamental wealth recursion and returns a time‑indexed series.
- **Metrics**: `compute_metrics` produces `final_wealth`, `total_contributions`, `cagr`, `vol`, and `max_drawdown` for reporting.

This modular separation lets you replace any piece (income model, returns generator, metrics) without touching the optimization solver logic.

---

## 6) Usage Snippets

### 6.1 Minimum Contribution (Closed Form)

```python
from fin_opt.src.optimization import MinContributionInput, min_constant_contribution
from fin_opt.src.investment import fixed_rate_path

T = 24
r = fixed_rate_path(T, 0.004)  # 0.4% monthly
inp = MinContributionInput(target_amount=20_000_000.0, start_wealth=0.0, returns_path=r)
res = min_constant_contribution(inp)
print(res.a_star, res.annuity_factor, res.growth_W0, res.T)
```

### 6.2 Minimum Time (Binary Search)

```python
from fin_opt.src.optimization import MinTimeInput, min_time_given_contribution
from fin_opt.src.investment import fixed_rate_path

r = fixed_rate_path(60, 0.004)  # up to 60 months available
inp = MinTimeInput(contribution=700_000.0, start_wealth=0.0,
                   returns_path=r, success_threshold=6_000_000.0)
res = min_time_given_contribution(inp)
print(res.T_hat)
print(res.wealth_path.tail())
```

### 6.3 Chance‑Constraints with Monte Carlo

```python
from datetime import date
from fin_opt.src.optimization import ChanceConstraintsInput, chance_constraints
from fin_opt.src.income import FixedIncome, VariableIncome, IncomeModel
from fin_opt.src.simulation import ScenarioConfig
from fin_opt.src.goals import Goal

income = IncomeModel(
    fixed=FixedIncome(base=1_400_000.0, annual_growth=0.00),
    variable=VariableIncome(base=200_000.0, sigma=0.00),
)
cfg = ScenarioConfig(
    months=24, start=date(2025, 9, 1),
    alpha_fixed=0.35, beta_variable=1.0,
    base_r=0.004, optimistic_r=0.007, pessimistic_r=0.001,
    mc_mu=0.004, mc_sigma=0.02, mc_paths=500, seed=42
)
goals = [
    Goal(name="housing", target_amount=20_000_000.0, target_month_index=23),
    Goal(name="emergency", target_amount=6_000_000.0, target_month_index=11),
]
inp = ChanceConstraintsInput(goals=goals, income_model=income, scen_cfg=cfg, mc_paths=500)
res = chance_constraints(inp)
print(res.success_prob_by_goal)
print(res.summary)
```

---

## 7) Extensibility & Future Directions

- **Alternative return models**: regime‑switching, historical bootstraps, factor models.
- **Glidepaths and schedules**: replace the constant \( a \) with time‑varying \( a_t \) and solve via LP/MILP.
- **Multi‑asset allocation**: couple with `simulate_portfolio` and introduce decision variables for weights (QP/SOCP).
- **Taxes, fees, and constraints**: transaction costs, contribution caps, and goal priorities.
- **Risk‑aware objectives**: e.g., minimize \( a \) subject to \( \mathbb{P}(W_T \ge B) \ge \tau \), or minimize time subject to chance‑constraints.

The **solver registry** makes it simple to provide parallel implementations (e.g. `min_contribution.lp`, `min_time.dp`) while preserving a stable public facade.

---

## 8) Testing Philosophy

- **Unit tests** for closed‑form and binary search with controlled paths.
- **Property‑based tests** (optional) to check monotonicity: larger \( B \) → larger \( a^\* \); larger \( a \) → smaller \( T_{\hat{}} \).
- **Integration smoke tests**: run the manual block to verify end‑to‑end consistency with `SimulationEngine` and `goals`.

---

## 9) Summary

`optimization.py` delivers **clean, minimal solvers** that answer key personal‑finance questions with **transparent math** and **reproducible simulations**. Problems are framed so that they can scale from MVP (closed form + search + Monte Carlo) to advanced formulations (LP/QP/MILP/SOCP) without changing the user‑facing API.
