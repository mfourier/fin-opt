# FinOpt — Optimization Problems (Technical Design)

> Tagline: *Intelligent financial planning based on simulation and optimization.*

This document describes the **theoretical and technical framework** of **FinOpt**, a modular system for **personal financial planning** through simulation and optimization.  

The goal is to connect **user objectives** (housing fund, emergency reserve, retirement) with **investment strategies**, under uncertainty in both **income** and **returns**.

---

# 0. System Overview

FinOpt consists of **modular components**, each with a clear role:

- [`income.py`] → **Cash inflows** (fixed salary, variable income with seasonality/noise).  
- [`investment.py`] → **Capital accumulation** under return paths (single or multi-asset).  
- [`simulation.py`] → **Scenario orchestration** (base/optimistic/pessimistic, Monte Carlo).  
- [`goals.py`] → **Goal evaluation** (success/shortfall ratios, required contributions, contribution splitting).  
- [`utils.py`] → **Shared helpers** (validation, rate conversion, index handling, drawdown, CAGR).  

These modules connect seamlessly:

$$
\text{Income} \;\longrightarrow\; \text{Contributions} \;\longrightarrow\; \text{Investment Growth} \;\longrightarrow\; \text{Scenario Simulation} \;\longrightarrow\; \text{Goal Evaluation}.
$$

---

# 1. Theoretical Framework

We adopt a **discrete monthly horizon**:

$$
t = 0,1,\dots,T-1
$$

## Incomes and Contributions

- Net income:

$$
y_t = y_t^{\text{fixed}} + y_t^{\text{variable}}
$$

  - $y_t^{\text{fixed}}$: deterministic salary (with optional annual growth).  
  - $y_t^{\text{variable}}$: seasonal/stochastic stream.  

- Contributions (decision variable):

$$
a_t \in [0,\; y_t - g_t]
$$

where $g_t$ are monthly expenses.

- MVP contribution rule:

$$
a_t = \alpha\,y_t^{\text{fixed}} + \beta\,y_t^{\text{variable}}, \quad \alpha,\beta \in [0,1].
$$

---

## Goals

A set $\mathcal{M}$ of goals $m$, each defined by:
- Target amount $B_m$.  
- Deadline $T_m$ (date or month index).  

Tracking:
- Per goal: $W_{m,t}$  
- Aggregate: $W_t = \sum_m W_{m,t}$  

---

## Investment Dynamics

- Single asset (MVP):

$$
W_{t+1} = (W_t + a_t)\,(1+r_t).
$$

- Multi-asset portfolio:

$$
W_{t+1} = \big(W_t + a_t\big)\,(1+ R_{p,t}), \quad R_{p,t} = \sum_i w_{i,t}R_{i,t}.
$$

---

## Scenarios

- **Deterministic (three-case):**
  - Base, optimistic, pessimistic fixed monthly rates.
- **Stochastic (Monte Carlo):**
  - IID lognormal returns

$$
R_t \sim \text{LogNormal}(\mu,\sigma).
$$

- Reproducibility via explicit RNG seeds.

---

## Metrics

For a wealth path $\{W_t\}$:
- Final wealth $W_T$.  
- Total contributions $\sum_t a_t$.  
- CAGR:

$$
\text{CAGR} = \Big(\tfrac{W_T}{W_0}\Big)^{1/\text{years}}-1.
$$

- Volatility of increments.  
- Max drawdown:

$$
\max_t \frac{W_t - \max_{u\le t} W_u}{\max_{u\le t} W_u}.
$$

---

# 2. Optimization Problems

We prioritize in **three phases**, increasing in complexity.

---

## Phase I — Core (MVP)

### (1) Minimum monthly contribution (fixed horizon)

**Question:** Smallest constant $a$ such that $W_{m,T_m}\ge B_m$.

Closed form with time-varying returns $\{r_t\}$:

$$
W_T = W_0 G_0 + a\sum_{t=0}^{T-1}G_{t+1}, 
\quad G_t = \prod_{u=t}^{T-1}(1+r_u).
$$

$$
a^* = \max\!\Big(0,\; \frac{B - W_0 G_0}{\sum_{t=0}^{T-1} G_{t+1}}\Big).
$$

- Solver: closed form or root-finding.  
- Output: $a^*$.

---

### (2) Minimum time (fixed contribution)

**Question:** Given constant $a$, what is the smallest $T$ with $W_{m,T}\ge B_m$?

- Solver: binary search over $T$.  
- Output: $\hat T$ and attainment probability curve $\Pr(W_{m,t}\ge B_m)$.

---

## Phase II — Portfolio & Risk

### (3) Portfolio allocation
- **Mean–Variance (QP):**  

  $$
  \max_w \;\mu^\top w - \lambda\,w^\top \Sigma w
  $$

- **CVaR minimization (LP)**.  
- **Robust optimization** under uncertainty sets.

Output: $w$, efficient frontier.

---

### (4) Probability of success (chance constraints)

- Maximize 

$$
\Pr(W_{m,T_m}\ge B_m)
$$ 

or enforce 

$$
\Pr(W_{m,T_m}\ge B_m) \ge 1-\varepsilon.
$$  

- Implemented via scenario counting or CVaR surrogates.  

Output: success probability per goal.

---

## Phase III — Multi-goal Planning & Dynamics

### (5) Multi-goal allocation
- **Lexicographic:** prioritize high-priority goals first.  
- **Shortfall-penalized:**  

$$
\min \sum_m \alpha_m \xi_m \quad \text{s.t. } W_{m,T_m}+\xi_m \ge B_m
$$  

- **Utility-based:** maximize $\sum_m u_m(\text{attainment}_m)$.  

---

### (6) Dynamic programming / RL
- State:  

$$
x_t = (W_t,\{W_{m,t}\},y_t,g_t).
$$  

- Action: $a_{m,t},w_{i,t}$.  
- Transition: wealth recurrence with stochastic returns.  
- Methods: Approximate DP, policy gradients, actor–critic.

---

### (7) Glidepath & smooth rebalancing

- Penalize sharp changes in contributions/weights:

$$
\lambda_a\sum_t (a_t-a_{t-1})^2 + \lambda_w \sum_t \|w_t-w_{t-1}\|.
$$

---

# 3. Validation & Testing

- **Unit tests**: income growth, metrics, goal evaluation.  
- **Integration tests**: simulate baseline scenarios, validate metrics consistency.  
- **Property-based tests**: conservation  

  $$
  \sum_m a_{m,t}\le y_t-g_t
  $$

- **Reproducibility**: fixed RNG seeds.  
- **Sensitivity analysis**: perturb $\mu,\Sigma,y_t,r$.

---

# 4. Illustrative Example (Current Parameters)

- Monthly income: **1.4M CLP fixed + 0.2M CLP variable**.  
- Contribution rule: $\alpha=0.35$, $\beta=1.0$ → ~0.7M CLP invested monthly.  

### Problem (1): Required contribution for housing fund

Target $B=20$M CLP, horizon 24 months, $r=0.004$ (0.4% monthly).  

Closed form yields:  

$$
a^* \approx 0.8\ \text{M CLP}.
$$

### Problem (2): Minimum time for emergency fund

Target $B=6$M CLP (≈ 6 months of expenses).  

Simulation shows with $a\approx 0.7$M CLP/month:  

$$
\hat T \approx 10\ \text{months}.
$$

### Goal evaluation (`goals.py`)

```python
goals = [
  Goal("housing", target_amount=20_000_000, target_month_index=23),
  Goal("emergency", target_amount=6_000_000, target_month_index=11),
]
df = evaluate_goals(results["base"].wealth, goals)
