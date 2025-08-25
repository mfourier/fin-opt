# `optimization` — Philosophy and Role in FinOpt

> **Core idea:** turn **financial goals** into **optimization problems**, searching over time horizons and allocation strategies to determine if, when, and how they can be satisfied.  
> `optimization.py` acts as the **decision layer**: while `scenario.py` simulates wealth paths, `optimization.py` asks *what allocations make goals feasible, and in the shortest time possible?*

---

## Why a dedicated optimization module?

- **Separation of roles**
  - `income.py` → how much money is available.  
  - `investment.py` → how capital grows given contributions and returns.  
  - `scenario.py` → generates possible wealth paths.  
  - `goals.py` → defines success criteria.  
  - `optimization.py` → searches over *time* and *allocations* to satisfy goals.

- **Factoring feasibility**  
  Not all targets are attainable. Optimization starts with feasibility checks, then minimizes time or resources.

- **Extensibility**  
  The module is designed to grow: from MVP feasibility solvers to more advanced optimization routines (linear programming, stochastic optimization, dynamic programming).

---

## Problem 1 — Minimum Time Multi-Goal Feasibility

**Question:**  
Find the *smallest horizon* \(T\) and a contribution allocation matrix  
\(C \in \mathbb{R}^{T \times K}\) such that every goal is satisfied:

$$
W^{(k)}_{T} \;\;\ge\;\; B_k, \quad \forall k \in \{1,\dots,K\}.
$$

---

### Dynamics

Each account \(k\) evolves as:

$$
W^{(k)}_{t+1} = \big(W^{(k)}_{t} + a_t C_{t,k}\big)(1 + r^{(k)}_t).
$$

Here:
- $a_t$ = total contribution at time $t$ (from `income.py`).  
- $C_{t,k}$ = fraction allocated to account $k$.  
- $r^{(k)}_t$ = return of account $k$ at time $t$.  
- $W^{(k)}_0$ = initial wealth of account $k$ (not necessarily zero).

---

### Change of variables

Define decision variables in **peso space**:

$$
x_{t,k} := a_t C_{t,k}, \quad \sum_{k} x_{t,k} = a_t.
$$

The recursion unrolls into a **closed form**:

$$
W^{(k)}_{T} = W^{(k)}_0 \, G^{(k)}_0 \;+\; \sum_{t=0}^{T-1} x_{t,k} \, H^{(k)}_{t+1},
$$

where:

- **Growth factor of initial wealth**:
  $$
  G^{(k)}_0 = \prod_{u=0}^{T-1} (1+r^{(k)}_u).
  $$
- **Future value factor of contributions**:
  $$
  H^{(k)}_{t+1} = \prod_{u=t+1}^{T-1} (1+r^{(k)}_u).
  $$

Thus, terminal wealth is **affine in contributions**: a linear combination of peso allocations plus the scaled starting wealth.

---

### Feasibility as Linear Program

For a fixed \(T\), the feasibility problem becomes:

- **Variables:** $x_{t,k} \ge 0$.  
- **Row sums:**  
  $$
  \sum_k x_{t,k} = a_t .
  $$
- **Target constraints:**  
  $$
  \sum_t x_{t,k} \, H^{(k)}_{t+1} \;\;\ge\;\; B_k - W^{(k)}_0 G^{(k)}_0, 
  \quad \forall k.
  $$

If a feasible solution exists, we can reconstruct a normalized allocation matrix \(C\) from \(x\).

- **Objective:** minimize $T$.  
- **Search:** increase $T=t_{\min}, t_{\min}+1, \dots, T_{\max}$ until feasibility holds.  
  - `t_min` can be set manually or `"auto"`, in which case a lower bound is computed by checking whether dedicating *all* contributions to each goal could theoretically meet its deficit.

---

## Backends

1. **Scipy LP backend**  
   Uses `scipy.optimize.linprog` with objective = 0 (pure feasibility).  

2. **Greedy backend**  
   Heuristic that:
   - Computes remaining deficits in future-value space.  
   - At each month $t$, allocates $a_t$ to the account with the highest multiplier $H^{(k)}_{t+1}$.  
   - Caps allocations once a goal’s deficit is covered.  
   - Resolves ties deterministically (or with RNG seed if provided).  

---

## Outputs

Both backends return:

- $T^*$: the minimal feasible horizon.  
- $C^*$: row-stochastic allocation matrix (fractions).  
- $A$: allocation in pesos per account and month.  
- `wealth_by_account`: simulated wealth trajectories per account (plus total).  
- **Diagnostics**:  
  - `t_hit_by_account`: first month each goal is reached (or -1 if never).  
  - `margin_at_T`: wealth minus target at horizon $T$.  
  - `feasible`: whether a solution was found within $T_{\max}$.

---

## Roadmap of Extensions

1. **Phase I — Feasibility search**
   - Incremental search of $T$.  
   - LP and greedy backends as implemented.  

2. **Phase II — Structured allocations**
   - Restrict $C$ to special forms:  
     - Constant weights.  
     - Glidepaths (linear tilts).  
     - Piecewise constant policies.  

3. **Phase III — Optimization under uncertainty**
   - Stochastic returns (Monte Carlo).  
   - Chance-constrained feasibility:  
     $\Pr(W^{(k)}_{T} \ge B_k) \ge 1-\varepsilon$.  

4. **Phase IV — Advanced methods**
   - Convex optimization (LP/QP).  
   - Approximate Dynamic Programming / RL for state-dependent policies.  
   - Multi-objective formulations (lexicographic, utility-based).

---

## Integration with FinOpt

1. **Income generation** (`income.py`) → contributions $a_t$.  
2. **Scenario orchestration** (`scenario.py`) → build return paths + allocation candidate $C$.  
3. **Investment accumulation** (`investment.py`) → simulate per-account wealth.  
4. **Goal evaluation** (`goals.py`) → check feasibility.  
5. **Optimization loop** (`optimization.py`) → search over $T, C$ until feasible.  

---
