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

---

# Wealth Optimization Problem with Time-Dependent Goals

## Description

We aim to construct feasible investment plans that allocate contributions across multiple portfolios over time, so that specific financial targets are met at designated times, while optionally maximizing (or minimizing) an objective function, such as total wealth or exposure to risk.

The evolution of the wealth of each portfolio $m \in \mathcal{M}$ at month $t$ is given by:

$$
W_{t+1}^m = \big(W_t^m + A_t^m\big)\,(1 + R_t^m)
$$

where:  
- $W_t^m$ is the wealth of portfolio $m$ at month $t$.  
- $A_t^m$ is the monthly contribution to portfolio $m$ at month $t$, determined by the allocation policy $X = \{x_{t}^m\}$.  
- $R_t^m$ is the return of portfolio $m$ at month $t$.  

The allocation policy must satisfy **budget constraints** and non-negativity:

$$
x_{t}^m \ge 0, \quad \sum_{m \in \mathcal{M}} x_{t}^m = 1, \quad \forall t \in \{0,1,\dots,T\}
$$

We define the **feasible allocation set** for horizon $T$ as:

$$
\mathcal{X}_T = \Big\{ X \in \mathbb{R}_{\ge 0}^{T \times |\mathcal{M}|} \;\big|\; \sum_{m \in \mathcal{M}} x_{t}^m = 1, \;\forall t \in \{0,1,\dots,T\} \Big\}.
$$

## Affine Wealth Representation

The recursive wealth evolution formula:

$$
W_{t+1}^m = (W_t^m + A_t^m) \,(1 + R_t^m)
$$

can be equivalently expressed in a **closed-form (affine) representation**:

$$
\boxed{
W_{t}^m = W_0^m \prod_{r=0}^{t-1} (1 + R_r^m) + \sum_{s=0}^{t-1} A_s^m \prod_{r=s}^{t-1} (1 + R_r^m)
}
$$

To simplify notation, we define the **accumulation factor** from month $s$ to month $t$ for portfolio $m$ as:

$$
F_{s,t}^m := \prod_{r=s}^{t-1} (1 + R_r^m), \quad 0 \le s \le t.
$$

Using this, and the relation $A_s^m = A_s x_s^m$, the wealth at time $t$ can be expressed in **closed-form affine representation**:

$$
\boxed{
W_t^m = W_0^m \, F_{0,t}^m + \sum_{s=0}^{t-1} A_s \, x_s^m \, F_{s,t}^m
}
$$

**Key consequences:**

1. Each term $A_s x_s^m F_{s,t}^m$ shows the **linear contribution** of allocation $x_s^m$ to the final wealth, scaled by the accumulated returns.  
2. The initial wealth $W_0^m$ grows independently of contributions through $F_{0,t}^m$.  
3. Wealth $W_t^m$ is an **affine function of the allocation policy** $X$.  
4. Constraints or objectives that depend on $W_t^m(X)$ are therefore **linear-affine in the decision variables**, simplifying optimization.  
5. Gradients with respect to allocations are immediate:
$$
\frac{\partial W_t^m}{\partial x_s^m} = A_s \, F_{s,t}^m,
$$
facilitating analytical or numerical optimization methods.


## Financial Targets

We define a set of goals $\mathcal{G}$ as triplets $(t,m,b_t^m,\varepsilon_t^m)$, each specifying a portfolio $m$ and a target time $t$ with a threshold $b_t^m$ and a probability tolerance $\varepsilon_t^m$:

$$
\mathcal{G} = \{(t,m,b_t^m,\varepsilon_t^m) \;|\; \text{we want } \mathbb{P}(W_t^m(X) \ge b_t^m) \ge 1-\varepsilon_t^m \}.
$$

## Nested Optimization Formulation

We seek the **minimum time** $T$ to achieve the goals, while maximizing (or minimizing) an objective function $f(X)$:

$$
\min_{T \in \mathbb{N}} \;\; 
\Bigg\{ 
\max_{X \in \mathcal{X}_T} f(X) \;\;\text{s.t.}\;\; 
\mathbb{P}\big(W_t^m(X) \ge b_t^m\big) \ge 1 - \varepsilon_t^m, \;\forall (t,m,b_t^m,\varepsilon_t^m) \in \mathcal{G}
\Bigg\}.
$$

- The **inner problem** ($\max_{X \in \mathcal{X}_T} f(X)$) finds the best feasible allocation policy for a given horizon $T$, satisfying all goals in $\mathcal{G}$, $f(X)$ can include expected wealth, risk-adjusted return, or other financial metrics.  
- The **outer problem** ($\min_T$) finds the minimum horizon $T$ for which a feasible allocation policy exists.

## Notes

- This framework allows setting different goals for each portfolio at different times, e.g.:  
  - $2M$ in an emergency account at month 12 with probability $\ge 1-\varepsilon$.  
  - $12M$ in a housing account at month 24 with probability $\ge 1-\varepsilon$.  
  - $4M$ in the emergency account at month 24 with probability $\ge 1-\varepsilon$.  

- The objective function $f(X)$ can model different strategies, e.g.:  
  - Maximize expected total wealth.  
  - Minimize risk or volatility.  
  - Optimize a combination of financial metrics.
 
---
