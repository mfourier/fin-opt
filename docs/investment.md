# `investment` — Philosophy and Role in FinOpt

> **Core idea:** model **how capital grows** when you make monthly contributions and face returns (constant, scenario-based, or simulated).  
> `investment.py` is the **accumulation engine**: it takes monthly contributions (from `income.py` or contribution rules) and return paths, and produces the **wealth trajectory** and **key metrics**.

---


## Main Components


### 1) Capital accumulation (single asset)

**Function:** `simulate_capital(contributions, returns, start_value=0.0, ...)`

**Monthly dynamics**
$$
W_{t+1} = (W_t + a_t)\,(1 + r_t)
$$

- `contributions`: vector \(a_t\) (negatives allowed for withdrawals).  
- `returns`: either a scalar (broadcast to all months) or a path \((r_t)\) of length \(T\).  
- Optional safeguard: `clip_negative_wealth=True` floors wealth at zero after each step.  
- Index handling: if `index_like` is provided, the output attempts to reuse a monthly `DatetimeIndex`.

**When to use:** when treating the portfolio as a single aggregate asset (e.g., a balanced fund or a single savings account).

---

### 2) Multiple independent portfolios / accounts

**Function:** `simulate_portfolio(contributions_matrix, returns_matrix, start_values=0.0, ...)`

Extends `simulate_capital` to $K$ parallel accounts. Each column in the input matrix represents one account (e.g., *housing*, *emergency*, *brokerage*).  

**Per-account dynamics** (\(k=1,\dots,K\)):
$$
W^{(k)}_{t+1} = \big(W^{(k)}_{t} + a^{(k)}_{t}\big)\,(1 + r^{(k)}_{t})
$$

**Aggregate wealth**:
$$
W_t = \sum_{k=1}^{K} W^{(k)}_{t}, 
\qquad 
\sum_{k=1}^{K} a^{(k)}_{t} = a_t
$$

**Inputs**
- `contributions_matrix`: $(T, K)$ array or DataFrame with per-account contributions.  
- `returns_matrix`: scalar, $(T,)$ path, or $(T, K)$ matrix of returns.  
- `start_values`: scalar or vector of length $K$ with initial wealths.  
- `column_names`: optional labels for accounts; by default adds `"acct_1" ... "acct_K"`.  
- Includes a `"total"` column (sum across accounts) unless disabled.

**When to use:** when tracking multiple accounts with distinct contributions and return paths, while also needing an aggregate wealth trajectory.

