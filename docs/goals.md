# `goals.py` — Philosophy and Role in FinOpt

> **Core idea:** translate **financial goals** (target amount and deadline) into **verifiable criteria** over simulated wealth trajectories.  
> `goals.py` connects what the user **wants to achieve** (e.g., “20M CLP in 24 months”) with what the simulation engine **produces** (`wealth` series), and outputs clear evaluations: **success**, **shortfall**, and **attainment ratio**.

---

## Why a dedicated goals module?

- **Grounds objectives in numbers**: converts “buy a house in 2 years” into `B_m` (amount) and `T_m` (deadline).  
- **Separates the *what* from the *how***: `investment.py` grows capital; `simulation.py` orchestrates scenarios; `goals.py` checks if results meet the target.  
- **Enables optimization**: by quantifying shortfalls, it allows solving for minimum contributions, minimum time, or allocation across multiple goals (goal-based investing).

---

## Design philosophy

1. **Minimal and deterministic**  
   - Depends only on `numpy`, `pandas`, and project utilities (`utils.py`).  
   - Deterministic: any randomness comes from return paths or contribution flows, not from this module.

2. **Consistent calendar indexing**  
   - Supports deadlines as **calendar dates** (`target_date`) or as **month indices** relative to the start (`target_month_index`).  
   - Uses helpers (`month_index`, `align_index_like`) to align goals with the wealth series.

3. **Simple, serializable interfaces**  
   - `Goal` and evaluation results can be **(de)serialized** (`to_dict`, `from_dict`) for configs and reporting.

---

## Key concepts

- **Goal (`Goal`)**: pair of (target amount `B_m`, deadline `T_m`) plus metadata (`name`, `priority`, `notes`).  
- **Evaluation**: given a wealth path, compute:  
  - `success` (boolean),  
  - `shortfall = max(0, B_m - W_{m,T_m})`,  
  - `attainment_ratio = min(W_{m,T_m}/B_m, 1)`.  
- **Contribution split** (MVP): split aggregate contributions into per-goal series by proportions summing to 1.  
- **Required constant contribution**: compute the `a*` needed to guarantee the goal under a given returns path (possibly time-varying).

---

## Main surfaces (API)

### 1) `Goal` (dataclass)
```python
Goal(
  name: str,
  target_amount: float,
  target_date: Optional[date] = None,        # or...
  target_month_index: Optional[int] = None,  # ...exactly one must be provided
  priority: int = 0,
  notes: Optional[str] = None
)
```

## 2) Goal evaluation

- **Single goal**:  
  `evaluate_goal(wealth: pd.Series, goal: Goal) -> GoalEvaluation`

- **Multiple goals**:  
  `evaluate_goals(wealth: pd.Series, goals: Iterable[Goal]) -> pd.DataFrame`

The multiple-goal version returns a **DataFrame** ordered by priority (if available), with the following columns:

- `goal`  
- `deadline_pos`  
- `deadline_timestamp`  
- `target_amount`  
- `wealth_at_deadline`  
- `success`  
- `shortfall`  
- `attainment_ratio`

**Interpretation:**  
- `attainment_ratio` allows comparison of heterogeneous goals on a [0,1] scale.  
- `shortfall` quantifies the missing amount in currency units.

## 3) Proportional contribution split (MVP)

`allocate_contributions_proportional(contributions: pd.Series, weights_by_goal: dict) -> pd.DataFrame`

- **Purpose**: obtain per-goal contribution columns that **sum** to the aggregate.  
- **Rules**: weights must be non-negative; normalized to sum to 1; index preserved.

**Use case**: build per-goal reports or run per-goal simulations (via `simulate_capital` on each column).

## 4) Required constant contribution under a returns path

`required_constant_contribution(target_amount, start_wealth, returns_path) -> float`

- **Dynamics:**
  $$
  W_{t+1} = (W_t + a)\,(1 + r_t), \quad t=0,\dots,T-1
  $$

- **Closed form for time-varying \(r_t\):**  
  Define 
  $$
  G_t = \prod_{u=t}^{T-1}(1+r_u), \quad
  AF = \sum_{t=0}^{T-1} G_{t+1}
  $$
  Then:
  $$
  W_T = W_0\,G_0 + a\,AF \;\;\Rightarrow\;\;
  a^* = \max\!\left(0, \frac{B - W_0 G_0}{AF}\right)
  $$

- **Interpretation:**  
  `a*` is the minimum constant monthly contribution required to **guarantee** reaching the target amount `B` at horizon `T`, under the given return path.

## Integration in the FinOpt workflow

1. **`income.py`** → produces aggregate contributions.  
2. **`investment.py`** → simulates wealth (single or multi-asset).  
3. **`simulation.py`** → orchestrates scenarios and outputs wealth.  
4. **`goals.py`** → evaluates goal attainment, splits contributions per goal, and computes required `a*`.
