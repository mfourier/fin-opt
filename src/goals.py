
"""Goal modeling module for FinOpt

Concepts
--------
- Goal: target amount B_m to be achieved by a target date T_m (or by month index).
- Evaluation: success, shortfall, attainment ratio given a wealth path.
- Contribution split: proportional allocations across goals that sum
  to the aggregate contribution series produced by `IncomeModel`.
- Deterministic requirement: compute the constant contribution that would
  satisfy a goal under a *given* monthly returns path (possibly time-varying).

Design goals
------------
- Minimal dependencies (pandas, numpy, stdlib) and tight coupling to existing
  FinOpt modules (`simulation.py`, `investment.py`, `utils.py`).
- Deterministic by default; stochasticity only through provided return paths.
- Friendly serialization (`to_dict` / `from_dict`) for configs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import (
    check_non_negative,
    ensure_1d,
    month_index,
    align_index_like,
)

__all__ = [
    "Goal",
    "GoalEvaluation",
    "evaluate_goal",
    "evaluate_goals",
    "allocate_contributions_proportional",
    "required_constant_contribution",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    """A financial goal with a target amount by a target time.

    Parameters
    ----------
    name : str
        Identifier for the goal (e.g., "housing", "emergency").
    target_amount : float
        Target wealth B_m to be reached by `target_date` / `target_month_index`.
    target_date : Optional[date], default None
        Calendar deadline (1st-of-month resolution). If not provided, use
        `target_month_index` relative to the simulation start.
    target_month_index : Optional[int], default None
        Zero-based index of the deadline month within the simulation horizon.
        Exactly one of (target_date, target_month_index) should be provided.
    priority : int, default 0
        Lower value = higher priority (for lexicographic strategies if used later).
    notes : Optional[str], default None
        Free-form notes for reporting.
    """

    name: str
    target_amount: float
    target_date: Optional[date] = None
    target_month_index: Optional[int] = None
    priority: int = 0
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        check_non_negative("target_amount", float(self.target_amount))
        if (self.target_date is None) == (self.target_month_index is None):
            raise ValueError("Provide exactly one of target_date or target_month_index.")
        if self.target_month_index is not None and self.target_month_index < 0:
            raise ValueError("target_month_index must be >= 0.")

    # ----------------------- Index mapping helpers -----------------------
    def resolve_deadline_pos(
        self,
        months: int,
        start: Optional[date] = None,
        like: Optional[pd.Index | pd.Series | pd.DataFrame] = None,
    ) -> int:
        """Return the integer position (0-based) of the goal deadline within horizon.

        If `target_month_index` is set, it's returned directly (bounded by months-1).
        If `target_date` is set, it is mapped onto the simulation index.
        """
        if months <= 0:
            raise ValueError("months must be positive.")
        if self.target_month_index is not None:
            return min(int(self.target_month_index), months - 1)
        # Build / reuse a monthly DatetimeIndex (first of month) to locate the deadline.
        idx = align_index_like(months, like) if like is not None else month_index(start, months)
        deadline = pd.Timestamp(self.target_date.year, self.target_date.month, 1)
        try:
            pos = int(idx.get_indexer([deadline])[0])
        except Exception:
            # If deadline falls outside the index, clamp to the nearest bound.
            if deadline <= idx[0]:
                pos = 0
            elif deadline >= idx[-1]:
                pos = months - 1
            else:
                # If not exact MS date (unlikely), find the closest month
                diffs = np.abs((idx - deadline).days)
                pos = int(np.argmin(diffs))
        return max(0, min(pos, months - 1))

    # ----------------------- Serialization helpers -----------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_amount": float(self.target_amount),
            "target_date": None if self.target_date is None else str(self.target_date),
            "target_month_index": self.target_month_index,
            "priority": int(self.priority),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Goal":
        td = payload.get("target_date", None)
        td_parsed = None
        if isinstance(td, str) and td:
            td_parsed = pd.to_datetime(td).date()
        return cls(
            name=str(payload.get("name", "goal")),
            target_amount=float(payload.get("target_amount", 0.0)),
            target_date=td_parsed,
            target_month_index=payload.get("target_month_index"),
            priority=int(payload.get("priority", 0)),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class GoalEvaluation:
    """Evaluation result of a goal under a given wealth path."""
    goal_name: str
    deadline_pos: int
    deadline_timestamp: Optional[pd.Timestamp]
    target_amount: float
    wealth_at_deadline: float
    success: bool
    shortfall: float           # positive if missing amount, else 0
    attainment_ratio: float    # min(wealth/target, 1)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_goal(
    wealth: pd.Series,
    goal: Goal,
    *,
    start: Optional[date] = None,
) -> GoalEvaluation:
    """Evaluate a single goal on a simulated wealth path.

    Parameters
    ----------
    wealth : pd.Series
        Wealth path of length T (from `simulate_capital` or `simulate_portfolio`).
    goal : Goal
        Goal specification (amount and deadline).
    start : Optional[date]
        Start date used when resolving deadlines if wealth has no DatetimeIndex.
    """
    if not isinstance(wealth, pd.Series) or wealth.empty:
        raise ValueError("wealth must be a non-empty pandas Series.")
    T = len(wealth)
    idx_like = wealth.index if isinstance(wealth.index, pd.DatetimeIndex) else None
    pos = goal.resolve_deadline_pos(T, start=start, like=wealth if idx_like is not None else None)
    wT = float(wealth.iloc[pos])
    target = float(goal.target_amount)
    sf = max(0.0, target - wT)
    ratio = 0.0 if target <= 0 else min(wT / target, 1.0)
    ts = wealth.index[pos] if isinstance(wealth.index, pd.DatetimeIndex) else None
    return GoalEvaluation(
        goal_name=goal.name,
        deadline_pos=pos,
        deadline_timestamp=ts,
        target_amount=target,
        wealth_at_deadline=wT,
        success=sf <= 1e-8,
        shortfall=sf,
        attainment_ratio=ratio,
    )


def evaluate_goals(
    wealth: pd.Series,
    goals: Iterable[Goal],
    *,
    start: Optional[date] = None,
) -> pd.DataFrame:
    """Evaluate multiple goals and return a tidy DataFrame.

    Columns: [goal, deadline_pos, deadline_timestamp, target_amount,
              wealth_at_deadline, success, shortfall, attainment_ratio]
    """
    rows = []
    for g in goals:
        ev = evaluate_goal(wealth, g, start=start)
        rows.append({
            "goal": ev.goal_name,
            "deadline_pos": ev.deadline_pos,
            "deadline_timestamp": ev.deadline_timestamp,
            "target_amount": ev.target_amount,
            "wealth_at_deadline": ev.wealth_at_deadline,
            "success": ev.success,
            "shortfall": ev.shortfall,
            "attainment_ratio": ev.attainment_ratio,
        })
    df = pd.DataFrame(rows)
    # Order by priority if available (stable merge)
    try:
        pr = pd.DataFrame([{"goal": g.name, "priority": g.priority} for g in goals])
        df = df.merge(pr, on="goal", how="left").sort_values(["priority", "goal"]).reset_index(drop=True)
    except Exception:
        pass
    return df

# ---------------------------------------------------------------------------
# Manual quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example wealth path (deterministic 0.5% monthly for 24 months)
    T = 24
    idx = month_index(start=date(2025, 9, 1), months=T)
    # Suppose we already simulated wealth; here we mock a simple compounding
    a = 700000.0
    r = np.full(T, 0.005)
    W = np.zeros(T, dtype=float)
    w_prev = 0.0
    for t in range(T):
        w_prev = (w_prev + a) * (1.0 + r[t])
        W[t] = w_prev
    wealth = pd.Series(W, index=idx, name="wealth")

    # Define two goals
    g1 = Goal(name="housing", target_amount=20_000_000.0, target_month_index=23)
    g2 = Goal(name="emergency", target_amount=6_000_000.0, target_month_index=11)

    print(evaluate_goals(wealth, [g1, g2]))

    # Split contributions 60/40 across goals (MVP)
    contrib = pd.Series(np.full(T, a), index=idx, name="contribution")
    split = allocate_contributions_proportional(contrib, {"housing": 0.6, "emergency": 0.4})
    print(split.head())
