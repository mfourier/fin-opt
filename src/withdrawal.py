"""
Withdrawal modeling module for FinOpt.

Purpose
-------
Models scheduled withdrawals (retiros) from investment accounts. Captures planned
cash outflows such as purchases, emergency expenses, or periodic distributions.
Produces clean, monthly arrays that downstream modules (portfolio, optimization)
consume to adjust wealth trajectories.

Mathematical Framework
----------------------
The wealth dynamics equation with withdrawals:

    W_{t+1}^m = (W_t^m + A_t·x_t^m - D_t^m)(1 + R_t^m)

where D_t^m is the withdrawal from account m in month t. The withdrawal occurs
at the START of the month (before returns are applied), meaning the withdrawn
amount does not earn returns that month.

Affine representation (critical for convex optimization):

    W_t^m = W_0^m·F_{0,t}^m + Σ_{s=0}^{t-1} (A_s·x_s^m - D_s^m)·F_{s,t}^m

Key insight: D is a PARAMETER (not a decision variable), so the representation
remains affine in X, preserving convexity for CVaR optimization.

Key components
--------------
- WithdrawalEvent:
    Single scheduled withdrawal at a specific date from a specific account.
    Immutable specification with amount, date, and optional description.

- WithdrawalSchedule:
    Collection of withdrawal events. Converts calendar dates to month offsets
    and produces arrays suitable for portfolio simulation.

Design principles
-----------------
- Immutability: WithdrawalEvent is a frozen dataclass (like Goal objects)
- Calendar-aware: Dates are resolved to month offsets relative to start_date
- Pattern match: Follows FixedIncome.salary_raises date-to-month conversion
- Backward compatible: D=None in portfolio.simulate() preserves existing behavior
- Validation: Warns about withdrawals exceeding available wealth

Example
-------
>>> from datetime import date
>>> from finopt.src.withdrawal import WithdrawalEvent, WithdrawalSchedule
>>> from finopt.src.portfolio import Account
>>>
>>> # Define withdrawals
>>> withdrawals = WithdrawalSchedule(events=[
...     WithdrawalEvent(
...         account="Conservador",
...         amount=400_000,
...         date=date(2025, 6, 1),
...         description="Compra bicicleta"
...     ),
...     WithdrawalEvent(
...         account="Agresivo",
...         amount=2_000_000,
...         date=date(2026, 12, 1),
...         description="Vacaciones"
...     )
... ])
>>>
>>> # Convert to array for simulation
>>> accounts = [
...     Account.from_annual("Conservador", 0.06, 0.08),
...     Account.from_annual("Agresivo", 0.12, 0.15)
... ]
>>> D = withdrawals.to_array(T=36, start_date=date(2025, 1, 1), accounts=accounts)
>>> D.shape
(36, 2)
>>> D[5, 0]  # 400k withdrawal from account 0 in month 5 (June 2025)
400000.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    from .portfolio import Account

__all__ = [
    "WithdrawalEvent",
    "WithdrawalSchedule",
]


# ---------------------------------------------------------------------------
# Withdrawal Event (Single Scheduled Withdrawal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WithdrawalEvent:
    """
    Single scheduled withdrawal from an investment account.

    Represents a cash outflow at a specific calendar date:
        D_t^m = amount  if t corresponds to date, else 0

    The withdrawal occurs at the START of the month, before returns are applied.
    This is more conservative (withdrawn amount doesn't earn returns).

    Parameters
    ----------
    account : int or str
        Target account identifier (index or name).
    amount : float
        Withdrawal amount (must be positive).
    date : datetime.date
        Calendar date of the withdrawal.
    description : str, optional
        Human-readable description (e.g., "Compra bicicleta").

    Notes
    -----
    - Immutable specification (frozen dataclass)
    - Date resolution: converted to month offset via _date_to_month_offset()
    - Amount validation: must be positive
    - Account resolution: validated during WithdrawalSchedule.to_array()

    Examples
    --------
    >>> event = WithdrawalEvent(
    ...     account="Conservador",
    ...     amount=400_000,
    ...     date=date(2025, 6, 1),
    ...     description="Compra bicicleta"
    ... )
    >>> event.amount
    400000
    """
    account: Union[int, str]
    amount: float
    date: date
    description: str = ""

    def __post_init__(self):
        """Validate withdrawal event parameters."""
        if self.amount <= 0:
            raise ValueError(f"amount must be positive, got {self.amount}")

    def resolve_month(self, start_date: date) -> int:
        """
        Convert withdrawal date to month offset from start_date.

        Parameters
        ----------
        start_date : datetime.date
            Simulation start date (month 0).

        Returns
        -------
        int
            Month offset (0-indexed). Can be negative if date < start_date.

        Examples
        --------
        >>> event = WithdrawalEvent("Account", 100_000, date(2025, 6, 1))
        >>> event.resolve_month(date(2025, 1, 1))
        5
        """
        year_diff = self.date.year - start_date.year
        month_diff = self.date.month - start_date.month
        return year_diff * 12 + month_diff

    def __repr__(self) -> str:
        """Readable representation."""
        desc = f", {self.description!r}" if self.description else ""
        return (
            f"WithdrawalEvent(account={self.account!r}, "
            f"amount={self.amount:,.0f}, date={self.date.isoformat()}{desc})"
        )


# ---------------------------------------------------------------------------
# Withdrawal Schedule (Collection)
# ---------------------------------------------------------------------------

@dataclass
class WithdrawalSchedule:
    """
    Collection of scheduled withdrawals for portfolio simulation.

    Converts calendar-based withdrawal events to numpy arrays suitable for
    the portfolio wealth dynamics:

        W_{t+1}^m = (W_t^m + A_t·x_t^m - D_t^m)(1 + R_t^m)

    Parameters
    ----------
    events : List[WithdrawalEvent]
        List of withdrawal events to schedule.

    Methods
    -------
    to_array(T, start_date, accounts) -> np.ndarray
        Convert events to (T, M) array for simulation.
    total_by_account(accounts) -> Dict[str, float]
        Sum of withdrawals per account.
    get_events_for_account(account) -> List[WithdrawalEvent]
        Filter events by account.

    Notes
    -----
    - Events with dates outside [start_date, start_date + T months) are ignored
      with a warning (same behavior as FixedIncome.salary_raises)
    - Multiple events on the same month/account are summed
    - Empty events list is valid (D = 0 everywhere)

    Examples
    --------
    >>> schedule = WithdrawalSchedule(events=[
    ...     WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
    ...     WithdrawalEvent("Agresivo", 2_000_000, date(2026, 12, 1))
    ... ])
    >>>
    >>> D = schedule.to_array(T=36, start_date=date(2025, 1, 1), accounts=accounts)
    >>> D.shape
    (36, 2)
    """
    events: List[WithdrawalEvent] = field(default_factory=list)

    def to_array(
        self,
        T: int,
        start_date: date,
        accounts: List[Account]
    ) -> np.ndarray:
        """
        Convert withdrawal events to array for portfolio simulation.

        Parameters
        ----------
        T : int
            Simulation horizon in months.
        start_date : datetime.date
            Simulation start date (month 0).
        accounts : List[Account]
            Portfolio accounts for name-to-index resolution.

        Returns
        -------
        np.ndarray, shape (T, M)
            Withdrawal array where D[t, m] = total withdrawal from account m
            in month t. Zero for months/accounts without withdrawals.

        Warns
        -----
        UserWarning
            If any withdrawal date falls outside the simulation horizon.

        Examples
        --------
        >>> D = schedule.to_array(T=36, start_date=date(2025, 1, 1), accounts=accounts)
        >>> D[5, 0]  # Withdrawal in month 5 from account 0
        400000.0
        """
        M = len(accounts)
        D = np.zeros((T, M), dtype=float)

        # Build account name-to-index mapping
        account_names = [acc.name for acc in accounts]

        for event in self.events:
            # Resolve account index
            if isinstance(event.account, int):
                if not (0 <= event.account < M):
                    raise ValueError(
                        f"Account index {event.account} out of range [0, {M}). "
                        f"Available accounts: {account_names}"
                    )
                acc_idx = event.account
            else:
                try:
                    acc_idx = account_names.index(event.account)
                except ValueError:
                    raise ValueError(
                        f"Account name {event.account!r} not found. "
                        f"Available accounts: {account_names}"
                    ) from None

            # Resolve month offset
            month = event.resolve_month(start_date)

            # Check if within simulation horizon
            if month < 0:
                warnings.warn(
                    f"Withdrawal {event.description or 'event'} at {event.date} "
                    f"is before simulation start ({start_date}). Ignoring.",
                    UserWarning
                )
                continue

            if month >= T:
                warnings.warn(
                    f"Withdrawal {event.description or 'event'} at {event.date} "
                    f"(month {month}) is beyond horizon T={T}. Ignoring.",
                    UserWarning
                )
                continue

            # Add to array (supports multiple events on same month/account)
            D[month, acc_idx] += event.amount

        return D

    def total_by_account(
        self,
        accounts: List[Account]
    ) -> Dict[str, float]:
        """
        Compute total withdrawals per account.

        Parameters
        ----------
        accounts : List[Account]
            Portfolio accounts for name resolution.

        Returns
        -------
        Dict[str, float]
            Mapping from account name to total withdrawal amount.

        Examples
        --------
        >>> schedule.total_by_account(accounts)
        {'Conservador': 400000.0, 'Agresivo': 2000000.0}
        """
        totals: Dict[str, float] = {acc.name: 0.0 for acc in accounts}
        account_names = [acc.name for acc in accounts]

        for event in self.events:
            # Resolve account name
            if isinstance(event.account, int):
                if 0 <= event.account < len(accounts):
                    name = account_names[event.account]
                else:
                    continue  # Invalid index, skip
            else:
                name = event.account
                if name not in totals:
                    continue  # Unknown account, skip

            totals[name] += event.amount

        return totals

    def get_events_for_account(
        self,
        account: Union[int, str],
        accounts: Optional[List[Account]] = None
    ) -> List[WithdrawalEvent]:
        """
        Filter events by account.

        Parameters
        ----------
        account : int or str
            Account identifier to filter by.
        accounts : List[Account], optional
            Required if account is int (for name resolution).

        Returns
        -------
        List[WithdrawalEvent]
            Events targeting the specified account.
        """
        # Normalize to string if accounts provided
        if isinstance(account, int) and accounts is not None:
            target_name = accounts[account].name
        else:
            target_name = account

        return [
            e for e in self.events
            if e.account == target_name or e.account == account
        ]

    def __len__(self) -> int:
        """Number of withdrawal events."""
        return len(self.events)

    def __repr__(self) -> str:
        """Readable representation."""
        if not self.events:
            return "WithdrawalSchedule(empty)"

        total = sum(e.amount for e in self.events)
        return (
            f"WithdrawalSchedule(n_events={len(self.events)}, "
            f"total={total:,.0f})"
        )

    # -------------------------- Serialization helpers ----------------------
    def to_dict(self) -> dict:
        """
        Serialize WithdrawalSchedule to dictionary.

        Returns
        -------
        dict
            Dictionary with 'events' key containing list of event dicts.
        """
        return {
            "events": [
                {
                    "account": e.account,
                    "amount": e.amount,
                    "date": e.date.isoformat(),
                    "description": e.description
                }
                for e in self.events
            ]
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WithdrawalSchedule":
        """
        Deserialize WithdrawalSchedule from dictionary.

        Parameters
        ----------
        payload : dict
            Dictionary with 'events' key.

        Returns
        -------
        WithdrawalSchedule
            Reconstructed schedule.
        """
        events = []
        for e in payload.get("events", []):
            events.append(WithdrawalEvent(
                account=e["account"],
                amount=float(e["amount"]),
                date=date.fromisoformat(e["date"]),
                description=e.get("description", "")
            ))
        return cls(events=events)
