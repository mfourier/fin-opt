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
    "StochasticWithdrawal",
    "WithdrawalModel",
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


# ---------------------------------------------------------------------------
# Stochastic Withdrawal (Withdrawal with Uncertainty)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StochasticWithdrawal:
    """
    Withdrawal with variability/uncertainty.

    Models withdrawals that have a base expected amount but may vary across
    scenarios due to uncertainty (e.g., variable medical expenses, emergency costs).
    Generates samples from a truncated Gaussian distribution.

    Parameters
    ----------
    account : int or str
        Target account identifier (index or name).
    base_amount : float
        Expected withdrawal amount (mean of distribution).
    sigma : float
        Standard deviation of withdrawal amount.
    month : int, optional
        Month offset from start_date (0-indexed). Mutually exclusive with date.
    date : datetime.date, optional
        Calendar date of withdrawal. Mutually exclusive with month.
    floor : float, default=0.0
        Minimum withdrawal amount (truncation lower bound).
    cap : float, optional
        Maximum withdrawal amount (truncation upper bound). None = no cap.
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    - Samples from N(base_amount, sigma²) truncated to [floor, cap]
    - Only one of month/date should be specified
    - Follows VariableIncome pattern for stochastic sampling
    - Immutable specification (frozen dataclass)

    Examples
    --------
    >>> withdrawal = StochasticWithdrawal(
    ...     account="Conservador",
    ...     base_amount=300_000,
    ...     sigma=50_000,
    ...     date=date(2025, 9, 1),
    ...     floor=200_000,
    ...     cap=500_000
    ... )
    >>> samples = withdrawal.sample(n_sims=1000, start_date=date(2025, 1, 1))
    >>> samples.shape
    (1000,)
    >>> (samples >= 200_000).all() and (samples <= 500_000).all()
    True
    """
    account: Union[int, str]
    base_amount: float
    sigma: float
    month: Optional[int] = None
    date: Optional[date] = None
    floor: float = 0.0
    cap: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate stochastic withdrawal parameters."""
        # Mutually exclusive month/date
        if self.month is not None and self.date is not None:
            raise ValueError("Specify either month or date, not both")
        if self.month is None and self.date is None:
            raise ValueError("Must specify either month or date")

        # Validate amounts
        if self.base_amount < 0:
            raise ValueError(f"base_amount must be non-negative, got {self.base_amount}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.floor < 0:
            raise ValueError(f"floor must be non-negative, got {self.floor}")
        if self.cap is not None and self.cap < self.floor:
            raise ValueError(f"cap ({self.cap}) must be >= floor ({self.floor})")

        # Check if month is valid
        if self.month is not None and self.month < 0:
            raise ValueError(f"month must be non-negative, got {self.month}")

    def resolve_month(self, start_date: date) -> int:
        """
        Resolve withdrawal timing to month offset.

        Parameters
        ----------
        start_date : datetime.date
            Simulation start date.

        Returns
        -------
        int
            Month offset (0-indexed).
        """
        if self.month is not None:
            return self.month
        else:
            # Use same logic as WithdrawalEvent
            year_diff = self.date.year - start_date.year
            month_diff = self.date.month - start_date.month
            return year_diff * 12 + month_diff

    def sample(self, n_sims: int, start_date: Optional[date] = None) -> np.ndarray:
        """
        Generate random samples of withdrawal amount.

        Parameters
        ----------
        n_sims : int
            Number of scenarios to generate.
        start_date : datetime.date, optional
            Required if self.date is specified (for month resolution).

        Returns
        -------
        np.ndarray, shape (n_sims,)
            Sampled withdrawal amounts, truncated to [floor, cap].

        Examples
        --------
        >>> withdrawal = StochasticWithdrawal(
        ...     account="Conservador",
        ...     base_amount=300_000,
        ...     sigma=50_000,
        ...     month=6,
        ...     floor=200_000,
        ...     cap=500_000,
        ...     seed=42
        ... )
        >>> samples = withdrawal.sample(n_sims=5)
        >>> samples.shape
        (5,)
        """
        # Set random seed if provided
        rng = np.random.default_rng(self.seed)

        # Sample from Gaussian
        samples = rng.normal(loc=self.base_amount, scale=self.sigma, size=n_sims)

        # Apply floor
        samples = np.maximum(samples, self.floor)

        # Apply cap
        if self.cap is not None:
            samples = np.minimum(samples, self.cap)

        return samples

    def __repr__(self) -> str:
        """Readable representation."""
        timing = f"month={self.month}" if self.month is not None else f"date={self.date.isoformat()}"
        return (
            f"StochasticWithdrawal(account={self.account!r}, "
            f"base={self.base_amount:,.0f}, sigma={self.sigma:,.0f}, {timing})"
        )


# ---------------------------------------------------------------------------
# Withdrawal Model (Facade combining deterministic + stochastic)
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class WithdrawalModel:
    """
    Unified withdrawal model combining scheduled and stochastic withdrawals.

    Facade that orchestrates deterministic (WithdrawalSchedule) and stochastic
    (StochasticWithdrawal) withdrawals into a single (n_sims, T, M) array for
    portfolio simulation.

    Parameters
    ----------
    scheduled : WithdrawalSchedule, optional
        Fixed scheduled withdrawals (broadcast to all scenarios).
    stochastic : List[StochasticWithdrawal], optional
        Withdrawals with uncertainty (sampled per scenario).

    Methods
    -------
    to_array(T, start_date, accounts, n_sims, seed) -> np.ndarray
        Generate combined withdrawal array (n_sims, T, M).
    total_expected(accounts) -> Dict[str, float]
        Expected total withdrawal per account.

    Notes
    -----
    - Scheduled withdrawals: same across all scenarios
    - Stochastic withdrawals: independent sampling per scenario
    - Empty model (both None) returns zeros (no withdrawals)
    - Follows IncomeModel pattern (facade combining FixedIncome + VariableIncome)

    Examples
    --------
    >>> from datetime import date
    >>> withdrawals = WithdrawalModel(
    ...     scheduled=WithdrawalSchedule(events=[
    ...         WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
    ...     ]),
    ...     stochastic=[
    ...         StochasticWithdrawal(
    ...             account="Conservador",
    ...             base_amount=300_000,
    ...             sigma=50_000,
    ...             date=date(2025, 9, 1),
    ...             seed=42
    ...         )
    ...     ]
    ... )
    >>> D = withdrawals.to_array(
    ...     T=36,
    ...     start_date=date(2025, 1, 1),
    ...     accounts=accounts,
    ...     n_sims=100,
    ...     seed=42
    ... )
    >>> D.shape
    (100, 36, 2)
    >>> D[:, 5, 0].std()  # Month 5 (June): deterministic, std=0
    0.0
    >>> D[:, 8, 0].std()  # Month 8 (Sep): stochastic, std>0
    47832.5
    """
    scheduled: Optional[WithdrawalSchedule] = None
    stochastic: Optional[List[StochasticWithdrawal]] = None

    def to_array(
        self,
        T: int,
        start_date: date,
        accounts: List[Account],
        n_sims: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate withdrawal array combining scheduled and stochastic withdrawals.

        Parameters
        ----------
        T : int
            Simulation horizon in months.
        start_date : datetime.date
            Simulation start date.
        accounts : List[Account]
            Portfolio accounts for name-to-index resolution.
        n_sims : int, default=1
            Number of Monte Carlo scenarios.
        seed : int, optional
            Random seed for stochastic withdrawals. Overrides individual seeds.

        Returns
        -------
        np.ndarray, shape (n_sims, T, M)
            Withdrawal array where D[i, t, m] = withdrawal from account m
            in month t for scenario i.

        Warns
        -----
        UserWarning
            If any withdrawal date falls outside the simulation horizon.

        Examples
        --------
        >>> D = model.to_array(T=36, start_date=date(2025, 1, 1),
        ...                    accounts=accounts, n_sims=500, seed=42)
        >>> D.shape
        (500, 36, 2)
        """
        M = len(accounts)
        D = np.zeros((n_sims, T, M), dtype=float)

        # Add scheduled withdrawals (broadcast to all scenarios)
        if self.scheduled is not None:
            D_scheduled = self.scheduled.to_array(T, start_date, accounts)  # (T, M)
            D += D_scheduled[np.newaxis, :, :]  # Broadcast to (n_sims, T, M)

        # Add stochastic withdrawals (sample per scenario)
        if self.stochastic is not None:
            account_names = [acc.name for acc in accounts]

            for withdrawal in self.stochastic:
                # Resolve account index
                if isinstance(withdrawal.account, int):
                    if not (0 <= withdrawal.account < M):
                        raise ValueError(
                            f"Account index {withdrawal.account} out of range [0, {M})"
                        )
                    acc_idx = withdrawal.account
                else:
                    try:
                        acc_idx = account_names.index(withdrawal.account)
                    except ValueError:
                        raise ValueError(
                            f"Account name {withdrawal.account!r} not found. "
                            f"Available accounts: {account_names}"
                        ) from None

                # Resolve month
                month = withdrawal.resolve_month(start_date)

                # Check if within horizon
                if month < 0:
                    warnings.warn(
                        f"Stochastic withdrawal at month {month} is before start. Ignoring.",
                        UserWarning
                    )
                    continue

                if month >= T:
                    warnings.warn(
                        f"Stochastic withdrawal at month {month} is beyond horizon T={T}. Ignoring.",
                        UserWarning
                    )
                    continue

                # Sample withdrawals for all scenarios
                # Use global seed if provided, otherwise use withdrawal's seed
                effective_seed = seed if seed is not None else withdrawal.seed
                if effective_seed is not None:
                    # Derive unique seed per withdrawal to avoid correlation
                    effective_seed = effective_seed + hash((withdrawal.account, month)) % 10000

                # Create a new StochasticWithdrawal with the effective seed
                temp_withdrawal = StochasticWithdrawal(
                    account=withdrawal.account,
                    base_amount=withdrawal.base_amount,
                    sigma=withdrawal.sigma,
                    month=month,
                    floor=withdrawal.floor,
                    cap=withdrawal.cap,
                    seed=effective_seed
                )
                samples = temp_withdrawal.sample(n_sims, start_date)

                # Add to array
                D[:, month, acc_idx] += samples

        return D

    def total_expected(self, accounts: List[Account]) -> Dict[str, float]:
        """
        Compute expected total withdrawals per account.

        Parameters
        ----------
        accounts : List[Account]
            Portfolio accounts for name resolution.

        Returns
        -------
        Dict[str, float]
            Mapping from account name to expected total withdrawal amount.
            For stochastic withdrawals, uses base_amount as expectation.

        Examples
        --------
        >>> model.total_expected(accounts)
        {'Conservador': 700000.0, 'Agresivo': 2000000.0}
        """
        totals: Dict[str, float] = {acc.name: 0.0 for acc in accounts}
        account_names = [acc.name for acc in accounts]

        # Add scheduled withdrawals
        if self.scheduled is not None:
            scheduled_totals = self.scheduled.total_by_account(accounts)
            for name, amount in scheduled_totals.items():
                totals[name] += amount

        # Add expected stochastic withdrawals
        if self.stochastic is not None:
            for withdrawal in self.stochastic:
                # Resolve account name
                if isinstance(withdrawal.account, int):
                    if 0 <= withdrawal.account < len(accounts):
                        name = account_names[withdrawal.account]
                    else:
                        continue  # Invalid index
                else:
                    name = withdrawal.account
                    if name not in totals:
                        continue  # Unknown account

                totals[name] += withdrawal.base_amount

        return totals

    def __repr__(self) -> str:
        """Readable representation."""
        parts = []
        if self.scheduled is not None:
            n_scheduled = len(self.scheduled)
            parts.append(f"scheduled={n_scheduled}")
        if self.stochastic is not None:
            parts.append(f"stochastic={len(self.stochastic)}")

        if not parts:
            return "WithdrawalModel(empty)"

        return f"WithdrawalModel({', '.join(parts)})"

    # -------------------------- Serialization helpers ----------------------
    def to_dict(self) -> dict:
        """
        Serialize WithdrawalModel to dictionary.

        Returns
        -------
        dict
            Dictionary with 'scheduled' and 'stochastic' keys.
        """
        result = {}

        if self.scheduled is not None:
            result["scheduled"] = self.scheduled.to_dict()

        if self.stochastic is not None:
            result["stochastic"] = [
                {
                    "account": w.account,
                    "base_amount": w.base_amount,
                    "sigma": w.sigma,
                    "month": w.month,
                    "date": w.date.isoformat() if w.date is not None else None,
                    "floor": w.floor,
                    "cap": w.cap,
                    "seed": w.seed
                }
                for w in self.stochastic
            ]

        return result

    @classmethod
    def from_dict(cls, payload: dict) -> "WithdrawalModel":
        """
        Deserialize WithdrawalModel from dictionary.

        Parameters
        ----------
        payload : dict
            Dictionary with 'scheduled' and/or 'stochastic' keys.

        Returns
        -------
        WithdrawalModel
            Reconstructed model.
        """
        scheduled = None
        if "scheduled" in payload:
            scheduled = WithdrawalSchedule.from_dict(payload["scheduled"])

        stochastic = None
        if "stochastic" in payload:
            stochastic = []
            for w in payload["stochastic"]:
                stochastic.append(StochasticWithdrawal(
                    account=w["account"],
                    base_amount=float(w["base_amount"]),
                    sigma=float(w["sigma"]),
                    month=w.get("month"),
                    date=date.fromisoformat(w["date"]) if w.get("date") else None,
                    floor=float(w.get("floor", 0.0)),
                    cap=float(w["cap"]) if w.get("cap") is not None else None,
                    seed=w.get("seed")
                ))

        return cls(scheduled=scheduled, stochastic=stochastic)
