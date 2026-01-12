"""
Rewards and withdrawal scheduling module for FinOpt.

Purpose
-------
Models planned purchases (rewards) that require portfolio withdrawals.
Provides scheduling and policy mechanisms for how withdrawals are
distributed across accounts.

Key Mathematical Model
----------------------
Withdrawals modify wealth dynamics:
    W_{t+1}^m = (W_t^m + A_t x_t^m - y_t^m)(1 + R_t^m)

Where y_t^m is the withdrawal from account m at time t.

The extended wealth remains AFFINE in (X, Y):
    W_t^m(X,Y) = W_0^m F_{0,t}^m + sum_{s=0}^{t-1} (A_s x_s^m - y_s^m) F_{s,t}^m

This preserves convexity for CVaR optimization.

Key components
--------------
- Reward:
    Single planned purchase requiring a withdrawal at a specific month.
    Can be optional (optimizer decides) or mandatory.

- RewardSchedule:
    Collection of rewards with a withdrawal policy determining how
    to distribute withdrawals across accounts.

Design principles
-----------------
- Frozen dataclasses for immutability (like goals.py)
- Separation from portfolio execution (rewards define schedule, portfolio executes)
- Support both fixed schedules and optimization-ready parameters
- Withdrawal policies: single_account, proportional, priority

Example
-------
>>> from src.rewards import Reward, RewardSchedule
>>> rewards = [
...     Reward(name="Laptop", amount=1_000_000, month=6),
...     Reward(name="Vacation", amount=2_000_000, month=12, optional=True),
... ]
>>> schedule = RewardSchedule(
...     rewards=rewards,
...     withdrawal_policy="single_account",
...     default_account=0
... )
>>> Y = schedule.get_fixed_withdrawals(T=24, M=2, accounts=accounts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Union, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .portfolio import Account

__all__ = [
    "Reward",
    "RewardSchedule",
]


@dataclass(frozen=True)
class Reward:
    """
    Single planned purchase/reward requiring portfolio withdrawal.

    Purpose
    -------
    Represents a single financial goal that requires withdrawing funds
    from the portfolio at a specific time. Can be mandatory (must occur)
    or optional (optimizer decides based on utility).

    Parameters
    ----------
    name : str
        Description of the purchase (e.g., "Vacation", "New Computer").
    amount : float
        Nominal cost in currency units (e.g., CLP).
        Must be non-negative.
    month : int
        Target month for the purchase (1-indexed offset from start).
        Month 1 = first month of simulation.
    account : Optional[Union[int, str]], default None
        Preferred account to withdraw from.
        - int: account index (0-indexed)
        - str: account name
        - None: uses RewardSchedule's withdrawal_policy
    optional : bool, default False
        If True, optimizer can decide purchase fraction u_k in [0, 1].
        If False, purchase is mandatory and fully funded.
    priority : int, default 1
        Relative importance for optional rewards (higher = more important).
        Used in objective function weighting.

    Notes
    -----
    - Rewards with month <= 0 or month > T are ignored in scheduling
    - Optional rewards contribute to optimizer objective as: priority * u_k * amount
    - Mandatory rewards generate hard constraints: sum_m y_t^m >= amount

    Examples
    --------
    >>> laptop = Reward(name="Laptop", amount=1_000_000, month=6)
    >>> vacation = Reward(name="Vacation", amount=2_000_000, month=12, 
    ...                   optional=True, priority=2)
    """
    name: str
    amount: float
    month: int
    account: Optional[Union[int, str]] = None
    optional: bool = False
    priority: int = 1

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f"amount must be non-negative, got {self.amount}")
        if self.month < 1:
            raise ValueError(f"month must be >= 1, got {self.month}")
        if self.priority < 1:
            raise ValueError(f"priority must be >= 1, got {self.priority}")


@dataclass
class RewardSchedule:
    """
    Collection of planned purchases with withdrawal scheduling.

    Purpose
    -------
    Manages a set of rewards and provides methods for generating
    withdrawal schedules compatible with Portfolio.simulate() and
    CVaROptimizer.solve().

    Parameters
    ----------
    rewards : List[Reward]
        List of planned purchases.
    withdrawal_policy : {"proportional", "priority", "single_account"}, default "proportional"
        How to distribute withdrawals across accounts:
        - "proportional": equal split across all accounts (placeholder for wealth-weighted)
        - "priority": withdraw from lowest-volatility account first
        - "single_account": all from specified account (requires default_account)
    default_account : Optional[Union[int, str]], default None
        Default account for "single_account" policy.
        - int: account index (0-indexed)
        - str: account name

    Methods
    -------
    get_fixed_withdrawals(T, M, accounts, start=None) -> np.ndarray
        Generate fixed withdrawal matrix Y of shape (T, M).
    get_mandatory_rewards() -> List[Reward]
        Return list of mandatory (non-optional) rewards.
    get_optional_rewards() -> List[Reward]
        Return list of optional rewards.
    total_mandatory_amount() -> float
        Sum of all mandatory reward amounts.
    to_optimization_params(T, M, accounts) -> dict
        Return parameters for CVaROptimizer integration.

    Notes
    -----
    - For "proportional" policy with fixed schedule, uses equal split
      (true proportional requires runtime W_t knowledge)
    - Optional rewards are NOT included in get_fixed_withdrawals()
      (handled by optimizer decision variables)

    Examples
    --------
    >>> from src.rewards import Reward, RewardSchedule
    >>> from src.portfolio import Account
    >>> 
    >>> rewards = [
    ...     Reward(name="Laptop", amount=1_000_000, month=3),
    ...     Reward(name="Vacation", amount=2_000_000, month=6),
    ... ]
    >>> schedule = RewardSchedule(rewards, withdrawal_policy="single_account", 
    ...                           default_account=0)
    >>> 
    >>> accounts = [Account.from_annual("Cash", 0.03, 0.02, initial_wealth=5_000_000)]
    >>> Y = schedule.get_fixed_withdrawals(T=12, M=1, accounts=accounts)
    >>> Y[2, 0]  # Laptop at month 3 (0-indexed: 2)
    1000000.0
    """
    rewards: List[Reward]
    withdrawal_policy: Literal["proportional", "priority", "single_account"] = "proportional"
    default_account: Optional[Union[int, str]] = None

    def __post_init__(self) -> None:
        if not self.rewards:
            raise ValueError("rewards list cannot be empty")
        if self.withdrawal_policy == "single_account" and self.default_account is None:
            raise ValueError("single_account policy requires default_account")

    def _resolve_account(
        self,
        account: Optional[Union[int, str]],
        accounts: List["Account"],
    ) -> int:
        """
        Resolve account reference to index.

        Parameters
        ----------
        account : int, str, or None
            Account reference to resolve.
        accounts : List[Account]
            List of account objects for name lookup.

        Returns
        -------
        int
            Account index (0-indexed).

        Raises
        ------
        ValueError
            If account name not found or index out of range.
        """
        if account is None:
            raise ValueError("Cannot resolve None account reference")

        if isinstance(account, int):
            if account < 0 or account >= len(accounts):
                raise ValueError(f"Account index {account} out of range [0, {len(accounts)-1}]")
            return account

        # String lookup
        for i, acc in enumerate(accounts):
            if acc.name == account:
                return i
        raise ValueError(f"Account '{account}' not found in accounts list")

    def get_mandatory_rewards(self) -> List[Reward]:
        """Return list of mandatory (non-optional) rewards."""
        return [r for r in self.rewards if not r.optional]

    def get_optional_rewards(self) -> List[Reward]:
        """Return list of optional rewards."""
        return [r for r in self.rewards if r.optional]

    def total_mandatory_amount(self) -> float:
        """Return sum of all mandatory reward amounts."""
        return sum(r.amount for r in self.get_mandatory_rewards())

    def get_fixed_withdrawals(
        self,
        T: int,
        M: int,
        accounts: List["Account"],
        start: Optional[date] = None,
    ) -> np.ndarray:
        """
        Generate fixed withdrawal matrix for non-optional rewards.

        Parameters
        ----------
        T : int
            Number of time periods (months).
        M : int
            Number of accounts.
        accounts : List[Account]
            Account objects for policy resolution.
        start : date, optional
            Start date (currently unused, reserved for calendar alignment).

        Returns
        -------
        np.ndarray
            Withdrawal matrix Y of shape (T, M).
            Y[t, m] = withdrawal amount from account m at time t.

        Notes
        -----
        - Only mandatory rewards are included
        - Optional rewards return zeros (handled by optimizer)
        - For overlapping rewards at same month, amounts accumulate
        """
        Y = np.zeros((T, M), dtype=float)

        for reward in self.rewards:
            if reward.optional:
                continue  # Optimizer decides

            t = reward.month - 1  # Convert to 0-indexed
            if t < 0 or t >= T:
                continue  # Outside horizon

            # Determine withdrawal account(s)
            if self.withdrawal_policy == "single_account":
                account_ref = reward.account if reward.account is not None else self.default_account
                m = self._resolve_account(account_ref, accounts)
                Y[t, m] += reward.amount

            elif self.withdrawal_policy == "proportional":
                # Equal split (placeholder for wealth-proportional)
                Y[t, :] += reward.amount / M

            elif self.withdrawal_policy == "priority":
                # Withdraw from lowest-volatility account first
                sorted_accounts = sorted(
                    enumerate(accounts),
                    key=lambda x: x[1].return_strategy["sigma"]
                )
                m = sorted_accounts[0][0]
                Y[t, m] += reward.amount

        return Y

    def to_optimization_params(
        self,
        T: int,
        M: int,
        accounts: List["Account"],
    ) -> dict:
        """
        Generate parameters for CVaROptimizer integration.

        Parameters
        ----------
        T : int
            Number of time periods.
        M : int
            Number of accounts.
        accounts : List[Account]
            Account objects.

        Returns
        -------
        dict
            Parameters for optimizer:
            - "Y_fixed": fixed withdrawal matrix (T, M)
            - "mandatory_months": list of (month, amount) for constraints
            - "optional_rewards": list of optional Reward objects
            - "policy": withdrawal policy string
        """
        return {
            "Y_fixed": self.get_fixed_withdrawals(T, M, accounts),
            "mandatory_months": [
                (r.month - 1, r.amount) 
                for r in self.get_mandatory_rewards() 
                if 0 < r.month <= T
            ],
            "optional_rewards": self.get_optional_rewards(),
            "policy": self.withdrawal_policy,
        }

    def validate_liquidity(
        self,
        W: np.ndarray,
        Y: np.ndarray,
    ) -> bool:
        """
        Check if withdrawal schedule is feasible given wealth trajectories.

        Parameters
        ----------
        W : np.ndarray
            Wealth trajectories of shape (n_sims, T+1, M) or (T+1, M).
        Y : np.ndarray
            Withdrawal schedule of shape (T, M).

        Returns
        -------
        bool
            True if Y[t, m] <= W[:, t, m] for all t, m across all simulations.

        Notes
        -----
        - Conservative check: requires feasibility in ALL scenarios
        - For probabilistic check, use optimization with CVaR constraints
        """
        if W.ndim == 2:
            W = W[None, :, :]  # Add sim dimension

        T, M = Y.shape
        for t in range(T):
            for m in range(M):
                if (W[:, t, m] < Y[t, m]).any():
                    return False
        return True

    def summary(self) -> str:
        """Return human-readable summary of reward schedule."""
        lines = [f"RewardSchedule ({len(self.rewards)} rewards, policy={self.withdrawal_policy})"]
        
        mandatory = self.get_mandatory_rewards()
        optional = self.get_optional_rewards()
        
        if mandatory:
            lines.append(f"  Mandatory ({len(mandatory)}):")
            for r in mandatory:
                lines.append(f"    - {r.name}: ${r.amount:,.0f} @ month {r.month}")
        
        if optional:
            lines.append(f"  Optional ({len(optional)}):")
            for r in optional:
                lines.append(f"    - {r.name}: ${r.amount:,.0f} @ month {r.month} (priority={r.priority})")
        
        lines.append(f"  Total mandatory: ${self.total_mandatory_amount():,.0f}")
        
        return "\n".join(lines)
