"""
Demonstration of dual metric reporting for CVaR transparency.

This script demonstrates how the new dual metric reporting works in FinOpt,
showing both the specified confidence (CVaR guarantee) and the empirical
probability (actual observed success rate).

Author: FinOpt Development Team
Date: 2025-05-01
"""

import numpy as np
from datetime import date

from finopt.portfolio import Account
from finopt.goals import TerminalGoal, IntermediateGoal, check_goals, print_goal_status
from finopt.model import FinancialModel, SimulationResult
from finopt.income import IncomeModel, FixedIncome


def demo_basic_dual_metrics():
    """Demonstrate basic dual metric computation."""
    print("=" * 80)
    print("DEMO 1: Basic Dual Metrics")
    print("=" * 80)
    print()

    # Setup simple scenario
    accounts = [
        Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09)
    ]

    # Create a simple simulation result
    n_sims = 100
    T = 12
    M = 1

    # Create wealth trajectories where 85% of scenarios exceed 5M threshold
    wealth = np.zeros((n_sims, T + 1, M))
    wealth[:85, -1, 0] = 6_000_000  # 85 scenarios above threshold
    wealth[85:, -1, 0] = 4_000_000  # 15 scenarios below threshold

    result = SimulationResult(
        wealth=wealth,
        total_wealth=wealth.sum(axis=2),
        contributions=np.ones((n_sims, T)),
        returns=np.zeros((n_sims, T, M)),
        income={
            "fixed": np.ones((n_sims, T)),
            "variable": np.zeros((n_sims, T)),
            "total": np.ones((n_sims, T))
        },
        allocation=np.array([[1.0]] * T),
        withdrawals=None,
        T=T,
        n_sims=n_sims,
        M=M,
        start=date(2025, 1, 1),
        seed=42,
        account_names=["Conservative"]
    )

    # Goal with 80% confidence
    goals = [
        TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.80)
    ]

    # Check goals with new dual metrics
    status = check_goals(result, goals, accounts, date(2025, 1, 1))

    print("Goal specification:")
    print(f"  Target: ${goals[0].threshold:,.0f}")
    print(f"  Specified confidence: {goals[0].confidence:.1%}")
    print()

    print("Results:")
    metrics = status[goals[0]]
    print(f"  Violation rate: {metrics['violation_rate']:.1%}")
    print(f"  Empirical probability: {metrics['empirical_probability']:.1%}")
    print(f"  Confidence gap: {metrics['confidence_gap']:+.1%}")
    print(f"  Satisfied: {metrics['satisfied']}")
    print()

    print("Interpretation:")
    print(f"  {metrics['note']}")
    print()


def demo_significant_conservatism():
    """Demonstrate significant CVaR conservatism (gap > 1%)."""
    print("=" * 80)
    print("DEMO 2: Significant CVaR Conservatism")
    print("=" * 80)
    print()

    accounts = [
        Account.from_annual("Aggressive", annual_return=0.14, annual_volatility=0.15)
    ]

    # Create scenario with high conservatism
    n_sims = 100
    T = 24
    M = 1

    # 95% success rate vs 85% specified → 10% gap
    wealth = np.zeros((n_sims, T + 1, M))
    wealth[:95, -1, 0] = 12_000_000  # 95 scenarios above threshold
    wealth[95:, -1, 0] = 8_000_000   # 5 scenarios below threshold

    result = SimulationResult(
        wealth=wealth,
        total_wealth=wealth.sum(axis=2),
        contributions=np.ones((n_sims, T)),
        returns=np.zeros((n_sims, T, M)),
        income={
            "fixed": np.ones((n_sims, T)),
            "variable": np.zeros((n_sims, T)),
            "total": np.ones((n_sims, T))
        },
        allocation=np.array([[1.0]] * T),
        withdrawals=None,
        T=T,
        n_sims=n_sims,
        M=M,
        start=date(2025, 1, 1),
        seed=42,
        account_names=["Aggressive"]
    )

    goals = [
        TerminalGoal(account="Aggressive", threshold=10_000_000, confidence=0.85)
    ]

    # Use print_goal_status for formatted output
    print("Using print_goal_status() for formatted display:")
    print_goal_status(result, goals, accounts, date(2025, 1, 1))


def demo_multiple_goals():
    """Demonstrate dual metrics with multiple goals."""
    print("=" * 80)
    print("DEMO 3: Multiple Goals with Different Conservatism Levels")
    print("=" * 80)
    print()

    accounts = [
        Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
        Account.from_annual("Growth", annual_return=0.12, annual_volatility=0.18)
    ]

    n_sims = 100
    T = 24
    M = 2

    # Create varied scenarios
    wealth = np.zeros((n_sims, T + 1, M))

    # Emergency fund (account 0): 92% success vs 90% specified (mild conservatism)
    wealth[:92, 6, 0] = 3_000_000
    wealth[92:, 6, 0] = 1_500_000

    # Growth account (account 1): 88% success vs 85% specified (mild conservatism)
    wealth[:88, -1, 1] = 8_000_000
    wealth[88:, -1, 1] = 4_000_000

    result = SimulationResult(
        wealth=wealth,
        total_wealth=wealth.sum(axis=2),
        contributions=np.ones((n_sims, T)),
        returns=np.zeros((n_sims, T, M)),
        income={
            "fixed": np.ones((n_sims, T)),
            "variable": np.zeros((n_sims, T)),
            "total": np.ones((n_sims, T))
        },
        allocation=np.array([[0.5, 0.5]] * T),
        withdrawals=None,
        T=T,
        n_sims=n_sims,
        M=M,
        start=date(2025, 1, 1),
        seed=42,
        account_names=["Emergency", "Growth"]
    )

    goals = [
        IntermediateGoal(
            date=date(2025, 7, 1),
            account="Emergency",
            threshold=2_000_000,
            confidence=0.90
        ),
        TerminalGoal(
            account="Growth",
            threshold=6_000_000,
            confidence=0.85
        )
    ]

    print_goal_status(result, goals, accounts, date(2025, 1, 1))


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 80)
    print("FinOpt CVaR Transparency: Dual Metric Reporting")
    print("*" * 80)
    print()
    print("This demonstration shows how FinOpt reports both:")
    print("  1. Specified confidence (CVaR theoretical guarantee)")
    print("  2. Empirical probability (actual observed success rate)")
    print()
    print("The 'confidence gap' measures CVaR conservatism - the difference")
    print("between what we observe empirically and what we specified.")
    print()

    demo_basic_dual_metrics()
    demo_significant_conservatism()
    demo_multiple_goals()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  • CVaR reformulation guarantees AT LEAST the specified confidence")
    print("  • Empirical probability typically exceeds specified confidence")
    print("  • Confidence gap quantifies this conservatism")
    print("  • Both metrics are reported for intellectual transparency")
    print()
    print("This dual reporting maintains CVaR's computational efficiency while")
    print("being honest about the actual risk levels achieved.")
    print()


if __name__ == "__main__":
    main()
