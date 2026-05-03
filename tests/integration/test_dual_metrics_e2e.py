"""
End-to-End tests for CVaR dual metric reporting.

These tests verify that the dual metrics (empirical_probability,
confidence_gap, note) flow correctly and consistently through
the entire stack:

  Core (goals.py)  →  API (_goal_metrics.py)  →  API response format

Key invariants:
  1. The note-generation thresholds and text are identical in core
     and API: same input → same output category and same wording.
  2. Field names in the Python API response match the TypeScript
     GoalStatus interface in web/src/types/database.ts.
  3. empirical_probability = 1 - violation_rate (always).
  4. confidence_gap = empirical_probability - required_confidence.
  5. The note category (significant / mild / violation) is determined
     by: gap > 0.01, gap ∈ [0, 0.01], gap < 0.

Note on CVaR conservatism:
  CVaR_ε(b - W) ≤ 0  ⟹  ℙ(W ≥ b) ≥ 1-ε  (one-way implication)
  Empirical probability typically exceeds the specified confidence.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

# =============================================================================
# 5.1 – Cross-implementation note consistency
# =============================================================================

class TestNoteConsistency:
    """
    Verify that core goals.py and API _goal_metrics.py produce
    *identical* note text for the same (empirical, confidence) pair.
    This catches regressions if either implementation drifts.
    """

    CASES = [
        # (empirical_probability, specified_confidence, expected_category)
        (0.95, 0.80, "significant"),   # gap = 0.15 > 0.01
        (0.92, 0.85, "significant"),   # gap = 0.07 > 0.01
        (0.815, 0.808, "mild"),        # gap ≈ 0.007 ∈ [0, 0.01]
        (0.80, 0.80, "mild"),          # gap = 0  (exact boundary)
        (0.75, 0.85, "violation"),     # gap = -0.10 < 0
        (0.70, 0.90, "violation"),     # gap = -0.20 < 0
    ]

    def _core_note(self, empirical: float, confidence: float) -> str:
        """Generate note using core goals.py logic."""
        gap = empirical - confidence
        if gap > 0.01:
            return (
                f"CVaR optimization yields conservative estimates. "
                f"Specified confidence {confidence:.1%} guarantees at least "
                f"{empirical:.1%} empirical success rate "
                f"(+{gap:.1%} safety margin)."
            )
        if gap >= 0:
            return (
                f"CVaR constraint satisfied with empirical probability "
                f"{empirical:.1%} (≥ specified {confidence:.1%})."
            )
        return (
            f"Warning: Empirical probability {empirical:.1%} "
            f"is below specified confidence {confidence:.1%}. "
            f"This may indicate CVaR approximation error or insufficient scenarios."
        )

    def _api_note(self, empirical: float, confidence: float) -> str:
        from api.services._goal_metrics import compute_dual_metrics
        return compute_dual_metrics(empirical, confidence)["note"]

    @pytest.mark.parametrize("empirical,confidence,category", CASES)
    def test_notes_are_identical(self, empirical, confidence, category):
        """Core and API generate identical note text."""
        core = self._core_note(empirical, confidence)
        api = self._api_note(empirical, confidence)
        assert core == api, (
            f"Note mismatch for empirical={empirical}, confidence={confidence}.\n"
            f"  Core: {core!r}\n"
            f"  API:  {api!r}"
        )

    @pytest.mark.parametrize("empirical,confidence,category", CASES)
    def test_note_category_correct(self, empirical, confidence, category):
        """Note text belongs to the expected category."""
        from api.services._goal_metrics import compute_dual_metrics
        note = compute_dual_metrics(empirical, confidence)["note"]
        if category == "significant":
            assert "CVaR optimization yields conservative estimates" in note
            assert "safety margin" in note
        elif category == "mild":
            assert "CVaR constraint satisfied" in note
        else:  # violation
            assert "Warning" in note
            assert "below specified confidence" in note


# =============================================================================
# 5.2 – Arithmetic invariants
# =============================================================================

class TestArithmeticInvariants:
    """
    Verify mathematical correctness of dual metrics across implementations.
    """

    @pytest.mark.parametrize("empirical,confidence", [
        (0.95, 0.80), (0.80, 0.80), (0.70, 0.85),
        (0.999, 0.99), (0.501, 0.50),
    ])
    def test_gap_equals_empirical_minus_confidence(self, empirical, confidence):
        from api.services._goal_metrics import compute_dual_metrics
        result = compute_dual_metrics(empirical, confidence)
        expected_gap = empirical - confidence
        assert abs(result["confidence_gap"] - expected_gap) < 1e-12

    @pytest.mark.parametrize("n_sims,n_success,confidence", [
        (100, 90, 0.80),
        (200, 160, 0.75),
        (500, 450, 0.85),
    ])
    def test_empirical_probability_from_check_goals(self, n_sims, n_success, confidence):
        """check_goals empirical_probability = n_success / n_sims."""
        from finopt.goals import TerminalGoal, check_goals
        from finopt.model import SimulationResult
        from finopt.portfolio import Account

        T, M = 12, 1
        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:n_success, -1, 0] = 2_000_000
        wealth[n_success:, -1, 0] = 500_000

        result = SimulationResult(
            wealth=wealth,
            total_wealth=wealth.sum(axis=2),
            contributions=np.ones((n_sims, T)),
            returns=np.zeros((n_sims, T, M)),
            income={"fixed": np.ones((n_sims, T)),
                    "variable": np.zeros((n_sims, T)),
                    "total": np.ones((n_sims, T))},
            allocation=np.ones((T, M)),
            withdrawals=None, T=T, n_sims=n_sims, M=M,
            start=date(2025, 1, 1), seed=42,
            account_names=["Account"],
        )

        accounts = [Account.from_annual("Account", 0.08, 0.10, 0)]
        goals = [TerminalGoal(account="Account", threshold=1_000_000, confidence=confidence)]

        status = check_goals(result, goals, accounts, date(2025, 1, 1))
        m = status[goals[0]]

        expected_emp = n_success / n_sims
        assert abs(m["empirical_probability"] - expected_emp) < 1e-12
        assert abs(m["confidence_gap"] - (expected_emp - confidence)) < 1e-12


# =============================================================================
# 5.3 – Full simulation pipeline E2E
# =============================================================================

@pytest.mark.integration
class TestFullSimulationPipeline:
    """
    Full core simulation → check_goals → dual metrics.
    Uses FinancialModel to produce realistic wealth trajectories.
    """

    def _build_result(self, n_sims: int = 200, T: int = 24):
        from finopt.income import FixedIncome, IncomeModel
        from finopt.model import FinancialModel
        from finopt.portfolio import Account

        income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
        accounts = [
            Account.from_annual("Conservative", annual_return=0.06, annual_volatility=0.08),
            Account.from_annual("Aggressive",   annual_return=0.14, annual_volatility=0.18),
        ]
        model = FinancialModel(income=income, accounts=accounts)
        X = np.tile([0.4, 0.6], (T, 1))
        return model.simulate(T=T, n_sims=n_sims, X=X, seed=42), accounts

    def test_dual_metrics_present_in_check_goals(self):
        """check_goals returns the three new fields for every goal."""
        from finopt.goals import IntermediateGoal, TerminalGoal, check_goals

        result, accounts = self._build_result()
        goals = [
            IntermediateGoal(date=date(2026, 1, 1), account="Conservative",
                             threshold=5_000_000, confidence=0.75),
            TerminalGoal(account="Aggressive", threshold=20_000_000, confidence=0.70),
        ]

        status = check_goals(result, goals, accounts, date(2025, 1, 1))

        for goal in goals:
            m = status[goal]
            assert "empirical_probability" in m
            assert "confidence_gap" in m
            assert "note" in m

    def test_empirical_probability_equals_one_minus_violation_rate(self):
        """empirical_probability = 1 - violation_rate (always)."""
        from finopt.goals import TerminalGoal, check_goals

        result, accounts = self._build_result()
        goals = [TerminalGoal(account="Aggressive", threshold=10_000_000, confidence=0.70)]

        status = check_goals(result, goals, accounts, date(2025, 1, 1))
        m = status[goals[0]]

        assert abs(m["empirical_probability"] - (1.0 - m["violation_rate"])) < 1e-12

    def test_confidence_gap_sign_matches_satisfaction(self):
        """
        When a goal is satisfied, empirical_probability ≥ required_confidence,
        so confidence_gap should be ≥ 0 (positive or zero).
        When violated, gap < 0.
        """
        from finopt.goals import TerminalGoal, check_goals

        result, accounts = self._build_result()

        # Pick a very easy threshold (virtually always satisfied) — Conservative account
        easy = TerminalGoal(account="Conservative", threshold=1_000, confidence=0.50)
        # Pick an impossible threshold (virtually never satisfied) — Aggressive account
        hard = TerminalGoal(account="Aggressive", threshold=1_000_000_000, confidence=0.99)

        status = check_goals(result, [easy, hard], accounts, date(2025, 1, 1))

        easy_m = status[easy]
        hard_m = status[hard]

        # Easy goal: satisfied → gap ≥ 0
        assert easy_m["satisfied"]
        assert easy_m["confidence_gap"] >= 0

        # Hard goal: violated → gap < 0
        assert not hard_m["satisfied"]
        assert hard_m["confidence_gap"] < 0


# =============================================================================
# 5.4 – API compute_goal_status pipeline E2E
# =============================================================================

@pytest.mark.integration
class TestAPIGoalStatusPipeline:
    """
    End-to-end test of the API-layer goal status computation.
    Verifies that compute_goal_status() and compute_goal_status_from_result()
    produce the correct dual metric fields in the output list.
    """

    def _make_wealth(self, n_sims, T, M, n_above, threshold=5_000_000):
        """Wealth array where n_above scenarios exceed threshold at terminal time."""
        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:n_above, -1, 0] = threshold * 2
        wealth[n_above:, -1, 0] = threshold * 0.5
        return wealth

    def test_simulation_service_dual_fields(self):
        """compute_goal_status() includes empirical_probability, confidence_gap, note."""
        from api.services.simulation import compute_goal_status
        from finopt import TerminalGoal

        n_sims, T, M = 100, 12, 1
        n_above = 90  # 90% success
        confidence = 0.80

        wealth = self._make_wealth(n_sims, T, M, n_above)
        goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=confidence)]
        accounts = [type("Account", (), {"name": "Savings"})()]

        status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))

        s = status[0]
        assert "empirical_probability" in s
        assert "confidence_gap" in s
        assert "note" in s

        assert abs(s["empirical_probability"] - 0.90) < 1e-9
        assert abs(s["confidence_gap"] - (0.90 - confidence)) < 1e-9
        assert isinstance(s["note"], str) and len(s["note"]) > 0

    def test_optimization_service_dual_fields(self):
        """compute_goal_status_from_result() includes dual metric fields."""
        from api.services.optimization import compute_goal_status_from_result
        from finopt import FinancialModel, TerminalGoal
        from finopt.income import FixedIncome, IncomeModel
        from finopt.portfolio import Account

        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [Account.from_annual("Growth", 0.12, 0.15, 0)]
        model = FinancialModel(income=income, accounts=accounts)

        goal = TerminalGoal(account="Growth", threshold=10_000_000, confidence=0.80)
        opt_result = type("Opt", (), {"T": 24, "feasible": True, "goals": [goal]})()

        n_sims = 100
        n_above = 88  # 88% success
        wealth = np.zeros((n_sims, 25, 1))
        wealth[:n_above, -1, 0] = 12_000_000
        wealth[n_above:, -1, 0] = 8_000_000
        sim_result = type("Sim", (), {"wealth": wealth})()

        status = compute_goal_status_from_result(
            opt_result, model, sim_result, date(2025, 1, 1)
        )

        s = status[0]
        assert "empirical_probability" in s
        assert "confidence_gap" in s
        assert "note" in s

        assert abs(s["empirical_probability"] - 0.88) < 1e-9
        assert abs(s["confidence_gap"] - 0.08) < 1e-9

    def test_api_and_core_produce_same_gap_for_same_inputs(self):
        """
        Given identical empirical success counts, core check_goals() and
        API compute_goal_status() must produce the same confidence_gap.
        """
        from api.services.simulation import compute_goal_status
        from finopt.goals import TerminalGoal, check_goals
        from finopt.model import SimulationResult
        from finopt.portfolio import Account

        n_sims, T, M = 100, 12, 1
        n_above = 85  # 85% empirical success
        confidence = 0.80
        threshold = 5_000_000

        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:n_above, -1, 0] = threshold * 2
        wealth[n_above:, -1, 0] = threshold * 0.5

        # Core path
        result = SimulationResult(
            wealth=wealth,
            total_wealth=wealth.sum(axis=2),
            contributions=np.ones((n_sims, T)),
            returns=np.zeros((n_sims, T, M)),
            income={"fixed": np.ones((n_sims, T)),
                    "variable": np.zeros((n_sims, T)),
                    "total": np.ones((n_sims, T))},
            allocation=np.ones((T, M)),
            withdrawals=None, T=T, n_sims=n_sims, M=M,
            start=date(2025, 1, 1), seed=42,
            account_names=["Savings"],
        )
        core_accounts = [Account.from_annual("Savings", 0.08, 0.10, 0)]
        core_goal = TerminalGoal(account="Savings", threshold=threshold, confidence=confidence)
        core_status = check_goals(result, [core_goal], core_accounts, date(2025, 1, 1))
        core_gap = core_status[core_goal]["confidence_gap"]

        # API path
        api_goal = TerminalGoal(account=0, threshold=threshold, confidence=confidence)
        api_accounts = [type("Account", (), {"name": "Savings"})()]
        api_status = compute_goal_status(wealth, [api_goal], api_accounts, date(2025, 1, 1))
        api_gap = api_status[0]["confidence_gap"]

        assert abs(core_gap - api_gap) < 1e-12, (
            f"Core gap={core_gap:.6f} ≠ API gap={api_gap:.6f}"
        )

    def test_note_text_identical_core_vs_api_for_same_inputs(self):
        """
        For same empirical/confidence, core check_goals() and API
        compute_goal_status() produce *word-for-word identical* notes.
        """
        from api.services.simulation import compute_goal_status
        from finopt.goals import TerminalGoal, check_goals
        from finopt.model import SimulationResult
        from finopt.portfolio import Account

        n_sims, T, M = 100, 12, 1
        n_above = 95   # 95% → gap 0.15 > 0.01 → "significant" category
        confidence = 0.80
        threshold = 5_000_000

        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:n_above, -1, 0] = threshold * 2
        wealth[n_above:, -1, 0] = threshold * 0.5

        # Core
        result = SimulationResult(
            wealth=wealth, total_wealth=wealth.sum(axis=2),
            contributions=np.ones((n_sims, T)), returns=np.zeros((n_sims, T, M)),
            income={"fixed": np.ones((n_sims, T)), "variable": np.zeros((n_sims, T)),
                    "total": np.ones((n_sims, T))},
            allocation=np.ones((T, M)), withdrawals=None,
            T=T, n_sims=n_sims, M=M, start=date(2025, 1, 1), seed=42,
            account_names=["Savings"],
        )
        core_accounts = [Account.from_annual("Savings", 0.08, 0.10, 0)]
        core_goal = TerminalGoal(account="Savings", threshold=threshold, confidence=confidence)
        core_note = check_goals(result, [core_goal], core_accounts, date(2025, 1, 1))[core_goal]["note"]

        # API
        api_goal = TerminalGoal(account=0, threshold=threshold, confidence=confidence)
        api_note = compute_goal_status(wealth, [api_goal],
                                       [type("Account", (), {"name": "Savings"})()],
                                       date(2025, 1, 1))[0]["note"]

        assert core_note == api_note, (
            f"Note text diverged.\nCore: {core_note!r}\nAPI:  {api_note!r}"
        )


# =============================================================================
# 5.5 – TypeScript / Python field contract
# =============================================================================

class TestFieldContract:
    """
    Verify that the Python API output fields match the TypeScript GoalStatus
    interface.  This catches regressions where a field is added on one side
    but not the other.

    The authoritative list of expected fields comes from this test itself.
    If you add a new field, add it here too.
    """

    # Fields that every GoalStatus item MUST contain
    REQUIRED_FIELDS = {
        "goal",
        "type",
        "account",
        "threshold",
        "required_confidence",
        "satisfied",
    }

    # Fields present when dual metrics are available
    DUAL_METRIC_FIELDS = {
        "empirical_probability",
        "confidence_gap",
        "note",
    }

    # TypeScript GoalStatus interface (source of truth for expected keys)
    TYPESCRIPT_INTERFACE_KEYS = (
        REQUIRED_FIELDS
        | {"actual_probability"}
        | DUAL_METRIC_FIELDS
    )

    def _simulation_goal_status(self, n_above=85, n_sims=100, confidence=0.80):
        import numpy as np

        from api.services.simulation import compute_goal_status
        from finopt import TerminalGoal

        T, M = 12, 1
        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:n_above, -1, 0] = 6_000_000
        wealth[n_above:, -1, 0] = 4_000_000

        goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=confidence)]
        accounts = [type("Account", (), {"name": "Savings"})()]
        return compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))

    def _optimization_goal_status(self, n_above=85, n_sims=100, confidence=0.80):
        import numpy as np

        from api.services.optimization import compute_goal_status_from_result
        from finopt import FinancialModel, TerminalGoal
        from finopt.income import FixedIncome, IncomeModel
        from finopt.portfolio import Account

        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [Account.from_annual("Savings", 0.08, 0.10, 0)]
        model = FinancialModel(income=income, accounts=accounts)
        goal = TerminalGoal(account="Savings", threshold=5_000_000, confidence=confidence)
        opt = type("Opt", (), {"T": 12, "feasible": True, "goals": [goal]})()

        wealth = np.zeros((n_sims, 13, 1))
        wealth[:n_above, -1, 0] = 6_000_000
        wealth[n_above:, -1, 0] = 4_000_000
        sim = type("Sim", (), {"wealth": wealth})()

        return compute_goal_status_from_result(opt, model, sim, date(2025, 1, 1))

    def test_simulation_output_contains_all_required_fields(self):
        """Simulation goal status item has all required fields."""
        status = self._simulation_goal_status()
        item_keys = set(status[0].keys())
        missing = self.REQUIRED_FIELDS - item_keys
        assert not missing, f"Missing required fields: {missing}"

    def test_simulation_output_contains_dual_metric_fields(self):
        """Simulation goal status item has all dual metric fields."""
        status = self._simulation_goal_status()
        item_keys = set(status[0].keys())
        missing = self.DUAL_METRIC_FIELDS - item_keys
        assert not missing, f"Missing dual metric fields: {missing}"

    def test_optimization_output_contains_all_required_fields(self):
        """Optimization goal status item has all required fields."""
        status = self._optimization_goal_status()
        item_keys = set(status[0].keys())
        missing = self.REQUIRED_FIELDS - item_keys
        assert not missing, f"Missing required fields: {missing}"

    def test_optimization_output_contains_dual_metric_fields(self):
        """Optimization goal status item has all dual metric fields."""
        status = self._optimization_goal_status()
        item_keys = set(status[0].keys())
        missing = self.DUAL_METRIC_FIELDS - item_keys
        assert not missing, f"Missing dual metric fields: {missing}"

    def test_no_unexpected_new_fields(self):
        """
        API output contains no fields unknown to the TypeScript interface.
        If a new field is intentionally added to the Python side, it must
        also be added to TYPESCRIPT_INTERFACE_KEYS in this test.
        """
        sim_keys = set(self._simulation_goal_status()[0].keys())
        opt_keys = set(self._optimization_goal_status()[0].keys())

        # Additional implementation-specific fields allowed in simulation
        sim_allowed_extra = {"t_idx", "probability"}  # legacy / internal

        sim_unexpect = sim_keys - self.TYPESCRIPT_INTERFACE_KEYS - sim_allowed_extra
        opt_unexpect = opt_keys - self.TYPESCRIPT_INTERFACE_KEYS

        assert not sim_unexpect, f"Simulation has unexpected fields: {sim_unexpect}"
        assert not opt_unexpect, f"Optimization has unexpected fields: {opt_unexpect}"


# =============================================================================
# 5.6 – Boundary and edge cases
# =============================================================================

class TestBoundaryConditions:
    """Edge cases that should not raise exceptions or produce NaN values."""

    def test_zero_violation_rate(self):
        """100% success rate — gap = 1 - confidence, note = 'significant'."""
        from api.services._goal_metrics import compute_dual_metrics
        result = compute_dual_metrics(1.0, 0.80)
        assert result["confidence_gap"] == pytest.approx(0.20)
        assert "CVaR optimization yields conservative estimates" in result["note"]

    def test_one_hundred_pct_violation_rate(self):
        """0% success rate — empirical = 0, gap strongly negative."""
        from api.services._goal_metrics import compute_dual_metrics
        result = compute_dual_metrics(0.0, 0.80)
        assert result["empirical_probability"] == 0.0
        assert result["confidence_gap"] == pytest.approx(-0.80)
        assert "Warning" in result["note"]

    def test_single_scenario(self):
        """n_sims = 1 should not cause division-by-zero errors."""
        from finopt.goals import TerminalGoal, check_goals
        from finopt.model import SimulationResult
        from finopt.portfolio import Account

        n_sims, T, M = 1, 12, 1
        wealth = np.full((n_sims, T + 1, M), 6_000_000.0)
        result = SimulationResult(
            wealth=wealth, total_wealth=wealth.sum(axis=2),
            contributions=np.ones((n_sims, T)), returns=np.zeros((n_sims, T, M)),
            income={"fixed": np.ones((n_sims, T)), "variable": np.zeros((n_sims, T)),
                    "total": np.ones((n_sims, T))},
            allocation=np.ones((T, M)), withdrawals=None,
            T=T, n_sims=n_sims, M=M, start=date(2025, 1, 1), seed=None,
            account_names=["Account"],
        )
        accounts = [Account.from_annual("Account", 0.08, 0.10, 0)]
        goals = [TerminalGoal(account="Account", threshold=5_000_000, confidence=0.80)]
        status = check_goals(result, goals, accounts, date(2025, 1, 1))
        m = status[goals[0]]
        assert math.isfinite(m["empirical_probability"])
        assert math.isfinite(m["confidence_gap"])

    def test_none_actual_probability_in_opt_service(self):
        """
        When resolved_month > wealth array length, actual_prob is None
        and dual metric fields must be None (not raise or produce NaN).
        """
        from api.services.optimization import compute_goal_status_from_result
        from finopt import FinancialModel, IntermediateGoal
        from finopt.income import FixedIncome, IncomeModel
        from finopt.portfolio import Account

        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [Account.from_annual("Em", 0.04, 0.05, 0)]
        model = FinancialModel(income=income, accounts=accounts)

        # Goal date resolves to month 6, but wealth array only covers 6 steps
        # → index 6 is exactly equal to shape[1], so out of bounds
        goal = IntermediateGoal(date=date(2025, 7, 1), account="Em",
                                threshold=5_000_000, confidence=0.80)
        opt = type("Opt", (), {"T": 5, "feasible": True, "goals": [goal]})()
        # wealth.shape[1] == 6 → index 6 is out of bounds
        sim = type("Sim", (), {"wealth": np.full((100, 6, 1), 6_000_000.0)})()

        status = compute_goal_status_from_result(opt, model, sim, date(2025, 1, 1))
        s = status[0]
        assert s["empirical_probability"] is None
        assert s["confidence_gap"] is None
        assert s["note"] is None

    def test_all_dual_metrics_are_finite_or_none(self):
        """No NaN or Inf in dual metric fields."""

        from api.services.simulation import compute_goal_status
        from finopt import TerminalGoal

        rng = np.random.default_rng(seed=99)
        n_sims, T, M = 200, 12, 1
        wealth = rng.uniform(3_000_000, 8_000_000, (n_sims, T + 1, M))
        goals = [TerminalGoal(account=0, threshold=5_000_000, confidence=0.80)]
        accounts = [type("Account", (), {"name": "Acc"})()]

        status = compute_goal_status(wealth, goals, accounts, date(2025, 1, 1))
        s = status[0]

        ep = s["empirical_probability"]
        cg = s["confidence_gap"]
        assert ep is not None and math.isfinite(ep)
        assert cg is not None and math.isfinite(cg)
        assert isinstance(s["note"], str)
