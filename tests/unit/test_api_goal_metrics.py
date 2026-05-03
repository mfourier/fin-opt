"""
Unit tests for api/services/_goal_metrics.py

Tests the compute_dual_metrics() helper that both simulation and
optimization services use to produce CVaR transparency fields.
"""

import pytest
from api.services._goal_metrics import compute_dual_metrics


class TestComputeDualMetrics:
    """Tests for the shared dual metrics helper."""

    def test_returns_required_keys(self):
        """compute_dual_metrics always returns all three keys."""
        result = compute_dual_metrics(0.90, 0.80)
        assert "empirical_probability" in result
        assert "confidence_gap" in result
        assert "note" in result

    def test_empirical_probability_preserved(self):
        """empirical_probability equals the input value exactly."""
        result = compute_dual_metrics(0.923, 0.85)
        assert result["empirical_probability"] == 0.923

    def test_confidence_gap_calculation(self):
        """confidence_gap = empirical_probability - specified_confidence."""
        result = compute_dual_metrics(0.92, 0.85)
        assert abs(result["confidence_gap"] - 0.07) < 1e-10

    # ------------------------------------------------------------------ notes

    def test_note_significant_conservatism(self):
        """Gap > 1% → 'CVaR optimization yields conservative estimates' note."""
        result = compute_dual_metrics(0.95, 0.80)
        assert result["confidence_gap"] > 0.01
        assert "CVaR optimization yields conservative estimates" in result["note"]
        assert "safety margin" in result["note"]

    def test_note_mild_conservatism(self):
        """0% < gap ≤ 1% → 'CVaR constraint satisfied' note."""
        # 80.5% empirical vs 80% specified → gap = 0.5%
        result = compute_dual_metrics(0.805, 0.800)
        assert 0 <= result["confidence_gap"] <= 0.01
        assert "CVaR constraint satisfied" in result["note"]
        assert "safety margin" not in result["note"]

    def test_note_exact_zero_gap(self):
        """Gap = 0 exactly → mild conservatism note."""
        result = compute_dual_metrics(0.80, 0.80)
        assert result["confidence_gap"] == 0.0
        assert "CVaR constraint satisfied" in result["note"]

    def test_note_violation(self):
        """Negative gap → Warning note."""
        result = compute_dual_metrics(0.75, 0.85)
        assert result["confidence_gap"] < 0
        assert "Warning" in result["note"]
        assert "below specified confidence" in result["note"]

    # ----------------------------------------------------------------- types

    def test_return_types_are_float(self):
        """Numeric outputs are plain Python floats, not numpy floats."""
        result = compute_dual_metrics(0.90, 0.80)
        assert type(result["empirical_probability"]) is float
        assert type(result["confidence_gap"]) is float

    def test_note_is_string(self):
        """note is always a string."""
        result = compute_dual_metrics(0.90, 0.80)
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 0

    # ----------------------------------------------------------------- edge

    def test_high_confidence(self):
        """Works for high confidence levels (e.g., 0.99)."""
        result = compute_dual_metrics(0.995, 0.99)
        assert abs(result["confidence_gap"] - 0.005) < 1e-10
        assert "CVaR constraint satisfied" in result["note"]

    def test_low_confidence(self):
        """Works for low confidence levels (e.g., 0.50)."""
        result = compute_dual_metrics(0.65, 0.50)
        assert result["confidence_gap"] > 0.01
        assert "conservative" in result["note"]

    def test_note_includes_formatted_percentages(self):
        """Note contains human-readable percentages."""
        result = compute_dual_metrics(0.95, 0.85)
        assert "85.0%" in result["note"]
        assert "95.0%" in result["note"]
