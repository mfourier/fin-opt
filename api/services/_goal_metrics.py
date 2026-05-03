"""
Shared helpers for CVaR dual metric computation.

Both simulation and optimization services report the same dual metrics
for intellectual honesty about CVaR conservatism:
  CVaR_ε(b - W) ≤ 0  ⟹  ℙ(W ≥ b) ≥ 1-ε  (one-way implication)

Empirical probability typically exceeds specified confidence.
Both are reported so users understand the actual risk achieved.
"""

from __future__ import annotations

import numpy as np


def compute_goal_probability(
    wealth_slice: np.ndarray,
    threshold: float,
    confidence: float,
) -> tuple[float, dict]:
    """
    Compute achievement probability and dual metrics for a single goal.

    Parameters
    ----------
    wealth_slice : np.ndarray, shape (n_sims,)
        Wealth values across all scenarios at the goal's evaluation time.
    threshold : float
        Minimum wealth required to satisfy the goal.
    confidence : float
        Required confidence level (1 - ε).

    Returns
    -------
    tuple[float, dict]
        (actual_prob, dual_metrics) where dual_metrics has keys
        empirical_probability, confidence_gap, note.
    """
    actual_prob = float(np.mean(wealth_slice >= threshold))
    dual = compute_dual_metrics(actual_prob, confidence)
    return actual_prob, dual


def compute_dual_metrics(
    empirical_probability: float,
    specified_confidence: float,
) -> dict:
    """
    Compute CVaR dual metrics for a single goal.

    Parameters
    ----------
    empirical_probability : float
        Observed success rate: ℙ̂(W ≥ threshold) from Monte Carlo.
    specified_confidence : float
        Required confidence from goal definition (1 - ε).

    Returns
    -------
    dict
        Keys: empirical_probability, confidence_gap, note.
    """
    confidence_gap = empirical_probability - specified_confidence

    if confidence_gap > 0.01:
        note = (
            f"CVaR optimization yields conservative estimates. "
            f"Specified confidence {specified_confidence:.1%} guarantees at least "
            f"{empirical_probability:.1%} empirical success rate "
            f"(+{confidence_gap:.1%} safety margin)."
        )
    elif confidence_gap >= 0:
        note = (
            f"CVaR constraint satisfied with empirical probability "
            f"{empirical_probability:.1%} (≥ specified {specified_confidence:.1%})."
        )
    else:
        note = (
            f"Warning: Empirical probability {empirical_probability:.1%} "
            f"is below specified confidence {specified_confidence:.1%}. "
            f"This may indicate CVaR approximation error or insufficient scenarios."
        )

    return {
        "empirical_probability": float(empirical_probability),
        "confidence_gap": float(confidence_gap),
        "note": note,
    }
