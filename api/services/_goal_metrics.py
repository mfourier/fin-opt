"""
Shared helpers for CVaR dual metric computation.

Both simulation and optimization services report the same dual metrics
for intellectual honesty about CVaR conservatism:
  CVaR_ε(b - W) ≤ 0  ⟹  ℙ(W ≥ b) ≥ 1-ε  (one-way implication)

Empirical probability typically exceeds specified confidence.
Both are reported so users understand the actual risk achieved.
"""

from __future__ import annotations


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
