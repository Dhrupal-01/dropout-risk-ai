"""
Fairness and Algorithmic Bias Audit Package for DropoutGuard
Provides general audit functionality across empirical benchmarks (UCI ID 697, OULAD UCI ID 349)
and simulated collegiate cohorts, comparative mitigation benchmarking, and temporal presentation shift checks.
"""

from ml.fairness.audit import (
    compute_group_metrics,
    audit_model_fairness,
    compute_within_group_ece,
)

__all__ = [
    "compute_group_metrics",
    "audit_model_fairness",
    "compute_within_group_ece",
]
