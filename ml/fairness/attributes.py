"""
Protected vs audited attributes, per data source.

PROTECTED:    never model features. They live only in the audit frame.
AUDIT_GROUPS: grouping variables the fairness audits report on that MAY also be model features.
              Using them as features is deliberate and documented (e.g. debtor status and
              scholarship status are direct need/risk signals; income_slab_idx routes financial
              support), so they are audited rather than excluded.

Tests intersect the actual feature-matrix columns of each source with PROTECTED.
"""

from typing import Dict, List

PROTECTED: Dict[str, List[str]] = {
    "uci": ["gender", "age_at_enrollment"],
    "oulad": ["gender", "age_band", "disability", "imd_band", "region"],
    "simulated": ["gender", "category", "age"],
}

AUDIT_GROUPS: Dict[str, List[str]] = {
    "uci": ["scholarship_holder", "debtor", "displaced"],
    "oulad": ["highest_education"],
    "simulated": ["income_slab_idx"],
}
