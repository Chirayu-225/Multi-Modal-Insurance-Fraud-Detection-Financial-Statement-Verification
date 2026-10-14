"""Motor-claim rule pack.

Deliberately a thin re-export: services/consistency.py stays the single
source of truth for the 9 original checks, so every original test,
train_fusion_model.py and fast_track_analysis.py keep importing from
where they always did.
"""
from __future__ import annotations

from app.services.consistency import (  # noqa: F401  (re-exported)
    FAST_TRACK_RISK_THRESHOLD,
    ConsistencyResult,
    Flag,
    evaluate,
    is_fast_track,
)

PACK_NAME = "motor_claims"
PACK_VERSION = "rules_v1"
