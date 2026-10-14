"""
Regression tests for the fast-track threshold (see fast_track_analysis.py
for how FAST_TRACK_RISK_THRESHOLD was chosen). These don't re-derive the
threshold -- that's what the analysis script is for -- they just lock in
the two behaviors the API depends on: a clean claim is fast-tracked, and
a claim with a real fraud flag isn't.
"""
from app.ml.domain import InconsistencyType
from app.ml.synthetic_data import generate_dataset
from app.services.consistency import FAST_TRACK_RISK_THRESHOLD, evaluate, is_fast_track


def test_is_fast_track_is_a_plain_threshold_on_risk_score():
    assert is_fast_track(0.0) is True
    assert is_fast_track(FAST_TRACK_RISK_THRESHOLD - 0.01) is True
    assert is_fast_track(FAST_TRACK_RISK_THRESHOLD) is False
    assert is_fast_track(1.0) is False


def test_clean_claims_are_overwhelmingly_fast_tracked():
    claims = generate_dataset(300, inconsistency_rate=0.0, seed=42)
    hashes = {c.claim_id: c.photo_hash for c in claims}
    results = [evaluate(c, other_photo_hashes=hashes) for c in claims]
    fast_tracked = sum(1 for r in results if r.fast_track)
    assert fast_tracked / len(results) > 0.95


def test_flagged_fraud_types_are_not_fast_tracked():
    for kind in [InconsistencyType.INFLATED_BILL, InconsistencyType.WRONG_PANEL, InconsistencyType.EXCLUDED_ITEM]:
        claims = generate_dataset(100, inconsistency_rate=1.0, seed=9, inconsistency_types=[kind])
        # WRONG_PANEL on a PARKING_SCRAPE-described incident silently falls
        # back to a clean claim (see synthetic_data.py::_inject -- "any
        # panel is plausible" for a parking scrape, so there's no such
        # thing as a wrong panel there). Those are genuinely clean, not a
        # missed detection, so only check claims where the injection
        # actually landed.
        actually_injected = [c for c in claims if c.injected_inconsistency == kind]
        hashes = {c.claim_id: c.photo_hash for c in claims}
        results = [evaluate(c, other_photo_hashes=hashes) for c in actually_injected]
        fast_tracked = sum(1 for r in results if r.fast_track)
        assert fast_tracked / len(results) < 0.1, f"{kind} claims are being fast-tracked far more than expected"
