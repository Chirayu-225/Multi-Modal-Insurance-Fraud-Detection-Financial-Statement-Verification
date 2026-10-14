"""
Regression tests for the rules engine and the synthetic generator. Run
with: pytest -q (from backend/).

These are deliberately not the "does the model generalize" evaluation --
that's train_fusion_model.py, which is a research/reporting script, not
a fast CI-style test. These tests just lock in that each rule fires on
the inconsistency type it's meant to catch, and that clean claims stay
clean, so a future refactor can't silently break one of the five checks.
"""
from app.ml.domain import InconsistencyType
from app.ml.synthetic_data import generate_dataset
from app.services.consistency import evaluate


def test_clean_claims_mostly_score_low():
    claims = generate_dataset(100, inconsistency_rate=0.0, seed=1)
    hashes = {c.claim_id: c.photo_hash for c in claims}
    scores = [evaluate(c, other_photo_hashes=hashes).risk_score for c in claims]
    assert sum(1 for s in scores if s > 0.3) / len(scores) < 0.05


def test_each_inconsistency_type_is_detected():
    for kind in [
        InconsistencyType.WRONG_PANEL,
        InconsistencyType.INFLATED_BILL,
        InconsistencyType.EXCLUDED_ITEM,
        InconsistencyType.REUSED_PHOTO,
        InconsistencyType.TIMESTAMP_MISMATCH,
    ]:
        claims = generate_dataset(150, inconsistency_rate=1.0, seed=2, inconsistency_types=[kind])
        hashes = {c.claim_id: c.photo_hash for c in claims}
        injected = [c for c in claims if c.injected_inconsistency == kind]
        assert injected, f"generator produced no {kind} claims (check domain.py edge cases)"

        detected = sum(1 for c in injected if kind in evaluate(c, other_photo_hashes=hashes).triggered_types)
        detection_rate = detected / len(injected)
        assert detection_rate > 0.75, f"{kind} only detected {detection_rate:.0%} of the time"


def test_explanations_are_populated_for_every_flag():
    claims = generate_dataset(50, inconsistency_rate=1.0, seed=3)
    hashes = {c.claim_id: c.photo_hash for c in claims}
    for c in claims:
        result = evaluate(c, other_photo_hashes=hashes)
        for flag in result.flags:
            assert flag.message, "every flag must carry a human-readable explanation"
            assert 0.0 <= flag.severity <= 1.0
