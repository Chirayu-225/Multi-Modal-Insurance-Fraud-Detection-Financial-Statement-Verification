"""
Regression tests for GSTIN validation and the GST-rate-mismatch check.
The checksum algorithm itself is verified against a known-good public
reference example in test_gstin_checksum_matches_known_reference_example
-- not just "does it run", but "does it agree with a real GSTIN".
"""
from app.ml.domain import InconsistencyType
from app.ml.synthetic_data import generate_dataset
from app.services.consistency import evaluate
from app.services.gstin import compute_checksum_char, generate_plausible_gstin, validate_gstin


def test_gstin_checksum_matches_known_reference_example():
    # From the public GST Developer Network reference algorithm: the
    # first 14 characters "27AAPFU0939F1Z" produce checksum "V".
    assert compute_checksum_char("27AAPFU0939F1Z") == "V"
    assert validate_gstin("27AAPFU0939F1ZV") == (True, "valid")


def test_validate_gstin_rejects_corrupted_checksum():
    is_valid, reason = validate_gstin("27AAPFU0939F1ZX")
    assert not is_valid
    assert "checksum" in reason


def test_validate_gstin_handles_missing_and_malformed_input():
    assert validate_gstin(None) == (False, "no GSTIN present on the bill")
    assert not validate_gstin("too-short")[0]


def test_generate_plausible_gstin_is_self_consistent():
    import random

    rng = random.Random(1)
    for _ in range(50):
        gstin = generate_plausible_gstin(rng)
        assert validate_gstin(gstin)[0], f"generated GSTIN {gstin} should validate against its own checksum"


def test_invalid_gstin_and_gst_rate_claims_are_flagged_with_no_cross_contamination():
    for kind in [InconsistencyType.INVALID_GSTIN, InconsistencyType.GST_RATE_MISMATCH]:
        claims = generate_dataset(150, inconsistency_rate=1.0, seed=21, inconsistency_types=[kind])
        hashes = {c.claim_id: c.photo_hash for c in claims}
        detected = sum(1 for c in claims if kind in evaluate(c, other_photo_hashes=hashes).triggered_types)
        assert detected / len(claims) > 0.95

    clean = generate_dataset(300, inconsistency_rate=0.0, seed=22)
    hashes = {c.claim_id: c.photo_hash for c in clean}
    for c in clean:
        triggered = evaluate(c, other_photo_hashes=hashes).triggered_types
        assert InconsistencyType.INVALID_GSTIN not in triggered
        assert InconsistencyType.GST_RATE_MISMATCH not in triggered
