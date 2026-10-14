"""
Regression tests for the security-extension pieces: prompt-injection
detection and the bill cross-check comparator. These are fast pytest
checks (lock in behavior for CI); the fuller, honestly-caveated
reporting lives in test_prompt_injection_defense.py, which you run
separately when you want the numbers for your report, not on every
`pytest -q`.
"""
from app.ml.domain import BillLineItem, InconsistencyType
from app.ml.synthetic_data import FRAUD_INCONSISTENCY_TYPES, generate_dataset
from app.services.consistency import compare_bill_extractions, evaluate


def test_injection_claims_are_flagged():
    claims = generate_dataset(100, inconsistency_rate=1.0, seed=10, inconsistency_types=[InconsistencyType.PROMPT_INJECTION])
    hashes = {c.claim_id: c.photo_hash for c in claims}
    detected = sum(
        1 for c in claims if InconsistencyType.PROMPT_INJECTION in evaluate(c, other_photo_hashes=hashes).triggered_types
    )
    assert detected / len(claims) > 0.95


def test_ordinary_fraud_claims_never_trigger_injection_flag():
    """Cross-contamination check: the five original fraud types shouldn't
    accidentally trip the injection detector -- they don't touch
    description/OCR text at all, so this should be exactly 0."""
    claims = generate_dataset(300, inconsistency_rate=1.0, seed=11, inconsistency_types=FRAUD_INCONSISTENCY_TYPES)
    hashes = {c.claim_id: c.photo_hash for c in claims}
    false_alarms = sum(
        1 for c in claims if InconsistencyType.PROMPT_INJECTION in evaluate(c, other_photo_hashes=hashes).triggered_types
    )
    assert false_alarms == 0


def test_compare_bill_extractions_flags_large_divergence():
    a = [BillLineItem(category="front_bumper_repair", cost_inr=9000)]
    b = [
        BillLineItem(category="front_bumper_repair", cost_inr=9000),
        BillLineItem(category="windshield_repair", cost_inr=7000),
    ]
    flag = compare_bill_extractions(a, b)
    assert flag is not None
    assert flag.type == InconsistencyType.EXTRACTION_DIVERGENCE


def test_compare_bill_extractions_tolerates_minor_ocr_noise():
    a = [BillLineItem(category="front_bumper_repair", cost_inr=9000)]
    b = [BillLineItem(category="front_bumper_repair", cost_inr=9150)]  # OCR rounding-type noise
    assert compare_bill_extractions(a, b) is None
