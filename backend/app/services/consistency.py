"""
The rules-based consistency engine -- the "real IP" of this project.

It takes a ClaimEvidence (wherever it came from: a real claim processed
through vision.py + ocr.py + policy RAG, or a synthetic claim from
ml/synthetic_data.py) and cross-checks the pieces against each other. It
never talks to an LLM: every flag and every number in the explanation is
derived directly from the structured comparison, and the API layer only
uses an LLM (optionally) to *phrase* these flags in prose -- so the
explanation is guaranteed faithful to the score, per the review feedback
that free-form LLM explanations can be convincing without being true.

This module is also the "rules baseline" that train_fusion_model.py
compares a trained classifier against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.domain import (
    ADJACENT_LOCATIONS,
    BASE_REPAIR_COST_INR,
    EXPECTED_LOCATIONS,
    GST_RATE_TOLERANCE,
    REPAIR_GST_RATE,
    SEVERITY_COST_MULTIPLIER,
    BillLineItem,
    ClaimEvidence,
    InconsistencyType,
)
from app.services import injection_detection
from app.services.gstin import validate_gstin

# Tunable thresholds. Kept as module-level constants (not magic numbers
# buried in the logic) so they're easy to justify and adjust in the
# report, and so train_fusion_model.py can report "rules with threshold
# X" as a genuine baseline sweep if you want to take that further.
COST_RATIO_FLAG_THRESHOLD = 1.6
REUSED_PHOTO_HAMMING_THRESHOLD = 6
TIMESTAMP_GRACE_DAYS = 14
ELA_SCORE_FLAG_THRESHOLD = 0.6
EXTRACTION_DIVERGENCE_COST_RATIO_THRESHOLD = 0.3  # relative difference in bill totals between two independent extractions

# The same risk score used to catch fraud, read in the other direction:
# a claim scoring below this can skip manual review entirely. Chosen by
# fast_track_analysis.py, which sweeps thresholds against a realistic
# 5%-base-rate synthetic set and reports the trade-off explicitly -- at
# this threshold, catching >=98% of injected fraud (2% miss rate) still
# lets ~99% of genuine claims fast-track. Re-run that script and update
# this constant if the generator or rules engine changes meaningfully;
# don't hand-tune it without re-measuring.
FAST_TRACK_RISK_THRESHOLD = 0.30


@dataclass
class Flag:
    type: InconsistencyType
    severity: float  # 0-1, how confident/severe this individual flag is
    message: str


@dataclass
class ConsistencyResult:
    claim_id: str
    risk_score: float  # 0-1, combined across all flags
    flags: list[Flag] = field(default_factory=list)
    model_version: str = "rules_v1"

    @property
    def triggered_types(self) -> set[InconsistencyType]:
        return {f.type for f in self.flags}

    @property
    def fast_track(self) -> bool:
        return is_fast_track(self.risk_score)


def is_fast_track(risk_score: float) -> bool:
    """Plain-float version so callers reconstructing a result from a DB
    row (which stores risk_score but not a live ConsistencyResult) can
    use the exact same threshold without re-running evaluate()."""
    return risk_score < FAST_TRACK_RISK_THRESHOLD


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def _check_location(claim: ClaimEvidence) -> Flag | None:
    expected = EXPECTED_LOCATIONS[claim.described_incident_type]
    if claim.photo_damage_location in expected:
        return None

    adjacent_to_expected: set = set()
    for loc in expected:
        adjacent_to_expected |= ADJACENT_LOCATIONS.get(loc, set())

    if claim.photo_damage_location in adjacent_to_expected:
        return Flag(
            type=InconsistencyType.WRONG_PANEL,
            severity=0.3,
            message=(
                f"Photo shows damage at '{claim.photo_damage_location.value}', which is adjacent to but not "
                f"exactly what '{claim.described_incident_type.value}' would typically produce "
                f"(expected: {', '.join(l.value for l in expected)}). Minor concern -- descriptions are often imprecise."
            ),
        )

    return Flag(
        type=InconsistencyType.WRONG_PANEL,
        severity=0.85,
        message=(
            f"Incident description ('{claim.described_incident_type.value}') implies damage near "
            f"{', '.join(l.value for l in expected)}, but the photo shows damage at "
            f"'{claim.photo_damage_location.value}' -- not adjacent to any expected location. "
            f"The described incident does not explain this photo."
        ),
    )


def _check_bill_inflation(claim: ClaimEvidence) -> Flag | None:
    repair_items = [i for i in claim.bill_line_items if i.category.endswith("_repair")]
    repair_total = sum(i.cost_inr for i in repair_items)
    if repair_total == 0:
        return None

    expected_cost = (
        BASE_REPAIR_COST_INR[claim.photo_damage_location][claim.vehicle_segment]
        * SEVERITY_COST_MULTIPLIER[claim.photo_damage_severity]
    )
    ratio = repair_total / expected_cost
    if ratio <= COST_RATIO_FLAG_THRESHOLD:
        return None

    severity = min(1.0, 0.4 + 0.2 * (ratio - COST_RATIO_FLAG_THRESHOLD))
    return Flag(
        type=InconsistencyType.INFLATED_BILL,
        severity=severity,
        message=(
            f"Repair bill totals ₹{repair_total:,} for {claim.photo_damage_severity.value} damage at "
            f"'{claim.photo_damage_location.value}' on a {claim.vehicle_segment.value}; typical cost for this "
            f"is ₹{expected_cost:,.0f} (≈{ratio:.1f}x higher than expected)."
        ),
    )


def _check_excluded_items(claim: ClaimEvidence) -> Flag | None:
    excluded_hits = [i for i in claim.bill_line_items if i.category in claim.policy_excluded_categories]
    if not excluded_hits:
        return None
    names = ", ".join(sorted({i.category for i in excluded_hits}))
    total = sum(i.cost_inr for i in excluded_hits)
    return Flag(
        type=InconsistencyType.EXCLUDED_ITEM,
        severity=0.7,
        message=(
            f"Bill includes line item(s) this policy explicitly excludes from coverage: {names} "
            f"(₹{total:,} total). These should not be part of a valid claim payout."
        ),
    )


def _check_reused_photo(claim: ClaimEvidence, other_photo_hashes: dict[str, str]) -> Flag | None:
    best_claim_id, best_distance = None, None
    for other_id, other_hash in other_photo_hashes.items():
        if other_id == claim.claim_id:
            continue
        d = _hamming_distance(claim.photo_hash, other_hash)
        if best_distance is None or d < best_distance:
            best_distance, best_claim_id = d, other_id

    if best_distance is None or best_distance > REUSED_PHOTO_HAMMING_THRESHOLD:
        return None

    severity = 1.0 - (best_distance / (REUSED_PHOTO_HAMMING_THRESHOLD + 1))
    return Flag(
        type=InconsistencyType.REUSED_PHOTO,
        severity=max(0.5, severity),
        message=(
            f"This claim's photo is a near-match (Hamming distance {best_distance}) to a photo already on "
            f"file for claim {best_claim_id[:8]} -- possible reused image."
        ),
    )


def _check_timestamp(claim: ClaimEvidence) -> Flag | None:
    if claim.photo_capture_date is None:
        return None
    delta_days = (claim.photo_capture_date - claim.incident_date).days

    if delta_days < 0:
        return Flag(
            type=InconsistencyType.TIMESTAMP_MISMATCH,
            severity=0.95,
            message=(
                f"Photo's capture timestamp ({claim.photo_capture_date}) is before the claimed incident date "
                f"({claim.incident_date}) -- the photo cannot show damage from an incident that hadn't happened yet."
            ),
        )
    if delta_days > TIMESTAMP_GRACE_DAYS:
        severity = min(1.0, 0.3 + 0.02 * (delta_days - TIMESTAMP_GRACE_DAYS))
        return Flag(
            type=InconsistencyType.TIMESTAMP_MISMATCH,
            severity=severity,
            message=(
                f"Photo was captured {delta_days} days after the claimed incident date "
                f"({claim.incident_date}) -- unusually long for damage documentation."
            ),
        )
    return None


def _check_prompt_injection(claim: ClaimEvidence) -> Flag | None:
    hit = injection_detection.detect(claim.incident_description_text, claim.raw_bill_ocr_text)
    if hit is None:
        return None
    inconsistency_type, severity, message = hit
    return Flag(type=inconsistency_type, severity=severity, message=message)


def _check_ela_anomaly(claim: ClaimEvidence) -> Flag | None:
    """Only ever fires on real uploaded photos -- ela_score is None for
    every synthetic claim (see the field's docstring in domain.py), so
    this check is inert in train_fusion_model.py and the pytest suite by
    design, not by oversight."""
    if claim.ela_score is None or claim.ela_score <= ELA_SCORE_FLAG_THRESHOLD:
        return None
    severity = min(1.0, 0.3 + 0.5 * (claim.ela_score - ELA_SCORE_FLAG_THRESHOLD))
    return Flag(
        type=InconsistencyType.EXTRACTION_DIVERGENCE,  # reusing the "something doesn't add up at the pixel level" bucket
        severity=severity,
        message=(
            f"Photo shows an unusually high error-level-analysis score ({claim.ela_score:.2f}), consistent with "
            f"localized editing or splicing. This is a naive heuristic -- it catches obvious edits and recompression "
            f"artifacts, not careful AI inpainting -- and should be treated as a weak signal, not a verdict."
        ),
    )


def compare_bill_extractions(
    primary: list[BillLineItem], secondary: list[BillLineItem], threshold: float = EXTRACTION_DIVERGENCE_COST_RATIO_THRESHOLD
) -> Flag | None:
    """Cross-check defense: if two independent extractions of the same
    bill image (e.g. Gemini and Tesseract, run in parallel) disagree
    substantially, that's suspicious on its own -- a prompt injection
    aimed at the vision model usually corrupts exactly one of the two
    extractions, since the injected text only influences the model that
    reads it as instructions rather than pixels/characters.

    Wired up in pipeline.py behind `settings.enable_cross_check`, since
    it only makes sense when two backends actually ran -- with the
    default single-backend (mock-only) setup there's nothing to compare
    against, so this stays untested by the main evaluation scripts for
    the same honest reason ela_score is.
    """
    primary_total = sum(i.cost_inr for i in primary)
    secondary_total = sum(i.cost_inr for i in secondary)
    if primary_total == 0 and secondary_total == 0:
        return None

    denom = max(primary_total, secondary_total, 1)
    relative_diff = abs(primary_total - secondary_total) / denom

    primary_categories = {i.category for i in primary}
    secondary_categories = {i.category for i in secondary}
    category_overlap = len(primary_categories & secondary_categories) / max(len(primary_categories | secondary_categories), 1)

    if relative_diff <= threshold and category_overlap >= 0.5:
        return None

    severity = min(1.0, 0.4 + relative_diff)
    return Flag(
        type=InconsistencyType.EXTRACTION_DIVERGENCE,
        severity=severity,
        message=(
            f"Two independent extractions of the bill disagree: totals differ by {relative_diff:.0%} "
            f"({primary_total:,} vs {secondary_total:,}), category overlap {category_overlap:.0%}. "
            f"This can mean OCR noise -- but it is also the expected signature of a prompt injection that "
            f"successfully corrupted one extraction but not the other."
        ),
    )


def _check_invalid_gstin(claim: ClaimEvidence) -> Flag | None:
    if not claim.garage_gstin:
        return None  # no GSTIN extracted at all -- worth surfacing separately in the UI, not scored as fraud on its own
    is_valid, reason = validate_gstin(claim.garage_gstin)
    if is_valid:
        return None
    return Flag(
        type=InconsistencyType.INVALID_GSTIN,
        severity=0.75,
        message=(
            f"The garage's GSTIN on this bill ('{claim.garage_gstin}') fails validation: {reason}. "
            f"This is a deterministic structural check, not a judgment call -- a well-formed GSTIN "
            f"either has a correct checksum digit or it doesn't."
        ),
    )


def _check_gst_rate_mismatch(claim: ClaimEvidence) -> Flag | None:
    mismatched = [
        item
        for item in claim.bill_line_items
        if item.claimed_gst_rate is not None and abs(item.claimed_gst_rate - REPAIR_GST_RATE) > GST_RATE_TOLERANCE
    ]
    if not mismatched:
        return None
    details = ", ".join(f"{i.category} at {i.claimed_gst_rate:.0%}" for i in mismatched)
    return Flag(
        type=InconsistencyType.GST_RATE_MISMATCH,
        severity=0.5,
        message=(
            f"Bill charges a GST rate other than the {REPAIR_GST_RATE:.0%} standard rate for motor-vehicle "
            f"parts/labour on: {details}. This can mean an outdated rate, a miscategorized part, or a "
            f"fabricated bill -- verify against the current CBIC notification before treating this as conclusive."
        ),
    )


def evaluate(
    claim: ClaimEvidence,
    other_photo_hashes: dict[str, str] | None = None,
    extra_flags: list[Flag] | None = None,
) -> ConsistencyResult:
    """Run every rule and combine the flags into one risk score.

    `other_photo_hashes` should map claim_id -> photo_hash for every
    *other* claim you want to check reuse against (e.g. recent claims in
    the DB). Pass {} if you don't have that context yet.

    `extra_flags` lets a caller (pipeline.py) inject flags produced
    outside the rules engine -- e.g. "the vision model's response didn't
    match the required schema", which pipeline.py treats as a suspected
    injection attempt rather than crashing the request.
    """
    checks = [
        _check_location(claim),
        _check_bill_inflation(claim),
        _check_excluded_items(claim),
        _check_reused_photo(claim, other_photo_hashes or {}),
        _check_timestamp(claim),
        _check_prompt_injection(claim),
        _check_ela_anomaly(claim),
        _check_invalid_gstin(claim),
        _check_gst_rate_mismatch(claim),
    ]
    flags = [f for f in checks if f is not None] + (extra_flags or [])

    # Noisy-OR: combine independent flag severities into one score
    # without a plain sum blowing past 1.0.
    risk = 1.0
    for f in flags:
        risk *= (1.0 - f.severity)
    risk_score = round(1.0 - risk, 4)

    return ConsistencyResult(claim_id=claim.claim_id, risk_score=risk_score, flags=flags, model_version="rules_v1")
