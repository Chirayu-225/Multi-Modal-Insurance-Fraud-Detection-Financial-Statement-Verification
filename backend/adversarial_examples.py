"""
Template for the hand-built adversarial test set the project review
asked for -- 50-100 claims YOU construct by hand, with messier and more
subtle mismatches than the generator in app/ml/synthetic_data.py
produces, kept completely separate from anything used in training.

Why this can't be a script: the whole point is that these examples come
from you thinking like a careful fraudster, not from a random-injection
loop. A generator can't red-team itself. What follows are 3 illustrative
examples showing the format and the kind of subtlety to aim for -- not
a finished set. Expand this file (or load these from a spreadsheet/CSV
if that's easier to fill in) to 50-100 before you report numbers on it.

Ideas for genuinely subtle cases worth including:
  - damage consistent with the description AND the photo, but the bill
    padding is subtle (1.3x expected, not 3x -- does your threshold
    catch a *small* inflation, or only blatant ones?)
  - a claim where the photo is a slightly cropped/brightened version of
    another claim's photo (tests pHash robustness, not just exact reuse)
  - an excluded-category item disguised under a vague/misleading label
  - a claim that is genuinely clean but happens to be expensive (tests
    false-positive rate on legitimate severe damage)
  - a timestamp only 1-2 days off in a way that's plausible (weekend
    before reporting) vs. genuinely suspicious
"""
from datetime import date

from app.ml.domain import (
    BillLineItem,
    ClaimEvidence,
    DamageLocation,
    IncidentType,
    InconsistencyType,
    Severity,
    VehicleSegment,
)

ADVERSARIAL_EXAMPLES: list[ClaimEvidence] = [
    # 1. Subtle bill inflation (~1.3x, not the generator's 2.5-4x range) --
    #    does the threshold-based rule, tuned against blatant cases, also
    #    catch a mild one? This is a genuinely open question worth
    #    reporting either way.
    ClaimEvidence(
        claim_id="adv-001",
        vehicle_segment=VehicleSegment.SEDAN,
        incident_date=date(2026, 3, 4),
        described_incident_type=IncidentType.FRONT_COLLISION,
        photo_damage_location=DamageLocation.FRONT_BUMPER,
        photo_damage_severity=Severity.MODERATE,
        photo_hash="a1b2c3d4e5f60718",
        photo_capture_date=date(2026, 3, 5),
        bill_line_items=[BillLineItem(category="front_bumper_repair", cost_inr=17000)],  # expected ~13,000
        injected_inconsistency=InconsistencyType.INFLATED_BILL,
    ),
    # 2. Legitimately severe, legitimately expensive, and NOT fraud --
    #    a false-positive probe.
    ClaimEvidence(
        claim_id="adv-002",
        vehicle_segment=VehicleSegment.SUV,
        incident_date=date(2026, 5, 20),
        described_incident_type=IncidentType.FRONT_COLLISION,
        photo_damage_location=DamageLocation.HOOD,
        photo_damage_severity=Severity.SEVERE,
        photo_hash="1122334455667788",
        photo_capture_date=date(2026, 5, 20),
        bill_line_items=[BillLineItem(category="hood_repair", cost_inr=48000)],  # ~22,000 * 2.2 = 48,400, right at the line
        injected_inconsistency=InconsistencyType.NONE,
    ),
    # 3. Timestamp only slightly off, plausible innocent reason (photo
    #    taken the next morning in daylight) -- should NOT be flagged as
    #    hard as a photo taken 60 days later.
    ClaimEvidence(
        claim_id="adv-003",
        vehicle_segment=VehicleSegment.HATCHBACK,
        incident_date=date(2026, 6, 10),
        described_incident_type=IncidentType.PARKING_SCRAPE,
        photo_damage_location=DamageLocation.LEFT_DOOR,
        photo_damage_severity=Severity.MINOR,
        photo_hash="99aabbccddeeff00",
        photo_capture_date=date(2026, 6, 11),
        bill_line_items=[BillLineItem(category="left_door_repair", cost_inr=4200)],
        injected_inconsistency=InconsistencyType.NONE,
    ),
]
