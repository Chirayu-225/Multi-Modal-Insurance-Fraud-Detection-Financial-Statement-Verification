"""
Shared vocabulary for the claim-consistency pipeline.

This module is the one place that defines what a "damage location" is,
what an "incident type" is, and what the reference repair-cost table
looks like. Three very different pieces of code all import from here:

  1. app/ml/synthetic_data.py   - generates labeled synthetic claims for
                                   training/evaluation using this vocabulary
  2. app/services/consistency.py - the rules engine that cross-checks
                                   real (or mock-extracted) claim evidence
                                   against this same vocabulary
  3. app/services/vision.py     - the mock vision backend, which must
                                   emit outputs in this vocabulary so the
                                   consistency engine can consume them
                                   whether the evidence came from a real
                                   photo or a synthetic one

Keeping it in one file means the synthetic generator and the real
pipeline can never silently drift apart (e.g. one using "left_door" and
the other "door_left").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class DamageLocation(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    HOOD = "hood"
    TRUNK = "trunk"
    LEFT_DOOR = "left_door"
    RIGHT_DOOR = "right_door"
    LEFT_FENDER = "left_fender"
    RIGHT_FENDER = "right_fender"
    WINDSHIELD = "windshield"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


SEVERITY_COST_MULTIPLIER = {
    Severity.MINOR: 0.4,
    Severity.MODERATE: 1.0,
    Severity.SEVERE: 2.2,
}


class IncidentType(str, Enum):
    FRONT_COLLISION = "front_collision"
    REAR_COLLISION = "rear_collision"
    LEFT_SIDE_COLLISION = "left_side_collision"
    RIGHT_SIDE_COLLISION = "right_side_collision"
    PARKING_SCRAPE = "parking_scrape"

# Which damage locations a given incident type is expected to produce.
# The first set is "exact match", the second is "adjacent / plausible,
# don't hard-fail on this" -- real incident descriptions are imprecise,
# and a rules engine that flags every near-miss as fraud is useless.
EXPECTED_LOCATIONS: dict[IncidentType, set[DamageLocation]] = {
    IncidentType.FRONT_COLLISION: {DamageLocation.FRONT_BUMPER, DamageLocation.HOOD, DamageLocation.WINDSHIELD},
    IncidentType.REAR_COLLISION: {DamageLocation.REAR_BUMPER, DamageLocation.TRUNK},
    IncidentType.LEFT_SIDE_COLLISION: {DamageLocation.LEFT_DOOR, DamageLocation.LEFT_FENDER},
    IncidentType.RIGHT_SIDE_COLLISION: {DamageLocation.RIGHT_DOOR, DamageLocation.RIGHT_FENDER},
    IncidentType.PARKING_SCRAPE: set(DamageLocation),  # any panel, low severity expected
}

ADJACENT_LOCATIONS: dict[DamageLocation, set[DamageLocation]] = {
    DamageLocation.FRONT_BUMPER: {DamageLocation.HOOD, DamageLocation.LEFT_FENDER, DamageLocation.RIGHT_FENDER},
    DamageLocation.REAR_BUMPER: {DamageLocation.TRUNK},
    DamageLocation.HOOD: {DamageLocation.FRONT_BUMPER, DamageLocation.WINDSHIELD},
    DamageLocation.TRUNK: {DamageLocation.REAR_BUMPER},
    DamageLocation.LEFT_DOOR: {DamageLocation.LEFT_FENDER},
    DamageLocation.RIGHT_DOOR: {DamageLocation.RIGHT_FENDER},
    DamageLocation.LEFT_FENDER: {DamageLocation.FRONT_BUMPER, DamageLocation.LEFT_DOOR},
    DamageLocation.RIGHT_FENDER: {DamageLocation.FRONT_BUMPER, DamageLocation.RIGHT_DOOR},
    DamageLocation.WINDSHIELD: {DamageLocation.HOOD},
}


class VehicleSegment(str, Enum):
    HATCHBACK = "hatchback"
    SEDAN = "sedan"
    SUV = "suv"


# Synthetic reference repair-cost table (INR), base cost at "moderate"
# severity for a part replacement/repair. This is exactly the "you need a
# parts vocabulary and reference price table" gap called out in review --
# it's synthetic and clearly labeled as such, but it's what lets the
# system say *why* a bill looks inflated instead of just "it's a big
# number".
BASE_REPAIR_COST_INR: dict[DamageLocation, dict[VehicleSegment, int]] = {
    DamageLocation.FRONT_BUMPER: {VehicleSegment.HATCHBACK: 9000, VehicleSegment.SEDAN: 13000, VehicleSegment.SUV: 18000},
    DamageLocation.REAR_BUMPER: {VehicleSegment.HATCHBACK: 8500, VehicleSegment.SEDAN: 12000, VehicleSegment.SUV: 17000},
    DamageLocation.HOOD: {VehicleSegment.HATCHBACK: 11000, VehicleSegment.SEDAN: 16000, VehicleSegment.SUV: 22000},
    DamageLocation.TRUNK: {VehicleSegment.HATCHBACK: 10000, VehicleSegment.SEDAN: 14000, VehicleSegment.SUV: 19000},
    DamageLocation.LEFT_DOOR: {VehicleSegment.HATCHBACK: 9500, VehicleSegment.SEDAN: 13500, VehicleSegment.SUV: 19000},
    DamageLocation.RIGHT_DOOR: {VehicleSegment.HATCHBACK: 9500, VehicleSegment.SEDAN: 13500, VehicleSegment.SUV: 19000},
    DamageLocation.LEFT_FENDER: {VehicleSegment.HATCHBACK: 6000, VehicleSegment.SEDAN: 8500, VehicleSegment.SUV: 12000},
    DamageLocation.RIGHT_FENDER: {VehicleSegment.HATCHBACK: 6000, VehicleSegment.SEDAN: 8500, VehicleSegment.SUV: 12000},
    DamageLocation.WINDSHIELD: {VehicleSegment.HATCHBACK: 7000, VehicleSegment.SEDAN: 9000, VehicleSegment.SUV: 11000},
}

# Bill line-item categories that a typical motor policy excludes. Used to
# simulate the policy-RAG output without needing a real policy PDF yet.
POLICY_EXCLUDED_CATEGORIES = {"cosmetic_detailing", "personal_belongings", "consumables_topup", "mechanical_wear_and_tear"}


class InconsistencyType(str, Enum):
    """The taxonomy every injected synthetic claim is tagged with. This
    is what makes held-out-type evaluation possible: train on some of
    these, test on the ones the model never saw.

    The last three are a different axis from the first five: WRONG_PANEL
    through TIMESTAMP_MISMATCH are *fraud* signals (the claim itself is
    internally inconsistent). PROMPT_INJECTION and EXTRACTION_DIVERGENCE
    are *attack/integrity* signals (someone is trying to manipulate the
    system doing the checking, or two independent extractions of the same
    evidence disagree). They share this enum and the same evaluate()
    pipeline because the infrastructure -- generate, detect, score,
    explain -- is identical; see train_fusion_model.py and
    test_prompt_injection_defense.py for why they're still reported as
    separate tables rather than folded into one fraud number.
    """
    NONE = "none"  # a clean, consistent claim -- the negative class
    WRONG_PANEL = "wrong_panel"
    INFLATED_BILL = "inflated_bill"
    EXCLUDED_ITEM = "excluded_item"
    REUSED_PHOTO = "reused_photo"
    TIMESTAMP_MISMATCH = "timestamp_mismatch"
    PROMPT_INJECTION = "prompt_injection"
    EXTRACTION_DIVERGENCE = "extraction_divergence"
    INVALID_GSTIN = "invalid_gstin"
    GST_RATE_MISMATCH = "gst_rate_mismatch"


# GST rate applied to motor-vehicle body parts (HSN 8708) and to repair/
# service labour, per Notification No. 9/2025-Integrated Tax (Rate),
# effective 22 Sept 2025 (the "GST 2.0" rate rationalization). Both parts
# and labour sit at the same standard rate, split evenly as CGST+SGST for
# an intra-state supply. VERIFY AGAINST THE CURRENT CBIC NOTIFICATION
# before relying on this for anything beyond a plausibility check --
# GST rates are exactly the kind of fact that gets superseded, and this
# was last checked in September 2026.
REPAIR_GST_RATE = 0.18
GST_RATE_TOLERANCE = 0.02  # allow +/-2 percentage points before flagging (rounding, minor category ambiguity)


@dataclass
class BillLineItem:
    category: str  # e.g. "front_bumper_replacement" or "cosmetic_detailing"
    cost_inr: int
    claimed_gst_rate: float | None = None  # the GST rate printed on the bill for this line item, if extracted


@dataclass
class ClaimEvidence:
    """
    The structured output of the extraction stage (vision + OCR + policy
    RAG), regardless of whether that extraction came from a real photo
    via Gemini, or from the synthetic generator standing in for it. The
    consistency engine only ever sees this shape -- that decoupling is
    deliberate, see the module docstring in synthetic_data.py.
    """
    claim_id: str
    vehicle_segment: VehicleSegment
    incident_date: date

    described_incident_type: IncidentType
    photo_damage_location: DamageLocation
    photo_damage_severity: Severity
    photo_hash: str  # perceptual hash (hex string) of the claim photo
    photo_capture_date: date | None  # from EXIF, if present

    bill_line_items: list[BillLineItem] = field(default_factory=list)

    policy_excluded_categories: set[str] = field(default_factory=lambda: set(POLICY_EXCLUDED_CATEGORIES))

    # The garage's GSTIN as printed/OCR'd off the bill, if present. None
    # means "not extracted" (e.g. bill has no visible GSTIN) rather than
    # "invalid" -- those are different findings, see
    # services/gstin.py::validate_gstin and the _check_invalid_gstin rule.
    garage_gstin: str | None = None

    # Raw claimant-controlled text: the free-text incident description,
    # and whatever text an OCR/vision pass pulled off the bill image.
    # These are the prompt-injection attack surface -- see
    # app/services/injection_detection.py. Real claims populate these
    # from the actual submission; synthetic claims populate them from a
    # templated sentence (see synthetic_data.py::_generate_description),
    # optionally carrying an injected payload.
    incident_description_text: str = ""
    raw_bill_ocr_text: str = ""

    # Error-Level-Analysis score for the claim photo (0-1ish, higher =
    # more compression-artifact inconsistency, a weak signal for
    # AI-edited/spliced images). Only ever populated for real uploaded
    # photos -- see image_utils.compute_ela_score -- because producing a
    # meaningful synthetic ELA score would mean fabricating the exact
    # kind of evidence this check is supposed to independently verify.
    # None for every synthetic claim, which means this check is honestly
    # untested by generate_dataset()/train_fusion_model.py; it needs a
    # hand-collected set of real AI-edited photos to validate, same
    # caveat as adversarial_examples.py.
    ela_score: float | None = None

    # Ground truth, only ever populated by the synthetic generator for
    # training/evaluation. Real claims arriving through the API leave
    # this as None -- the whole point is that the system doesn't know it.
    injected_inconsistency: InconsistencyType | None = None

    @property
    def bill_total_inr(self) -> int:
        return sum(item.cost_inr for item in self.bill_line_items)
