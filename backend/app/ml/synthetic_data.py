"""
Synthetic claim + injected-inconsistency generator.

Why this exists (the leakage warning from review, taken seriously):
a model or rules engine that is both trained and evaluated on claims
produced by the *same* generator can trivially learn the generator's
quirks instead of anything about real inconsistency. This module is
built so that risk is visible and testable rather than hidden:

  - every generated claim is tagged with exactly which InconsistencyType
    (if any) was injected, so you can hold entire types out of training
    and test on the ones the model never saw (see train_fusion_model.py)
  - `generate_dataset` produces a class-balanced set, good for training
    and for "how well can this be learned at all" metrics
  - `generate_realistic_dataset` produces a set at a low, realistic
    fraud/inconsistency rate (default 5%), because precision on a 50/50
    set is a very different number from precision at a realistic base
    rate, and both should be reported

This generator produces *structured evidence* (damage location,
severity, bill line items, a photo hash, an EXIF-style capture date) --
not pixels. That is a deliberate scope cut: it lets the consistency
engine and the fusion model be built and evaluated end-to-end today,
independent of whether the damage-location/severity signal eventually
comes from a real fine-tuned classifier, from Gemini, or (as here) from
a synthetic label standing in for both while that part is built out.
See app/services/vision.py for where the real extraction plugs in later
without this module, or the consistency engine, needing to change.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import replace
from datetime import date, timedelta

from app.ml.domain import (
    BASE_REPAIR_COST_INR,
    EXPECTED_LOCATIONS,
    POLICY_EXCLUDED_CATEGORIES,
    REPAIR_GST_RATE,
    SEVERITY_COST_MULTIPLIER,
    BillLineItem,
    ClaimEvidence,
    DamageLocation,
    IncidentType,
    InconsistencyType,
    Severity,
    VehicleSegment,
)
from app.ml.injection_payloads import INJECTION_PAYLOADS, embed_as_hidden_bill_text, embed_in_description
from app.services.gstin import generate_plausible_gstin

_ALL_LOCATIONS = list(DamageLocation)
_ALL_INCIDENTS = list(IncidentType)
_ALL_SEGMENTS = list(VehicleSegment)
_ALL_SEVERITIES = list(Severity)

# The original five fraud-signal types -- this is the default set used
# whenever `inconsistency_types` isn't given explicitly. PROMPT_INJECTION
# and EXTRACTION_DIVERGENCE are deliberately *not* in this default: the
# fusion model's feature set (train_fusion_model.py::featurize) has no
# feature that could ever catch them (no description-text scanning is
# fed into it), so silently mixing them into "realistic base rate, all
# types" would quietly manufacture false negatives that look like a
# fusion-model failure but are really just an unlabeled blind spot. Ask
# for them explicitly (inconsistency_types=[InconsistencyType.PROMPT_INJECTION])
# when that's what you mean to test -- see test_prompt_injection_defense.py.
FRAUD_INCONSISTENCY_TYPES: list[InconsistencyType] = [
    InconsistencyType.WRONG_PANEL,
    InconsistencyType.INFLATED_BILL,
    InconsistencyType.EXCLUDED_ITEM,
    InconsistencyType.REUSED_PHOTO,
    InconsistencyType.TIMESTAMP_MISMATCH,
]

_DESCRIPTION_TEMPLATES: dict[IncidentType, list[str]] = {
    IncidentType.FRONT_COLLISION: [
        "I was driving and the car in front of me braked suddenly, so I hit the rear of their vehicle with my front end.",
        "A pedestrian ran out and I braked hard, clipping a pole with the front of the car.",
    ],
    IncidentType.REAR_COLLISION: [
        "I was stopped at a red light when another vehicle rear-ended me.",
        "Someone reversed into the back of my car in a parking lot.",
    ],
    IncidentType.LEFT_SIDE_COLLISION: [
        "A car merged into my lane from the left and hit the side of my vehicle.",
        "I scraped the left side of the car against a gate while pulling out of my driveway.",
    ],
    IncidentType.RIGHT_SIDE_COLLISION: [
        "A two-wheeler clipped the right side of my car while overtaking.",
        "I misjudged a turn and hit the right side against a concrete barrier.",
    ],
    IncidentType.PARKING_SCRAPE: [
        "Someone scraped my car while it was parked outside my apartment.",
        "I found a scrape on the car after returning to the parking lot; no note was left.",
    ],
}

# Same five incident types, in Hindi, Kannada and Tamil (native script) and
# a Bengaluru-style code-mixed register (Kanglish/Hinglish, Latin script).
# These exist for one reason: fairness_language_analysis.py measures
# whether the prompt-injection detector's false-positive rate on genuine
# claimant language differs by the language a claimant happens to write
# in -- an honest question, since injection_detection.py's patterns are
# English regex, and a system that works fine on English but misfires on
# Kannada would be flagging real people based on language, not fraud
# signal. Not machine-translated filler -- written as plausible claim
# sentences a person would actually type, same register as the English
# set above.
_DESCRIPTION_TEMPLATES_HINDI: dict[IncidentType, list[str]] = {
    IncidentType.FRONT_COLLISION: [
        "मैं गाड़ी चला रहा था तभी अचानक आगे वाली गाड़ी ने ब्रेक लगाया और मेरी गाड़ी का अगला हिस्सा उससे टकरा गया।",
        "एक पैदल यात्री अचानक सामने आ गया, मैंने जोर से ब्रेक लगाया और गाड़ी का अगला भाग एक खंभे से टकरा गया।",
    ],
    IncidentType.REAR_COLLISION: [
        "मैं लाल बत्ती पर गाड़ी रोककर खड़ा था तभी पीछे से किसी गाड़ी ने टक्कर मार दी।",
        "पार्किंग में किसी ने रिवर्स करते हुए मेरी गाड़ी के पिछले हिस्से से टक्कर मार दी।",
    ],
    IncidentType.LEFT_SIDE_COLLISION: [
        "बाईं ओर से एक गाड़ी मेरी लेन में आ गई और मेरी गाड़ी के साइड से टकरा गई।",
        "गैरेज से गाड़ी निकालते समय गेट से गाड़ी का बायाँ हिस्सा रगड़ गया।",
    ],
    IncidentType.RIGHT_SIDE_COLLISION: [
        "ओवरटेक करते समय एक दोपहिया वाहन ने मेरी गाड़ी के दाहिने हिस्से को छू दिया।",
        "मोड़ लेते समय अंदाजा गलत हो गया और गाड़ी का दायाँ हिस्सा एक कंक्रीट बैरियर से टकरा गया।",
    ],
    IncidentType.PARKING_SCRAPE: [
        "मेरी गाड़ी घर के बाहर खड़ी थी तभी किसी ने उसे रगड़ दिया।",
        "पार्किंग से वापस आने पर गाड़ी पर खरोंच मिली; कोई नोट नहीं छोड़ा गया था।",
    ],
}
_DESCRIPTION_TEMPLATES_KANNADA: dict[IncidentType, list[str]] = {
    IncidentType.FRONT_COLLISION: [
        "ನಾನು ಕಾರು ಓಡಿಸುತ್ತಿದ್ದಾಗ ಮುಂದಿನ ವಾಹನ ಇದ್ದಕ್ಕಿದ್ದಂತೆ ಬ್ರೇಕ್ ಹಾಕಿತು, ಇದರಿಂದ ನನ್ನ ಕಾರಿನ ಮುಂಭಾಗ ಅದಕ್ಕೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು.",
        "ಒಬ್ಬ ಪಾದಚಾರಿ ಇದ್ದಕ್ಕಿದ್ದಂತೆ ಎದುರಿಗೆ ಬಂದನು, ನಾನು ಜೋರಾಗಿ ಬ್ರೇಕ್ ಹಾಕಿದೆ ಮತ್ತು ಕಾರಿನ ಮುಂಭಾಗ ಒಂದು ಕಂಬಕ್ಕೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು.",
    ],
    IncidentType.REAR_COLLISION: [
        "ನಾನು ಸಿಗ್ನಲ್‌ನಲ್ಲಿ ಕಾರು ನಿಲ್ಲಿಸಿದ್ದಾಗ ಹಿಂದಿನಿಂದ ಬಂದ ವಾಹನ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು.",
        "ಪಾರ್ಕಿಂಗ್‌ನಲ್ಲಿ ಯಾರೋ ರಿವರ್ಸ್ ಮಾಡುವಾಗ ನನ್ನ ಕಾರಿನ ಹಿಂಭಾಗಕ್ಕೆ ಡಿಕ್ಕಿ ಹೊಡೆದರು.",
    ],
    IncidentType.LEFT_SIDE_COLLISION: [
        "ಎಡಬದಿಯಿಂದ ಒಂದು ಕಾರು ನನ್ನ ಲೇನ್‌ಗೆ ಬಂದು ನನ್ನ ವಾಹನದ ಬದಿಗೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು.",
        "ಗ್ಯಾರೇಜ್‌ನಿಂದ ಕಾರು ಹೊರಗೆ ತೆಗೆಯುವಾಗ ಗೇಟಿಗೆ ಕಾರಿನ ಎಡಭಾಗ ಉಜ್ಜಿತು.",
    ],
    IncidentType.RIGHT_SIDE_COLLISION: [
        "ಓವರ್‌ಟೇಕ್ ಮಾಡುವಾಗ ಒಂದು ದ್ವಿಚಕ್ರ ವಾಹನ ನನ್ನ ಕಾರಿನ ಬಲಭಾಗಕ್ಕೆ ತಾಗಿತು.",
        "ತಿರುವು ತೆಗೆದುಕೊಳ್ಳುವಾಗ ಅಂದಾಜು ತಪ್ಪಿ ಕಾರಿನ ಬಲಭಾಗ ಒಂದು ಕಾಂಕ್ರೀಟ್ ತಡೆಗೋಡೆಗೆ ಡಿಕ್ಕಿ ಹೊಡೆಯಿತು.",
    ],
    IncidentType.PARKING_SCRAPE: [
        "ನನ್ನ ಕಾರು ಅಪಾರ್ಟ್‌ಮೆಂಟ್ ಹೊರಗೆ ನಿಲ್ಲಿಸಿದ್ದಾಗ ಯಾರೋ ಉಜ್ಜಿ ಹೋಗಿದ್ದಾರೆ.",
        "ಪಾರ್ಕಿಂಗ್‌ನಿಂದ ಹಿಂತಿರುಗಿ ಬಂದಾಗ ಕಾರಿನ ಮೇಲೆ ಗೀರು ಕಂಡುಬಂತು; ಯಾವುದೇ ಚೀಟಿ ಬಿಟ್ಟಿರಲಿಲ್ಲ.",
    ],
}
_DESCRIPTION_TEMPLATES_TAMIL: dict[IncidentType, list[str]] = {
    IncidentType.FRONT_COLLISION: [
        "நான் காரை ஓட்டிக்கொண்டிருந்தபோது முன்னால் இருந்த வாகனம் திடீரென பிரேக் போட்டது, அதனால் என் காரின் முன்பகுதி அதில் மோதியது.",
        "ஒரு பாதசாரி திடீரென முன்னே வந்தார், நான் பலமாக பிரேக் போட்டேன், காரின் முன்பகுதி ஒரு கம்பத்தில் மோதியது.",
    ],
    IncidentType.REAR_COLLISION: [
        "நான் சிக்னலில் காரை நிறுத்தியிருந்தபோது பின்னால் இருந்து ஒரு வாகனம் மோதியது.",
        "பார்க்கிங்கில் யாரோ ரிவர்ஸ் எடுக்கும்போது என் காரின் பின்பகுதியில் மோதினார்கள்.",
    ],
    IncidentType.LEFT_SIDE_COLLISION: [
        "இடதுபுறத்திலிருந்து ஒரு கார் என் லேனுக்குள் வந்து என் வாகனத்தின் பக்கவாட்டில் மோதியது.",
        "கராஜிலிருந்து காரை எடுக்கும்போது கேட்டில் காரின் இடதுபுறம் உரசியது.",
    ],
    IncidentType.RIGHT_SIDE_COLLISION: [
        "ஓவர்டேக் செய்யும்போது ஒரு இருசக்கர வாகனம் என் காரின் வலதுபுறத்தில் உரசியது.",
        "திருப்பம் எடுக்கும்போது தவறாக மதிப்பிட்டு காரின் வலதுபுறம் ஒரு கான்கிரீட் தடுப்பில் மோதியது.",
    ],
    IncidentType.PARKING_SCRAPE: [
        "என் கார் அபார்ட்மென்ட் வெளியே நிறுத்தப்பட்டிருந்தபோது யாரோ உரசிவிட்டனர்.",
        "பார்க்கிங்கில் இருந்து திரும்பி வந்தபோது காரில் ஒரு கீறல் இருந்தது; எந்த குறிப்பும் விடப்படவில்லை.",
    ],
}
_DESCRIPTION_TEMPLATES_CODE_MIXED: dict[IncidentType, list[str]] = {
    IncidentType.FRONT_COLLISION: [
        "I was driving normal speed pe, suddenly front vehicle ne brake maara achanak, so car ka front side thoda damage ho gaya.",
        "Ondu pedestrian suddenly road madhe bandru, I braked hard aur front bumper pole ge hit aaytu.",
    ],
    IncidentType.REAR_COLLISION: [
        "Signal beside nalli car niltisiddaga, back side inda ondu vehicle bandu hit maadthu.",
        "Parking lot alli reverse taking madtiruvaga yaaro back se gaadi ko hit kar diya.",
    ],
    IncidentType.LEFT_SIDE_COLLISION: [
        "Ondu car left side inda lane ge bandittu, en car side ge tagithu.",
        "Gate hattira car nikaltiruvaaga left side thoda scratch aaytu, achanak ho gaya.",
    ],
    IncidentType.RIGHT_SIDE_COLLISION: [
        "Overtake maadtiruvaaga ondu bike right side ge tagithu, halge thoda dent aaytu.",
        "Turn tegedukolluvaaga andaaz galat aaythu, right side ondu concrete barrier ge hit aaytu.",
    ],
    IncidentType.PARKING_SCRAPE: [
        "Car apartment horage niltiddaaga yaaro scratch madidru, note ondhu bidalilla.",
        "Parking inda vapas bandaaga car mele scratch kanbandtu, yaar da antha gottilla.",
    ],
}

LANGUAGE_DESCRIPTION_TEMPLATES: dict[str, dict[IncidentType, list[str]]] = {
    "english": _DESCRIPTION_TEMPLATES,
    "hindi": _DESCRIPTION_TEMPLATES_HINDI,
    "kannada": _DESCRIPTION_TEMPLATES_KANNADA,
    "tamil": _DESCRIPTION_TEMPLATES_TAMIL,
    "code_mixed": _DESCRIPTION_TEMPLATES_CODE_MIXED,
}


def _generate_description(incident_type: IncidentType, rng: random.Random, language: str = "english") -> str:
    return rng.choice(LANGUAGE_DESCRIPTION_TEMPLATES[language][incident_type])


def _generate_bill_ocr_text(line_items: list[BillLineItem]) -> str:
    """A plausible plain-text rendering of what OCR would pull off a
    bill image listing these line items -- used as the raw text the
    injection detector and the cross-check scan, not shown to the user
    directly."""
    lines = [f"{item.category.replace('_', ' ').title()} ... Rs. {item.cost_inr:,}" for item in line_items]
    return "\n".join(lines)


def _random_photo_hash(rng: random.Random) -> str:
    """Stand-in for a real perceptual hash (imagehash produces a 64-bit
    hash rendered as 16 hex chars) -- random and unique per claim unless
    a REUSED_PHOTO injection deliberately copies one."""
    return f"{rng.getrandbits(64):016x}"


def _clean_claim(rng: random.Random, claim_id: str, language: str = "english") -> ClaimEvidence:
    incident_type = rng.choice(_ALL_INCIDENTS)
    # sorted(), not list() -- EXPECTED_LOCATIONS[...] is a set, and
    # DamageLocation is a (str, Enum) mixin, so its hash is str's hash,
    # which CPython randomizes per process (PYTHONHASHSEED) unless it's
    # pinned. list(some_set_of_these) would silently reorder on every
    # run, making rng.choice() pick a different element for the *same*
    # seed depending on which process happened to run it -- the exact bug
    # that made this script's numbers drift between two runs of
    # identical, unedited code (not just after a real upstream edit, as
    # the old caveat here used to claim). sorted() compares the
    # underlying string values directly, which is hash-seed-independent,
    # so the same seed now reproduces the exact same dataset everywhere.
    location = rng.choice(sorted(EXPECTED_LOCATIONS[incident_type]))
    severity = rng.choice(_ALL_SEVERITIES)
    segment = rng.choice(_ALL_SEGMENTS)
    incident_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 240))
    capture_date = incident_date + timedelta(days=rng.randint(0, 2))  # photo taken promptly, as expected

    base_cost = BASE_REPAIR_COST_INR[location][segment]
    cost = round(base_cost * SEVERITY_COST_MULTIPLIER[severity] * rng.uniform(0.9, 1.1))
    line_items = [BillLineItem(category=f"{location.value}_repair", cost_inr=cost, claimed_gst_rate=REPAIR_GST_RATE)]

    return ClaimEvidence(
        claim_id=claim_id,
        vehicle_segment=segment,
        incident_date=incident_date,
        described_incident_type=incident_type,
        photo_damage_location=location,
        photo_damage_severity=severity,
        photo_hash=_random_photo_hash(rng),
        photo_capture_date=capture_date,
        bill_line_items=line_items,
        policy_excluded_categories=set(POLICY_EXCLUDED_CATEGORIES),
        garage_gstin=generate_plausible_gstin(rng),
        incident_description_text=_generate_description(incident_type, rng, language),
        raw_bill_ocr_text=_generate_bill_ocr_text(line_items),
        injected_inconsistency=InconsistencyType.NONE,
    )


def _inject(claim: ClaimEvidence, kind: InconsistencyType, rng: random.Random, hash_pool: list[str]) -> ClaimEvidence:
    if kind == InconsistencyType.WRONG_PANEL:
        expected = EXPECTED_LOCATIONS[claim.described_incident_type]
        candidates = [loc for loc in _ALL_LOCATIONS if loc not in expected]
        if not candidates:
            # e.g. PARKING_SCRAPE is defined as "any panel is plausible" --
            # there's no such thing as a wrong panel for it, so this
            # injection type doesn't apply; leave the claim clean instead
            # of crashing or silently mislabeling it.
            return replace(claim, injected_inconsistency=InconsistencyType.NONE)
        new_location = rng.choice(candidates)
        return replace(claim, photo_damage_location=new_location, injected_inconsistency=kind)

    if kind == InconsistencyType.INFLATED_BILL:
        factor = rng.uniform(2.5, 4.0)
        inflated_items = [
            BillLineItem(category=item.category, cost_inr=round(item.cost_inr * factor))
            for item in claim.bill_line_items
        ]
        return replace(claim, bill_line_items=inflated_items, injected_inconsistency=kind)

    if kind == InconsistencyType.EXCLUDED_ITEM:
        excluded_category = rng.choice(sorted(claim.policy_excluded_categories))
        extra_item = BillLineItem(category=excluded_category, cost_inr=rng.randint(1500, 6000))
        return replace(claim, bill_line_items=[*claim.bill_line_items, extra_item], injected_inconsistency=kind)

    if kind == InconsistencyType.REUSED_PHOTO:
        if hash_pool:
            reused_hash = rng.choice(hash_pool)
            return replace(claim, photo_hash=reused_hash, injected_inconsistency=kind)
        # no pool yet (first claim in the batch) -- fall back to a clean claim
        return replace(claim, injected_inconsistency=InconsistencyType.NONE)

    if kind == InconsistencyType.TIMESTAMP_MISMATCH:
        if rng.random() < 0.5:
            # photo "taken" before the incident even happened
            bad_date = claim.incident_date - timedelta(days=rng.randint(1, 10))
        else:
            # photo surfaces suspiciously long after the incident
            bad_date = claim.incident_date + timedelta(days=rng.randint(30, 90))
        return replace(claim, photo_capture_date=bad_date, injected_inconsistency=kind)

    if kind == InconsistencyType.PROMPT_INJECTION:
        payload = rng.choice(INJECTION_PAYLOADS)
        if rng.random() < 0.5:
            new_description = embed_in_description(claim.incident_description_text, payload)
            return replace(claim, incident_description_text=new_description, injected_inconsistency=kind)
        new_ocr_text = embed_as_hidden_bill_text(claim.raw_bill_ocr_text, payload)
        return replace(claim, raw_bill_ocr_text=new_ocr_text, injected_inconsistency=kind)

    if kind == InconsistencyType.INVALID_GSTIN:
        # Corrupt one character of an otherwise well-formed GSTIN --
        # simulates a mistyped or fabricated number, not a garbled string
        # that would fail on length alone (that's a weaker, less
        # realistic test of the checksum logic specifically).
        valid_gstin = claim.garage_gstin or generate_plausible_gstin(rng)
        position = rng.randint(0, 13)  # corrupt anywhere except forcing the final checksum char specifically
        chars = list(valid_gstin)
        original_char = chars[position]
        replacement_pool = [c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != original_char]
        chars[position] = rng.choice(replacement_pool)
        return replace(claim, garage_gstin="".join(chars), injected_inconsistency=kind)

    if kind == InconsistencyType.GST_RATE_MISMATCH:
        wrong_rate = rng.choice([0.05, 0.12, 0.28])  # other real GST slabs, just not the correct one for this category
        bad_items = [replace(item, claimed_gst_rate=wrong_rate) for item in claim.bill_line_items]
        return replace(claim, bill_line_items=bad_items, injected_inconsistency=kind)

    return claim


def generate_dataset(
    n: int,
    inconsistency_rate: float = 0.5,
    seed: int | None = 42,
    inconsistency_types: list[InconsistencyType] | None = None,
    language: str = "english",
) -> list[ClaimEvidence]:
    """
    Class-balanced-ish synthetic dataset. `inconsistency_rate` is the
    fraction of claims that get *some* injected inconsistency; the rest
    are clean. `inconsistency_types` restricts which types can be
    injected -- this is the knob you use to build a held-out-type split:
    generate the training set with one subset of types, the test set
    with the complementary subset. `language` picks which of
    LANGUAGE_DESCRIPTION_TEMPLATES the free-text incident description is
    drawn from (default "english", unchanged behavior for every existing
    caller) -- see fairness_language_analysis.py for why this exists.
    """
    rng = random.Random(seed)
    types = inconsistency_types or FRAUD_INCONSISTENCY_TYPES
    claims: list[ClaimEvidence] = []
    hash_pool: list[str] = []

    for _ in range(n):
        claim_id = str(uuid.uuid4())
        claim = _clean_claim(rng, claim_id, language)
        if rng.random() < inconsistency_rate and types:
            kind = rng.choice(types)
            claim = _inject(claim, kind, rng, hash_pool)
        claims.append(claim)
        hash_pool.append(claim.photo_hash)

    return claims


def generate_realistic_dataset(
    n: int,
    fraud_rate: float = 0.05,
    seed: int | None = 7,
    inconsistency_types: list[InconsistencyType] | None = None,
    language: str = "english",
) -> list[ClaimEvidence]:
    """Same generator, but at a low base rate, for reporting precision
    the way it will actually look in production rather than on an
    artificially balanced set."""
    return generate_dataset(n, inconsistency_rate=fraud_rate, seed=seed, inconsistency_types=inconsistency_types, language=language)


if __name__ == "__main__":
    sample = generate_dataset(10)
    for c in sample:
        print(c.claim_id[:8], c.injected_inconsistency, c.photo_damage_location, c.bill_total_inr)
