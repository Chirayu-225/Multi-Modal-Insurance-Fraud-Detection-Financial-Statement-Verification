"""
Reports the fourth results table: how well the input-level injection
detector (app/services/injection_detection.py) does at telling apart
clean claims from claims carrying an injection payload.

Scope, stated honestly: this measures DETECTION -- does the heuristic
notice the payload is there -- not ATTACK SUCCESS in the fullest sense
the review meant (does the payload actually change what a live vision
LLM extracts or concludes). That second question can only be answered
by running claims through GeminiVisionClient with a real API key, since
MockVisionClient's output is derived from a file hash and literally
cannot be influenced by text content -- there's no reasoning surface
for an injection to exploit in mock mode. If you get a Gemini key, the
natural extension is: submit N claims with a payload embedded in the
description/bill text and N without, through the real API, and measure
how often the returned JSON itself changes in a way consistent with the
injected instruction (e.g. severity always comes back "minor",
inconsistencies always come back empty). That is a straightforward
extension of this script, not a different one -- swap generate_dataset's
output for real API calls and compare the same way.

Run with: python test_prompt_injection_defense.py
"""
from __future__ import annotations

from app.ml.domain import InconsistencyType
from app.ml.injection_payloads import INJECTION_PAYLOADS, embed_in_description
from app.ml.synthetic_data import FRAUD_INCONSISTENCY_TYPES, generate_dataset
from app.services.injection_detection import detect


def main() -> None:
    print("=" * 78)
    print("1. Detection recall: claims WITH an injection payload")
    print("=" * 78)
    injected_claims = generate_dataset(300, inconsistency_rate=1.0, seed=500, inconsistency_types=[InconsistencyType.PROMPT_INJECTION])
    detected = sum(1 for c in injected_claims if detect(c.incident_description_text, c.raw_bill_ocr_text) is not None)
    recall = detected / len(injected_claims)
    print(f"  {detected}/{len(injected_claims)} injected claims detected -> recall = {recall:.0%}")

    print()
    print("=" * 78)
    print("2. False-positive rate: clean claims, no payload, normal descriptions")
    print("=" * 78)
    clean_claims = generate_dataset(1000, inconsistency_rate=0.0, seed=501)
    false_positives = sum(1 for c in clean_claims if detect(c.incident_description_text, c.raw_bill_ocr_text) is not None)
    fpr = false_positives / len(clean_claims)
    print(f"  {false_positives}/{len(clean_claims)} clean claims incorrectly flagged -> FPR = {fpr:.1%}")
    if fpr == 0.0:
        print("  (0% here reflects the limited variety of the templated descriptions in synthetic_data.py --")
        print("   real claimant language is far more varied. Treat this as a floor, not a production estimate,")
        print("   until you test against a larger sample of real or more diverse descriptions.)")

    print()
    print("=" * 78)
    print("3. Per-payload breakdown (does the detector catch every phrasing equally?)")
    print("=" * 78)
    for payload in INJECTION_PAYLOADS:
        test_description = embed_in_description("My car was damaged in a minor incident.", payload)
        hit = detect(test_description, "")
        status = "DETECTED" if hit else "MISSED"
        print(f"  [{status:8s}] {payload[:70]}{'...' if len(payload) > 70 else ''}")

    print()
    print("=" * 78)
    print("4. Sanity check: fraud-type claims (no injection) never trip this detector")
    print("=" * 78)
    fraud_claims = generate_dataset(500, inconsistency_rate=1.0, seed=502, inconsistency_types=FRAUD_INCONSISTENCY_TYPES)
    cross_contamination = sum(1 for c in fraud_claims if detect(c.incident_description_text, c.raw_bill_ocr_text) is not None)
    print(f"  {cross_contamination}/{len(fraud_claims)} ordinary fraud-type claims incorrectly flagged as injection attempts")

    print()
    print("Honest summary, two caveats stacked:")
    print("1. This validates the input-scanning layer only -- it does not, and")
    print("   currently cannot, prove the system is safe against an injection that a")
    print("   live vision LLM actually falls for. That needs a real API key and the")
    print("   extension described in this file's module docstring.")
    print("2. The 100% recall above is partly circular: SUSPICIOUS_PATTERNS in")
    print("   injection_detection.py was written by looking at these same six")
    print("   payloads. A meaningful next step is having someone else (or you, a")
    print("   week later) write 10-15 NEW phrasings the detector has never seen and")
    print("   re-running this -- that recall number is the one worth reporting.")


if __name__ == "__main__":
    main()
