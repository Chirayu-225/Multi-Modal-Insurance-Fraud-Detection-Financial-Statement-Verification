"""
Regression coverage for the language-fairness/security finding in
fairness_language_analysis.py. Smaller sample sizes than the reporting
script (for speed), but the same three questions:

  1. clean claims in any supported language stay clean (no false
     positives from the injection detector)
  2. an English-phrased payload is still caught regardless of what
     language surrounds it
  3. a *translated* payload is not caught -- this is the real limitation
     the reporting script measures at scale; asserting it here keeps this
     from silently regressing into a false claim of full recall if
     SUSPICIOUS_PATTERNS is ever extended without updating the README.
"""
from app.ml.injection_payloads import INJECTION_PAYLOADS, TRANSLATED_INJECTION_PAYLOADS, embed_in_description
from app.ml.synthetic_data import LANGUAGE_DESCRIPTION_TEMPLATES, generate_dataset
from app.services import injection_detection

LANGUAGES = list(LANGUAGE_DESCRIPTION_TEMPLATES.keys())


def test_clean_claims_have_zero_false_positives_in_every_supported_language():
    for lang in LANGUAGES:
        claims = generate_dataset(100, inconsistency_rate=0.0, seed=1, language=lang)
        for c in claims:
            assert injection_detection.detect(c.incident_description_text, c.raw_bill_ocr_text) is None, (
                f"false positive on a clean {lang} description: {c.incident_description_text!r}"
            )


def test_english_payload_is_detected_regardless_of_surrounding_language():
    for lang in LANGUAGES:
        claims = generate_dataset(20, inconsistency_rate=0.0, seed=2, language=lang)
        for i, c in enumerate(claims):
            payload = INJECTION_PAYLOADS[i % len(INJECTION_PAYLOADS)]
            injected = embed_in_description(c.incident_description_text, payload)
            assert injection_detection.detect(injected, c.raw_bill_ocr_text) is not None, (
                f"missed an English payload embedded in a {lang} description"
            )


def test_translated_payload_currently_evades_the_english_only_detector():
    """Documents a real, honest limitation rather than a bug: the
    detector's patterns are English regex, so a translated attack is not
    caught. If this test starts failing because someone added
    multilingual patterns, that's good news -- update this test (and the
    README's fairness/security section) to match, don't just delete it."""
    for lang, translated_payload in TRANSLATED_INJECTION_PAYLOADS.items():
        claims = generate_dataset(20, inconsistency_rate=0.0, seed=3, language=lang)
        for c in claims:
            injected = embed_in_description(c.incident_description_text, translated_payload)
            assert injection_detection.detect(injected, c.raw_bill_ocr_text) is None, (
                f"translated {lang} payload was unexpectedly caught -- update the README's honest-limitation claim"
            )
