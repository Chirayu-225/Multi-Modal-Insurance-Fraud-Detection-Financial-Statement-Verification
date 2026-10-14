import pytest

from app.core.case import Sensitivity
from app.services.privacy_guard import RestrictedContentError, prepare_for_hosted, redact


def test_restricted_content_is_refused():
    with pytest.raises(RestrictedContentError):
        prepare_for_hosted("anything", Sensitivity.RESTRICTED)


def test_public_and_synthetic_content_pass_the_gate():
    assert prepare_for_hosted("Revenue rose 12%.", Sensitivity.PUBLIC).text == "Revenue rose 12%."
    assert prepare_for_hosted("hello", Sensitivity.SYNTHETIC).text == "hello"


def test_identifiers_are_redacted():
    text = ("GSTIN 27AAPFU0939F1ZV, PAN ABCDE1234F, call +91 9876543210 or mail a.b@example.com, "
            "Aadhaar 1234 5678 9012, vehicle KA 01 AB 1234, account 123456789012345")
    r = redact(text)
    for label in ["GSTIN", "PAN", "PHONE", "EMAIL", "AADHAAR", "VEHICLE_REG", "ACCOUNT_NO"]:
        assert f"[{label}]" in r.text, label
    assert "27AAPFU0939F1ZV" not in r.text and "ABCDE1234F" not in r.text


def test_financial_figures_survive_redaction():
    r = redact("Revenue from operations was 1,234.56 crore in FY2025, up from 1,100.00.")
    assert not r.redacted_anything
