"""The Case/Assertion layer the merge adds above ClaimEvidence."""
import pytest

from app.core.case import (
    Assertion,
    Case,
    CaseType,
    ExtractionPath,
    Sensitivity,
    SourceRef,
    assertions_from_claim_evidence,
)
from app.ml.synthetic_data import generate_dataset


def _a(case_id, value, document, attribute="revenue"):
    return Assertion(case_id, "company", attribute, value, SourceRef(document), ExtractionPath.MANUAL,
                     Sensitivity.PUBLIC, period="FY2025")


def test_case_rejects_assertion_from_another_case():
    case = Case("C1", CaseType.UNDERWRITING)
    with pytest.raises(ValueError):
        case.add(_a("C2", 10.0, "statement"))


def test_conflicting_pairs_only_across_documents_and_same_key():
    case = Case("C1", CaseType.UNDERWRITING)
    case.add(_a("C1", 120.0, "submitted_statement"))
    case.add(_a("C1", 96.0, "annual_report"))
    case.add(_a("C1", 96.0, "annual_report_copy"))      # agrees with the report: no conflict with it
    case.add(_a("C1", 5.0, "submitted_statement", attribute="net_profit"))  # different key
    pairs = case.conflicting_pairs()
    assert len(pairs) == 2
    assert all({p[0].value, p[1].value} == {120.0, 96.0} for p in pairs)


def test_synthetic_claim_evidence_maps_to_synthetic_assertions():
    claim = generate_dataset(5, inconsistency_rate=1.0, seed=3)[0]
    assertions = assertions_from_claim_evidence(claim)
    assert assertions, "expected assertions from a claim"
    assert all(a.sensitivity == Sensitivity.SYNTHETIC for a in assertions)
    attrs = {(a.entity, a.attribute) for a in assertions}
    assert ("vehicle_photo", "damage_location") in attrs
    assert ("repair_bill", "total") in attrs
