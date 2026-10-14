"""One entry point for the cross-verification engine: pick the rule pack
by case type. Callers that already know their domain (pipeline.py,
underwriting_pipeline.py) can keep calling their pack directly; this is
for code that handles both, such as evaluation scripts."""
from __future__ import annotations

from app.core.case import CaseType
from app.ml.domain import ClaimEvidence
from app.ml.underwriting_domain import UnderwritingEvidence
from app.services.rules import claims, underwriting


def evaluate_case(case_type: CaseType, evidence, **kwargs):
    if case_type == CaseType.MOTOR_CLAIM:
        if not isinstance(evidence, ClaimEvidence):
            raise TypeError("motor_claim cases need a ClaimEvidence")
        return claims.evaluate(evidence, **kwargs)
    if case_type == CaseType.UNDERWRITING:
        if not isinstance(evidence, UnderwritingEvidence):
            raise TypeError("underwriting cases need an UnderwritingEvidence")
        return underwriting.evaluate_underwriting(evidence, **kwargs)
    raise ValueError(f"no rule pack for case type {case_type!r}")


PACKS = {
    CaseType.MOTOR_CLAIM: (claims.PACK_NAME, claims.PACK_VERSION),
    CaseType.UNDERWRITING: (underwriting.PACK_NAME, underwriting.PACK_VERSION),
}
