"""
Commercial-underwriting rule pack.

Same contract as the claims engine: each check returns a flag (type,
severity, plain-language explanation) or None, and the flags combine
into one risk score with noisy-OR. Severities are hand-set constants for
now, exactly like the claims pack; a learned fusion model can replace
the combination later and be compared against this baseline.

Status: real and unit-tested. Inputs that come from models not yet built
(tamper_probability, market_stress_score) are simply None until those
models exist, and the corresponding checks stay silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.underwriting_domain import (
    FIGURE_TOLERANCE,
    MARKET_STRESS_THRESHOLD,
    TAMPER_PROBABILITY_THRESHOLD,
    AuditorOpinion,
    UnderwritingEvidence,
    UnderwritingInconsistencyType as UT,
)
from app.services.gstin import validate_gstin
from app.services.rules.scoring import noisy_or

PACK_NAME = "commercial_underwriting"
PACK_VERSION = "uw_rules_v1"

# Same meaning as FAST_TRACK_RISK_THRESHOLD for claims. Placeholder value:
# re-derive it with a threshold sweep (see underwriting_benchmark.py)
# before quoting it anywhere.
UW_FAST_TRACK_RISK_THRESHOLD = 0.30


@dataclass
class UnderwritingFlag:
    type: UT
    severity: float
    message: str


@dataclass
class UnderwritingResult:
    case_id: str
    risk_score: float
    flags: list[UnderwritingFlag] = field(default_factory=list)
    model_version: str = PACK_VERSION

    @property
    def triggered_types(self) -> set[UT]:
        return {f.type for f in self.flags}

    @property
    def fast_track(self) -> bool:
        return is_uw_fast_track(self.risk_score)


def is_uw_fast_track(risk_score: float) -> bool:
    return risk_score < UW_FAST_TRACK_RISK_THRESHOLD


def _check_figure_mismatch(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    worst: tuple[float, str] | None = None
    for sub in ev.submitted_figures:
        aud = ev.audited(sub.metric)
        if aud is None or aud.fiscal_year != sub.fiscal_year:
            continue
        denom = max(abs(aud.value_inr_crore), 1e-9)
        rel = abs(sub.value_inr_crore - aud.value_inr_crore) / denom
        if rel > FIGURE_TOLERANCE and (worst is None or rel > worst[0]):
            worst = (
                rel,
                f"Submitted {sub.metric.value} for {sub.fiscal_year} is INR {sub.value_inr_crore:,.2f} crore "
                f"({sub.source}), but the audited annual report states INR {aud.value_inr_crore:,.2f} crore "
                f"({aud.source}) -- a {rel:.0%} difference, above the {FIGURE_TOLERANCE:.0%} tolerance.",
            )
    if worst is None:
        return None
    severity = 0.85 if worst[0] > 0.10 else 0.6
    return UnderwritingFlag(UT.FIGURE_MISMATCH, severity, worst[1])


def _check_period_mismatch(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    if ev.audited_fiscal_year and ev.submitted_fiscal_year != ev.audited_fiscal_year:
        return UnderwritingFlag(
            UT.PERIOD_MISMATCH,
            0.5,
            f"Submitted statements are for {ev.submitted_fiscal_year}, but the latest audited annual report on file "
            f"is for {ev.audited_fiscal_year}, so the submitted figures cannot be verified against an audited source.",
        )
    return None


def _check_gstin(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    if not ev.company_gstin:
        return None  # not provided is a data-completeness question, not an inconsistency
    ok, reason = validate_gstin(ev.company_gstin)
    if not ok:
        return UnderwritingFlag(UT.INVALID_GSTIN, 0.7, f"Company GSTIN '{ev.company_gstin}' is invalid: {reason}.")
    return None


def _check_gstin_state(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    if not ev.company_gstin or not ev.registered_state_code:
        return None
    ok, _ = validate_gstin(ev.company_gstin)
    if not ok:
        return None  # already reported by _check_gstin; don't double-count
    gst_state = ev.company_gstin.strip()[:2]
    if gst_state != ev.registered_state_code.zfill(2):
        return UnderwritingFlag(
            UT.GSTIN_STATE_MISMATCH,
            0.4,
            f"GSTIN state code {gst_state} does not match the registered-office state code "
            f"{ev.registered_state_code.zfill(2)}. This can be legitimate (a branch registration), so treat it as a "
            f"prompt to confirm which registration the submission belongs to.",
        )
    return None


def _check_auditor_opinion(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    reasons = []
    severity = 0.0
    if ev.auditor_opinion in (AuditorOpinion.QUALIFIED, AuditorOpinion.ADVERSE, AuditorOpinion.DISCLAIMER):
        reasons.append(f"the auditor's opinion is {ev.auditor_opinion.value}")
        severity = 0.85 if ev.auditor_opinion != AuditorOpinion.QUALIFIED else 0.6
    if ev.going_concern_note:
        reasons.append("the annual report carries a going-concern note")
        severity = max(severity, 0.7)
    if not reasons:
        return None
    return UnderwritingFlag(UT.QUALIFIED_OPINION, severity, "The audited annual report needs review: " + " and ".join(reasons) + ".")


def _check_tampered_statement(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    if ev.tamper_probability is None or ev.tamper_probability < TAMPER_PROBABILITY_THRESHOLD:
        return None
    return UnderwritingFlag(
        UT.TAMPERED_STATEMENT,
        0.8,
        f"The document-forensics model rates the submitted statement {ev.tamper_probability:.0%} likely to contain "
        f"edited regions. Check the heatmap before relying on any figure from it.",
    )


def _check_market_stress(ev: UnderwritingEvidence) -> UnderwritingFlag | None:
    if ev.market_stress_score is None or ev.market_stress_score < MARKET_STRESS_THRESHOLD:
        return None
    # Deliberately low severity: market signals are noisy context and must
    # never be enough on their own to push a case over the review line.
    return UnderwritingFlag(
        UT.MARKET_STRESS,
        0.2,
        f"Market signals around the statement date indicate elevated stress (score {ev.market_stress_score:.2f}). "
        f"Weak context only, not evidence of misrepresentation.",
    )


def evaluate_underwriting(ev: UnderwritingEvidence, extra_flags: list[UnderwritingFlag] | None = None) -> UnderwritingResult:
    checks = [
        _check_figure_mismatch(ev),
        _check_period_mismatch(ev),
        _check_gstin(ev),
        _check_gstin_state(ev),
        _check_auditor_opinion(ev),
        _check_tampered_statement(ev),
        _check_market_stress(ev),
    ]
    flags = [f for f in checks if f is not None] + (extra_flags or [])
    return UnderwritingResult(case_id=ev.case_id, risk_score=noisy_or(f.severity for f in flags), flags=flags)
