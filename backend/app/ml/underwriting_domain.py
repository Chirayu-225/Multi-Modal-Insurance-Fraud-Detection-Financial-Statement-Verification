"""
Shared vocabulary for the commercial-underwriting domain pack -- the
underwriting counterpart of app/ml/domain.py.

The scenario: a business applies for commercial cover where the insurer's
risk depends on the applicant's financial health (surety, trade credit,
large commercial property). It submits financial statements. The insurer
cross-checks those submitted figures against the company's audited
annual report, its GSTIN, the auditor's opinion, and (for listed firms,
optionally) market signals.

Like domain.py, this one file is imported by the synthetic generator
(ml/underwriting_synthetic.py), the rule pack
(services/rules/underwriting.py) and the API, so they cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FinancialMetric(str, Enum):
    REVENUE = "revenue"
    NET_PROFIT = "net_profit"
    TOTAL_ASSETS = "total_assets"
    TOTAL_DEBT = "total_debt"


class AuditorOpinion(str, Enum):
    UNQUALIFIED = "unqualified"
    QUALIFIED = "qualified"
    ADVERSE = "adverse"
    DISCLAIMER = "disclaimer"


class UnderwritingInconsistencyType(str, Enum):
    """Taxonomy for underwriting cases, same role as InconsistencyType
    for claims: every synthetic case is tagged with one, which is what
    makes per-type and held-out-type evaluation possible.

    TAMPERED_STATEMENT and MARKET_STRESS come from model outputs (the
    forgery CNN and market fusion). The synthetic generator deliberately
    never fabricates those scores -- the same reasoning the original
    README gives for never fabricating ELA scores -- so those two rules
    are covered by hand-built unit tests only, until real detectors exist.
    """
    NONE = "none"
    FIGURE_MISMATCH = "figure_mismatch"
    PERIOD_MISMATCH = "period_mismatch"
    INVALID_GSTIN = "invalid_gstin"
    GSTIN_STATE_MISMATCH = "gstin_state_mismatch"
    QUALIFIED_OPINION = "qualified_opinion"
    TAMPERED_STATEMENT = "tampered_statement"
    MARKET_STRESS = "market_stress"


# Relative difference between a submitted and an audited figure above
# which the two are treated as contradicting each other. Audited totals
# should match a company's own submission almost exactly; 2% leaves room
# for rounding and restated-vs-original presentation differences.
FIGURE_TOLERANCE = 0.02

# Model-output thresholds. Placeholders until the forgery CNN and market
# fusion model exist and these can be chosen by a measured sweep, the way
# FAST_TRACK_RISK_THRESHOLD was for claims.
TAMPER_PROBABILITY_THRESHOLD = 0.7
MARKET_STRESS_THRESHOLD = 0.8


@dataclass
class ReportedFigure:
    metric: FinancialMetric
    value_inr_crore: float
    fiscal_year: str  # "FY2025"
    source: str       # "submitted_statement:p2" / "annual_report:p47"


@dataclass
class UnderwritingEvidence:
    case_id: str
    company_name: str
    registered_state_code: str | None  # 2-digit GST state code of the registered office
    company_gstin: str | None

    submitted_fiscal_year: str
    submitted_figures: list[ReportedFigure] = field(default_factory=list)

    audited_fiscal_year: str | None = None
    audited_figures: list[ReportedFigure] = field(default_factory=list)

    auditor_opinion: AuditorOpinion | None = None
    going_concern_note: bool = False

    # Model outputs -- None means "detector not run", not "clean".
    tamper_probability: float | None = None
    market_stress_score: float | None = None

    # Ground truth, synthetic only.
    injected_inconsistency: UnderwritingInconsistencyType | None = None

    def audited(self, metric: FinancialMetric) -> ReportedFigure | None:
        return next((f for f in self.audited_figures if f.metric == metric), None)
