from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class FlagOut(BaseModel):
    type: str
    severity: float
    message: str


class ConsistencyResultOut(BaseModel):
    risk_score: float
    model_version: str
    flags: list[FlagOut]
    fast_track: bool


class ClaimSummaryOut(BaseModel):
    id: str
    claimant_name: str
    incident_date: date
    described_incident_type: str
    created_at: datetime
    risk_score: float | None = None
    fast_track: bool | None = None


class ClaimDetailOut(BaseModel):
    id: str
    claimant_name: str
    vehicle_segment: str
    incident_date: date
    described_incident_type: str
    incident_description: str
    photo_damage_location: str
    photo_damage_severity: str
    photo_capture_date: date | None
    bill_line_items: list[dict]
    result: ConsistencyResultOut | None


# ------------------------------------------------------------ underwriting

class ReportedFigureIn(BaseModel):
    metric: str  # one of app.ml.underwriting_domain.FinancialMetric values
    value_inr_crore: float
    fiscal_year: str
    source: str = "manual_entry"


class UnderwritingCaseIn(BaseModel):
    """Phase 1 input: figures entered as structured data. In Phase 3 the
    submitted figures come from LayoutLMv3 on the uploaded statement and
    the audited ones from annual-report RAG, through the same shape."""
    company_name: str
    company_gstin: str | None = None
    registered_state_code: str | None = None
    submitted_fiscal_year: str
    submitted_figures: list[ReportedFigureIn]
    audited_fiscal_year: str | None = None
    audited_figures: list[ReportedFigureIn] = []
    auditor_opinion: str | None = None  # unqualified / qualified / adverse / disclaimer
    going_concern_note: bool = False
    tamper_probability: float | None = None
    market_stress_score: float | None = None


class UnderwritingSummaryOut(BaseModel):
    id: str
    company_name: str
    submitted_fiscal_year: str
    created_at: datetime
    risk_score: float | None = None
    fast_track: bool | None = None


class UnderwritingDetailOut(BaseModel):
    id: str
    company_name: str
    company_gstin: str | None
    registered_state_code: str | None
    submitted_fiscal_year: str
    submitted_figures: list[dict]
    audited_fiscal_year: str | None
    audited_figures: list[dict]
    auditor_opinion: str | None
    going_concern_note: bool
    result: ConsistencyResultOut | None
