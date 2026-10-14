"""
Persistence models (SQLModel = Pydantic + SQLAlchemy in one class, less
boilerplate than defining both separately). Structured data the app
needs to query lives as real columns; blobs that are only ever read as
a whole (line items, flags) are stored as JSON text columns rather than
extra tables, since a solo project doesn't need to query "all claims
with a specific line item" -- if that changes, split them out later.
"""
from __future__ import annotations

import json
from datetime import date, datetime

from sqlmodel import Field, SQLModel


class Claim(SQLModel, table=True):
    id: str = Field(primary_key=True)
    claimant_name: str
    vehicle_segment: str
    incident_date: date
    described_incident_type: str
    incident_description: str
    photo_path: str
    bill_path: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ClaimEvidenceRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    claim_id: str = Field(foreign_key="claim.id", index=True)
    photo_damage_location: str
    photo_damage_severity: str
    photo_hash: str = Field(index=True)
    photo_capture_date: date | None = None
    bill_line_items_json: str  # JSON list of {"category": str, "cost_inr": int}
    policy_excluded_categories_json: str  # JSON list of str

    def bill_line_items(self) -> list[dict]:
        return json.loads(self.bill_line_items_json)


class ConsistencyResultRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    claim_id: str = Field(foreign_key="claim.id", index=True)
    risk_score: float
    flags_json: str  # JSON list of {"type": str, "severity": float, "message": str}
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def flags(self) -> list[dict]:
        return json.loads(self.flags_json)


class AuditLogEntry(SQLModel, table=True):
    """One global, append-only, hash-chained ledger -- not one chain per
    claim. `id` ascending is insertion order, which is what makes
    services/audit_log.py::verify_full_chain meaningful: prev_hash on
    row N must equal entry_hash on row N-1, whichever claim that row
    belongs to. Never update or delete rows in this table from
    application code -- there is deliberately no update/delete endpoint
    for it.

    Merge note: `claim_id` now holds a *case* ID -- a motor claim ID or an
    underwriting case ID ("UW-..."), distinguished by event_type
    ("consistency_evaluated" vs "underwriting_evaluated"). The column keeps
    its name because it is part of the hashed canonical string, so
    renaming it would change nothing about security but would break
    verification of every existing entry. Its foreign key to claim.id was
    dropped for the same reason underwriting cases needed to be logged at
    all. Existing Postgres volumes still carry the old constraint: run
    `docker compose down -v` (dev data only) or
    `ALTER TABLE auditlogentry DROP CONSTRAINT auditlogentry_claim_id_fkey;`."""
    id: int | None = Field(default=None, primary_key=True)
    claim_id: str = Field(index=True)
    event_type: str
    payload_json: str
    model_version: str
    timestamp_iso: str  # canonical string used in the hash -- see audit_log.py
    prev_hash: str
    entry_hash: str = Field(index=True)


class UnderwritingCase(SQLModel, table=True):
    """A commercial-underwriting case. Figures are stored as JSON text,
    same convention as the claim tables: read as a whole, never queried
    by individual figure."""
    id: str = Field(primary_key=True)
    company_name: str
    company_gstin: str | None = None
    registered_state_code: str | None = None
    submitted_fiscal_year: str
    submitted_figures_json: str
    audited_fiscal_year: str | None = None
    audited_figures_json: str = "[]"
    auditor_opinion: str | None = None
    going_concern_note: bool = False
    tamper_probability: float | None = None
    market_stress_score: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UnderwritingResultRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    case_id: str = Field(foreign_key="underwritingcase.id", index=True)
    risk_score: float
    flags_json: str
    model_version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def flags(self) -> list[dict]:
        return json.loads(self.flags_json)
