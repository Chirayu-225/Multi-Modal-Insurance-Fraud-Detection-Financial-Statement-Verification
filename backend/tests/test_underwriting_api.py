"""Underwriting endpoints through the real app and a real (in-memory)
database, mirroring test_api_integration.py -- and checking that claims
and underwriting share one verifiable audit chain."""
from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import claims as claims_api
from app.db import get_session
from app.main import app


@pytest.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False},
                                 poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override():
        async with maker() as s:
            yield s

    claims_api._submission_log.clear()  # rate limiter is shared and in-memory
    app.dependency_overrides[get_session] = _override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    await engine.dispose()


def _payload(submitted_revenue=120.0):
    return {
        "company_name": "Kaveri Textiles Pvt Ltd",
        "company_gstin": "27AAPFU0939F1ZV",
        "registered_state_code": "27",
        "submitted_fiscal_year": "FY2025",
        "submitted_figures": [{"metric": "revenue", "value_inr_crore": submitted_revenue, "fiscal_year": "FY2025",
                               "source": "submitted_statement:p2"}],
        "audited_fiscal_year": "FY2025",
        "audited_figures": [{"metric": "revenue", "value_inr_crore": 96.0, "fiscal_year": "FY2025",
                             "source": "annual_report:p47"}],
        "auditor_opinion": "unqualified",
    }


async def test_mismatch_is_flagged_and_detail_round_trips(client):
    r = await client.post("/underwriting/cases", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "figure_mismatch" in {f["type"] for f in body["result"]["flags"]}
    assert body["result"]["fast_track"] is False

    listed = (await client.get("/underwriting/cases")).json()
    assert body["id"] in [c["id"] for c in listed]
    assert (await client.get(f"/underwriting/cases/{body['id']}")).json()["id"] == body["id"]


async def test_consistent_case_is_fast_tracked(client):
    body = (await client.post("/underwriting/cases", json=_payload(submitted_revenue=96.0))).json()
    assert body["result"]["flags"] == [] and body["result"]["fast_track"] is True


async def test_applicant_view_is_redacted(client):
    case_id = (await client.post("/underwriting/cases", json=_payload())).json()["id"]
    flags = (await client.get(f"/underwriting/cases/{case_id}?audience=applicant")).json()["result"]["flags"]
    assert flags and all("submitted" not in f["message"].lower() for f in flags)


async def test_bad_metric_is_rejected(client):
    p = _payload()
    p["submitted_figures"][0]["metric"] = "vibes"
    assert (await client.post("/underwriting/cases", json=p)).status_code == 422


async def test_claims_and_underwriting_share_one_verifiable_chain(client):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color="blue").save(buf, format="JPEG")
    r = await client.post("/claims", data={
        "claimant_name": "A", "vehicle_segment": "sedan", "incident_date": "2026-02-01",
        "described_incident_type": "rear_collision", "incident_description": "Rear-ended at a signal.",
    }, files={"photo": ("p.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200, r.text
    uw_id = (await client.post("/underwriting/cases", json=_payload())).json()["id"]

    entries = (await client.get(f"/audit-log/claims/{uw_id}")).json()
    assert entries and entries[0]["event_type"] == "underwriting_evaluated" and entries[0]["self_consistent"]
    verify = (await client.get("/audit-log/verify")).json()
    assert verify["valid"] is True and verify["entries_checked"] >= 2
