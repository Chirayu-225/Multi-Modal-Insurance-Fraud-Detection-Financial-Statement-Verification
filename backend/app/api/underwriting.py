from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.claims import rate_limit_submissions
from app.config import settings
from app.db import get_session
from app.models import UnderwritingCase, UnderwritingResultRecord
from app.schemas import (
    ConsistencyResultOut,
    FlagOut,
    UnderwritingCaseIn,
    UnderwritingDetailOut,
    UnderwritingSummaryOut,
)
from app.services.rules.underwriting import is_uw_fast_track
from app.services.underwriting_pipeline import UnderwritingPipeline

router = APIRouter(prefix="/underwriting", tags=["underwriting"])
pipeline = UnderwritingPipeline()


@router.post("/cases", response_model=UnderwritingDetailOut, dependencies=[Depends(rate_limit_submissions)])
async def submit_case(data: UnderwritingCaseIn, session: AsyncSession = Depends(get_session)):
    try:
        case, record = await pipeline.process(data, session)
    except ValueError as exc:  # bad enum value (metric / auditor opinion)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_detail(case, record, audience="underwriter")


@router.get("/cases", response_model=list[UnderwritingSummaryOut])
async def list_cases(session: AsyncSession = Depends(get_session)):
    cases = (await session.exec(select(UnderwritingCase).order_by(UnderwritingCase.created_at.desc()))).all()
    results = {r.case_id: r for r in (await session.exec(select(UnderwritingResultRecord))).all()}
    out = []
    for c in cases:
        r = results.get(c.id)
        out.append(
            UnderwritingSummaryOut(
                id=c.id,
                company_name=c.company_name,
                submitted_fiscal_year=c.submitted_fiscal_year,
                created_at=c.created_at,
                risk_score=r.risk_score if r else None,
                fast_track=is_uw_fast_track(r.risk_score) if r else None,
            )
        )
    return out


@router.get("/cases/{case_id}", response_model=UnderwritingDetailOut)
async def get_case(case_id: str, audience: str = "underwriter", session: AsyncSession = Depends(get_session)):
    case = await session.get(UnderwritingCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Underwriting case not found")
    record = (await session.exec(select(UnderwritingResultRecord).where(UnderwritingResultRecord.case_id == case_id))).first()
    return _to_detail(case, record, audience=audience)


def _to_detail(case: UnderwritingCase, record: UnderwritingResultRecord | None, audience: str) -> UnderwritingDetailOut:
    # Same reasoning as claims: full explanations for the underwriter,
    # category and severity only for the applicant.
    full = audience == "underwriter" or settings.expose_explanations_to_claimant

    def flag_out(f: dict) -> FlagOut:
        if full:
            return FlagOut(**f)
        return FlagOut(type=f["type"], severity=f["severity"], message="An inconsistency was detected in this category.")

    result = None
    if record:
        result = ConsistencyResultOut(
            risk_score=record.risk_score,
            model_version=record.model_version,
            flags=[flag_out(f) for f in record.flags()],
            fast_track=is_uw_fast_track(record.risk_score),
        )
    return UnderwritingDetailOut(
        id=case.id,
        company_name=case.company_name,
        company_gstin=case.company_gstin,
        registered_state_code=case.registered_state_code,
        submitted_fiscal_year=case.submitted_fiscal_year,
        submitted_figures=json.loads(case.submitted_figures_json),
        audited_fiscal_year=case.audited_fiscal_year,
        audited_figures=json.loads(case.audited_figures_json),
        auditor_opinion=case.auditor_opinion,
        going_concern_note=case.going_concern_note,
        result=result,
    )
