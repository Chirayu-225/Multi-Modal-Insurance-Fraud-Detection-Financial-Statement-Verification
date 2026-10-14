from __future__ import annotations

import os
import uuid
from datetime import date

import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.db import get_session
from app.models import Claim, ClaimEvidenceRecord, ConsistencyResultRecord
from app.schemas import ClaimDetailOut, ClaimSummaryOut, ConsistencyResultOut, FlagOut
from app.services.consistency import is_fast_track
from app.services.pipeline import ClaimPipeline

router = APIRouter(prefix="/claims", tags=["claims"])
pipeline = ClaimPipeline()

# Minimal in-memory rate limit, keyed by client IP: at most
# RATE_LIMIT_MAX submissions per RATE_LIMIT_WINDOW_SECONDS. Addresses
# the "score probing" concern -- without this, someone can submit many
# small variations of a claim to map out exactly where the risk-score
# threshold sits. This doesn't replace logging repeated near-duplicate
# submissions (the reused-photo pHash index in consistency.py already
# gives you that for free -- a resubmission with the same or a
# near-identical photo shows up as a REUSED_PHOTO flag against itself),
# it just slows down brute-force probing. In-memory means it resets on
# restart and doesn't share state across multiple backend replicas --
# fine for a solo demo, swap for a Redis-backed limiter (e.g. slowapi
# with a redis storage backend) before this ever runs multi-instance.
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW_SECONDS = 60
_submission_log: dict[str, deque] = defaultdict(deque)


def rate_limit_submissions(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    log = _submission_log[client_ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many claim submissions -- please slow down and try again shortly.")
    log.append(now)


def _save_upload(claim_id: str, upload: UploadFile) -> str:
    claim_dir = os.path.join(settings.upload_dir, claim_id)
    os.makedirs(claim_dir, exist_ok=True)
    dest = os.path.join(claim_dir, upload.filename)
    with open(dest, "wb") as f:
        f.write(upload.file.read())
    return dest


@router.post("", response_model=ClaimDetailOut, dependencies=[Depends(rate_limit_submissions)])
async def submit_claim(
    claimant_name: str = Form(...),
    vehicle_segment: str = Form(...),
    incident_date: date = Form(...),
    described_incident_type: str = Form(...),
    incident_description: str = Form(...),
    photo: UploadFile = File(...),
    bill: UploadFile | None = File(None),
    session: AsyncSession = Depends(get_session),
):
    claim_id = str(uuid.uuid4())
    photo_path = _save_upload(claim_id, photo)
    bill_path = _save_upload(claim_id, bill) if bill else None

    claim = Claim(
        id=claim_id,
        claimant_name=claimant_name,
        vehicle_segment=vehicle_segment,
        incident_date=incident_date,
        described_incident_type=described_incident_type,
        incident_description=incident_description,
        photo_path=photo_path,
        bill_path=bill_path,
    )
    session.add(claim)
    await session.commit()
    await session.refresh(claim)

    try:
        evidence, result = await pipeline.process(claim, session)
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors to the caller during development
        raise HTTPException(status_code=422, detail=f"Claim processing failed: {exc}") from exc

    return _to_detail(claim, evidence, result)


@router.get("", response_model=list[ClaimSummaryOut])
async def list_claims(session: AsyncSession = Depends(get_session)):
    claims = (await session.exec(select(Claim).order_by(Claim.created_at.desc()))).all()
    out = []
    for claim in claims:
        result = (
            await session.exec(
                select(ConsistencyResultRecord)
                .where(ConsistencyResultRecord.claim_id == claim.id)
                .order_by(ConsistencyResultRecord.created_at.desc())
            )
        ).first()
        out.append(
            ClaimSummaryOut(
                id=claim.id,
                claimant_name=claim.claimant_name,
                incident_date=claim.incident_date,
                described_incident_type=claim.described_incident_type,
                created_at=claim.created_at,
                risk_score=result.risk_score if result else None,
                fast_track=is_fast_track(result.risk_score) if result else None,
            )
        )
    return out


@router.get("/{claim_id}", response_model=ClaimDetailOut)
async def get_claim(claim_id: str, audience: str = "adjuster", session: AsyncSession = Depends(get_session)):
    claim = await session.get(Claim, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    evidence = (
        await session.exec(select(ClaimEvidenceRecord).where(ClaimEvidenceRecord.claim_id == claim_id))
    ).first()
    result = (
        await session.exec(select(ConsistencyResultRecord).where(ConsistencyResultRecord.claim_id == claim_id))
    ).first()
    return _to_detail(claim, evidence, result, audience=audience)


def _to_detail(
    claim: Claim,
    evidence: ClaimEvidenceRecord | None,
    result: ConsistencyResultRecord | None,
    audience: str = "adjuster",
) -> ClaimDetailOut:
    # A detailed explanation ("bill exceeds reference cost by 40%") is
    # exactly the cheat sheet a claimant needs to fix a resubmission so
    # it clears next time. Adjusters get the full message; anyone else
    # gets the category and severity only, never the reasoning. See
    # settings.expose_explanations_to_claimant.
    show_full_explanations = audience == "adjuster" or settings.expose_explanations_to_claimant

    def _flag_out(f: dict) -> FlagOut:
        if show_full_explanations:
            return FlagOut(**f)
        return FlagOut(type=f["type"], severity=f["severity"], message="An inconsistency was detected in this category.")

    return ClaimDetailOut(
        id=claim.id,
        claimant_name=claim.claimant_name,
        vehicle_segment=claim.vehicle_segment,
        incident_date=claim.incident_date,
        described_incident_type=claim.described_incident_type,
        incident_description=claim.incident_description,
        photo_damage_location=evidence.photo_damage_location if evidence else "",
        photo_damage_severity=evidence.photo_damage_severity if evidence else "",
        photo_capture_date=evidence.photo_capture_date if evidence else None,
        bill_line_items=evidence.bill_line_items() if evidence else [],
        result=(
            ConsistencyResultOut(
                risk_score=result.risk_score,
                model_version=result.model_version,
                flags=[_flag_out(f) for f in result.flags()],
                fast_track=is_fast_track(result.risk_score),
            )
            if result
            else None
        ),
    )
