from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import get_session
from app.services import audit_log

router = APIRouter(prefix="/audit-log", tags=["audit-log"])


@router.get("/claims/{claim_id}")
async def get_claim_audit_log(claim_id: str, session: AsyncSession = Depends(get_session)):
    entries = await audit_log.fetch_claim_entries(session, claim_id)
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "timestamp_iso": e.timestamp_iso,
            "model_version": e.model_version,
            "payload": e.payload_json,
            "prev_hash": e.prev_hash,
            "entry_hash": e.entry_hash,
            "self_consistent": audit_log.verify_entry_self_consistency(e),
        }
        for e in entries
    ]


@router.get("/verify")
async def verify_full_chain(session: AsyncSession = Depends(get_session)):
    """Walks the ENTIRE ledger (every claim, insertion order) and
    confirms no entry has been edited, reordered, or removed. This is
    the check that matters for an audit -- the per-claim endpoint above
    only tells you a given row is self-consistent, not that nothing was
    deleted around it."""
    entries = await audit_log.fetch_all_entries_in_order(session)
    is_valid, message = audit_log.verify_full_chain(entries)
    return {"valid": is_valid, "message": message, "entries_checked": len(entries)}
