"""
Append-only, hash-chained audit log for every scoring decision.

Why this exists: financial systems need to show exactly what decision
was made, on what evidence, by which model version, and when -- and be
able to prove that record hasn't been quietly edited afterward. IRDAI's
Insurance Fraud Monitoring Framework (see README) requires exactly this
kind of traceable reporting trail. A plain database row can be UPDATEd
by anyone with DB access with no trace; a hash chain makes that
detectable, the same principle a blockchain uses, applied here to a
single append-only table rather than anything distributed or consensus-
based -- no cryptocurrency, no separate infra, just SHA-256 over each
row's fields plus a pointer to the previous row's hash.

Two levels of tamper detection, deliberately kept separate:

  - verify_entry_self_consistency: recompute one entry's hash from its
    own stored fields and prev_hash, compare to its stored entry_hash.
    Catches a row whose payload/timestamp was edited after being
    written, independent of anything else in the table.
  - verify_full_chain: walk every entry in insertion order and confirm
    each one's prev_hash actually equals the previous entry's
    entry_hash (in addition to self-consistency). Catches a row being
    deleted or inserted out of band -- something self-consistency alone
    can't see, since a deleted row leaves no trace of itself for a
    per-row check to fail against.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import AuditLogEntry

GENESIS_HASH = "0" * 64


def _canonical_string(prev_hash: str, claim_id: str, event_type: str, payload_json: str, model_version: str, timestamp_iso: str) -> str:
    # Pipe-delimited with explicit field count -- simple and, for this
    # purpose, sufficient (none of these fields legitimately contain a
    # literal "|", and payload_json is itself already escaped JSON).
    return f"{prev_hash}|{claim_id}|{event_type}|{payload_json}|{model_version}|{timestamp_iso}"


def compute_entry_hash(prev_hash: str, claim_id: str, event_type: str, payload_json: str, model_version: str, timestamp_iso: str) -> str:
    canonical = _canonical_string(prev_hash, claim_id, event_type, payload_json, model_version, timestamp_iso)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_payload_json(payload: dict) -> str:
    """sort_keys=True so the same logical payload always serializes
    identically -- otherwise dict key order (which Python does not
    guarantee is stable across processes/versions) could make an
    honest, unmodified entry look tampered on re-verification."""
    return json.dumps(payload, sort_keys=True, default=str)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_entry_self_consistency(entry) -> bool:
    """`entry` needs .prev_hash, .claim_id, .event_type, .payload_json,
    .model_version, .timestamp_iso, .entry_hash attributes -- works for
    an AuditLogEntry ORM row or any duck-typed equivalent (e.g. in a
    unit test with a plain namespace/dict-like object)."""
    expected = compute_entry_hash(
        entry.prev_hash, entry.claim_id, entry.event_type, entry.payload_json, entry.model_version, entry.timestamp_iso
    )
    return expected == entry.entry_hash


def verify_full_chain(entries_in_order: list) -> tuple[bool, str]:
    """`entries_in_order` must be every entry in the table, in the exact
    order they were inserted (id ascending) -- a partial or reordered
    list will report false tampering, which is the correct behavior for
    what this function claims to check."""
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(entries_in_order):
        if entry.prev_hash != prev_hash:
            return False, f"entry {i} (claim {entry.claim_id}) does not chain from the previous entry -- a row may be missing, reordered, or inserted out of band"
        if not verify_entry_self_consistency(entry):
            return False, f"entry {i} (claim {entry.claim_id}) hash does not match its own stored fields -- this row was edited after being written"
        prev_hash = entry.entry_hash
    return True, f"chain verified, {len(entries_in_order)} entries, no tampering detected"


async def append_entry(session: AsyncSession, claim_id: str, event_type: str, payload: dict, model_version: str) -> AuditLogEntry:
    """Appends one entry to the global chain. Reads the current last row
    to get its hash as this entry's prev_hash -- callers don't (and
    shouldn't) supply prev_hash themselves, since only this function,
    reading the table at write time, can know what the true last entry
    is."""
    last = (await session.exec(select(AuditLogEntry).order_by(AuditLogEntry.id.desc()).limit(1))).first()
    prev_hash = last.entry_hash if last else GENESIS_HASH

    payload_json = make_payload_json(payload)
    timestamp_iso = now_iso()
    entry_hash = compute_entry_hash(prev_hash, claim_id, event_type, payload_json, model_version, timestamp_iso)

    entry = AuditLogEntry(
        claim_id=claim_id,
        event_type=event_type,
        payload_json=payload_json,
        model_version=model_version,
        timestamp_iso=timestamp_iso,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def fetch_claim_entries(session: AsyncSession, claim_id: str) -> list[AuditLogEntry]:
    return list((await session.exec(select(AuditLogEntry).where(AuditLogEntry.claim_id == claim_id).order_by(AuditLogEntry.id))).all())


async def fetch_all_entries_in_order(session: AsyncSession) -> list[AuditLogEntry]:
    return list((await session.exec(select(AuditLogEntry).order_by(AuditLogEntry.id))).all())
