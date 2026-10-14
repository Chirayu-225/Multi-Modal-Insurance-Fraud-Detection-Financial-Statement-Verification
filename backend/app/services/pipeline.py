"""
Orchestrates one claim through the full pipeline: vision extraction,
OCR, the policy stub, the consistency engine, and persistence. This is
the module the API layer calls; it's also the one place that assembles
a ClaimEvidence from real uploaded files (as opposed to
ml/synthetic_data.py assembling one synthetically).
"""
from __future__ import annotations

import json

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.ml.domain import ClaimEvidence, IncidentType, InconsistencyType, VehicleSegment
from app.models import Claim, ClaimEvidenceRecord, ConsistencyResultRecord
from app.services import audit_log
from app.services.consistency import Flag, compare_bill_extractions, evaluate
from app.services.image_utils import compute_ela_score, compute_phash
from app.services.ocr import OCRClient, TesseractOCR, get_ocr_client
from app.services.policy import PolicyService
from app.services.vision import MockVisionClient, SchemaViolationError, VisionClient, get_vision_client


class ClaimPipeline:
    def __init__(
        self,
        vision_client: VisionClient | None = None,
        ocr_client: OCRClient | None = None,
        policy_service: PolicyService | None = None,
    ):
        self.vision_client = vision_client or get_vision_client(
            settings.vision_backend, settings.gemini_api_key, settings.gemini_model
        )
        self.ocr_client = ocr_client or get_ocr_client(settings.vision_backend)
        self.policy_service = policy_service or PolicyService()

    async def _other_photo_hashes(self, session: AsyncSession, exclude_claim_id: str) -> dict[str, str]:
        rows = (
            await session.exec(
                select(ClaimEvidenceRecord.claim_id, ClaimEvidenceRecord.photo_hash).where(
                    ClaimEvidenceRecord.claim_id != exclude_claim_id
                )
            )
        ).all()
        return {claim_id: photo_hash for claim_id, photo_hash in rows}

    async def process(self, claim: Claim, session: AsyncSession) -> tuple[ClaimEvidenceRecord, ConsistencyResultRecord]:
        extra_flags: list[Flag] = []

        try:
            damage = self.vision_client.extract_damage(claim.photo_path)
        except SchemaViolationError as exc:
            # Don't crash the request and don't silently trust a
            # malformed response either -- fall back to a safe mock
            # extraction so the rest of the pipeline still runs, and
            # attach a maximum-severity flag so this claim gets a human
            # look rather than quietly passing through. See
            # SchemaViolationError's docstring in vision.py for why this
            # is treated as a security event, not an ordinary bug.
            damage = MockVisionClient().extract_damage(claim.photo_path)
            extra_flags.append(
                Flag(
                    type=InconsistencyType.PROMPT_INJECTION,
                    severity=0.95,
                    message=(
                        f"The vision model's response for this claim's photo did not match the required "
                        f"schema -- treated as a suspected prompt-injection attempt, not an ordinary parsing "
                        f"bug. Falling back to a conservative mock extraction; this claim needs manual review. "
                        f"({exc})"
                    ),
                )
            )

        bill_items = self.ocr_client.extract_bill_items(claim.bill_path) if claim.bill_path else []
        raw_bill_text = self.ocr_client.extract_raw_text(claim.bill_path) if claim.bill_path else ""
        excluded_categories = self.policy_service.get_excluded_categories(getattr(claim, "policy_path", None))
        photo_hash = compute_phash(claim.photo_path)
        ela_score = compute_ela_score(claim.photo_path)

        if settings.enable_cross_check and claim.bill_path and not isinstance(self.ocr_client, TesseractOCR):
            secondary_items = TesseractOCR().extract_bill_items(claim.bill_path)
            divergence_flag = compare_bill_extractions(bill_items, secondary_items)
            if divergence_flag:
                extra_flags.append(divergence_flag)

        evidence = ClaimEvidence(
            claim_id=claim.id,
            vehicle_segment=VehicleSegment(claim.vehicle_segment),
            incident_date=claim.incident_date,
            described_incident_type=IncidentType(claim.described_incident_type),
            photo_damage_location=damage.location,
            photo_damage_severity=damage.severity,
            photo_hash=photo_hash,
            photo_capture_date=damage.capture_date,
            bill_line_items=bill_items,
            policy_excluded_categories=excluded_categories,
            incident_description_text=claim.incident_description,
            raw_bill_ocr_text=raw_bill_text,
            ela_score=ela_score,
        )

        other_hashes = await self._other_photo_hashes(session, exclude_claim_id=claim.id)
        result = evaluate(evidence, other_photo_hashes=other_hashes, extra_flags=extra_flags)

        evidence_record = ClaimEvidenceRecord(
            claim_id=claim.id,
            photo_damage_location=evidence.photo_damage_location.value,
            photo_damage_severity=evidence.photo_damage_severity.value,
            photo_hash=evidence.photo_hash,
            photo_capture_date=evidence.photo_capture_date,
            bill_line_items_json=json.dumps([{"category": i.category, "cost_inr": i.cost_inr} for i in evidence.bill_line_items]),
            policy_excluded_categories_json=json.dumps(sorted(evidence.policy_excluded_categories)),
        )
        result_record = ConsistencyResultRecord(
            claim_id=claim.id,
            risk_score=result.risk_score,
            flags_json=json.dumps([{"type": f.type.value, "severity": f.severity, "message": f.message} for f in result.flags]),
            model_version=result.model_version,
        )

        session.add(evidence_record)
        session.add(result_record)
        await session.commit()
        await session.refresh(evidence_record)
        await session.refresh(result_record)

        # Append-only, hash-chained record of exactly what was decided
        # and on what evidence -- see services/audit_log.py. This is a
        # separate write from the two above: it happens after they
        # succeed, and its own failure (e.g. a hash mismatch bug) should
        # never be allowed to look like the claim itself failed to
        # process, so a real deployment would want this in its own
        # try/except with alerting, not silently swallowed either.
        await audit_log.append_entry(
            session,
            claim_id=claim.id,
            event_type="consistency_evaluated",
            payload={
                "risk_score": result.risk_score,
                "model_version": result.model_version,
                "flags": [{"type": f.type.value, "severity": f.severity} for f in result.flags],
                "photo_hash": evidence.photo_hash,
            },
            model_version=result.model_version,
        )

        return evidence_record, result_record
