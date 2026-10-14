"""
Orchestrates one underwriting case: build UnderwritingEvidence, run the
underwriting rule pack, persist the result, append to the same
hash-chained audit log that motor claims use.

Phase 1 takes structured figures from the API. Phase 3 adds extraction:
LayoutLMv3 on the uploaded statement (submitted figures, restricted, local
only) and annual-report RAG through the LLM gateway (audited figures,
public), both producing the same UnderwritingEvidence shape, so nothing
below this module changes.
"""
from __future__ import annotations

import json
import uuid

from sqlmodel.ext.asyncio.session import AsyncSession

from app.ml.underwriting_domain import AuditorOpinion, FinancialMetric, ReportedFigure, UnderwritingEvidence
from app.models import UnderwritingCase, UnderwritingResultRecord
from app.schemas import UnderwritingCaseIn
from app.services import audit_log
from app.services.rules.underwriting import evaluate_underwriting


def _figures(items) -> list[ReportedFigure]:
    return [ReportedFigure(FinancialMetric(i.metric), i.value_inr_crore, i.fiscal_year, i.source) for i in items]


def evidence_from_input(case_id: str, data: UnderwritingCaseIn) -> UnderwritingEvidence:
    return UnderwritingEvidence(
        case_id=case_id,
        company_name=data.company_name,
        registered_state_code=data.registered_state_code,
        company_gstin=data.company_gstin,
        submitted_fiscal_year=data.submitted_fiscal_year,
        submitted_figures=_figures(data.submitted_figures),
        audited_fiscal_year=data.audited_fiscal_year,
        audited_figures=_figures(data.audited_figures),
        auditor_opinion=AuditorOpinion(data.auditor_opinion) if data.auditor_opinion else None,
        going_concern_note=data.going_concern_note,
        tamper_probability=data.tamper_probability,
        market_stress_score=data.market_stress_score,
    )


class UnderwritingPipeline:
    async def process(self, data: UnderwritingCaseIn, session: AsyncSession) -> tuple[UnderwritingCase, UnderwritingResultRecord]:
        case_id = f"UW-{uuid.uuid4().hex[:10]}"
        evidence = evidence_from_input(case_id, data)  # validates enums before anything is written
        result = evaluate_underwriting(evidence)

        case = UnderwritingCase(
            id=case_id,
            company_name=data.company_name,
            company_gstin=data.company_gstin,
            registered_state_code=data.registered_state_code,
            submitted_fiscal_year=data.submitted_fiscal_year,
            submitted_figures_json=json.dumps([f.model_dump() for f in data.submitted_figures]),
            audited_fiscal_year=data.audited_fiscal_year,
            audited_figures_json=json.dumps([f.model_dump() for f in data.audited_figures]),
            auditor_opinion=data.auditor_opinion,
            going_concern_note=data.going_concern_note,
            tamper_probability=data.tamper_probability,
            market_stress_score=data.market_stress_score,
        )
        session.add(case)
        await session.commit()

        record = UnderwritingResultRecord(
            case_id=case_id,
            risk_score=result.risk_score,
            flags_json=json.dumps([{"type": f.type.value, "severity": f.severity, "message": f.message} for f in result.flags]),
            model_version=result.model_version,
        )
        session.add(record)
        await session.commit()
        await session.refresh(case)
        await session.refresh(record)

        await audit_log.append_entry(
            session,
            claim_id=case_id,
            event_type="underwriting_evaluated",
            payload={
                "risk_score": result.risk_score,
                "model_version": result.model_version,
                "flags": [{"type": f.type.value, "severity": f.severity} for f in result.flags],
            },
            model_version=result.model_version,
        )
        return case, record
