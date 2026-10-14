"""
Synthetic underwriting cases with known injected inconsistencies -- the
underwriting counterpart of synthetic_data.py.

What it generates: a company with plausible audited figures, a submitted
statement that either matches them (clean) or carries exactly one
injected inconsistency type, plus GSTIN, registered state and auditor
opinion.

What it deliberately does NOT generate: tamper probabilities or market
stress scores. Those are model outputs; fabricating them would mean
inventing the very evidence those models exist to produce independently
(the same reasoning the original README gives for ELA). Those two rules
are covered by hand-built unit tests until real detectors exist.

Reproducibility: everything is driven by one random.Random(seed), and
every choice is made from a sorted sequence, never from a set -- see
the original README's PYTHONHASHSEED bug for why that matters.

Next step (Phase 3): replace the random audited figures with real ones
taken from public annual reports, and render submitted statements as
PDFs/images (including clean re-rendered forgeries) so the forensics
model has something real to look at.
"""
from __future__ import annotations

import random
import uuid

from app.ml.underwriting_domain import (
    AuditorOpinion,
    FinancialMetric,
    ReportedFigure,
    UnderwritingEvidence,
    UnderwritingInconsistencyType as UT,
)
from app.services.gstin import compute_checksum_char, generate_plausible_gstin

GENERATED_TYPES: list[UT] = [
    UT.FIGURE_MISMATCH,
    UT.PERIOD_MISMATCH,
    UT.INVALID_GSTIN,
    UT.GSTIN_STATE_MISMATCH,
    UT.QUALIFIED_OPINION,
]

_FISCAL_YEARS = ["FY2023", "FY2024", "FY2025", "FY2026"]
_METRICS = sorted(FinancialMetric, key=lambda m: m.value)


def _company_name(rng: random.Random) -> str:
    first = ["Arka", "Deccan", "Kaveri", "Nilgiri", "Sahyadri", "Tungabhadra", "Vindhya", "Malnad"]
    second = ["Logistics", "Textiles", "Agro Foods", "Engineering", "Pharma", "Infra Projects", "Polymers"]
    return f"{rng.choice(first)} {rng.choice(second)} Pvt Ltd"


def _base_figures(rng: random.Random, fy: str, source: str) -> list[ReportedFigure]:
    revenue = round(rng.uniform(20, 2000), 2)  # INR crore
    values = {
        FinancialMetric.REVENUE: revenue,
        FinancialMetric.NET_PROFIT: round(revenue * rng.uniform(0.02, 0.15), 2),
        FinancialMetric.TOTAL_ASSETS: round(revenue * rng.uniform(0.6, 1.8), 2),
        FinancialMetric.TOTAL_DEBT: round(revenue * rng.uniform(0.1, 0.9), 2),
    }
    return [ReportedFigure(m, values[m], fy, source) for m in _METRICS]


def _break_checksum(gstin: str) -> str:
    correct = gstin[14]
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    wrong = alphabet[(alphabet.index(correct) + 7) % 36]
    return gstin[:14] + wrong


def generate_case(rng: random.Random, injected: UT = UT.NONE) -> UnderwritingEvidence:
    fy = rng.choice(_FISCAL_YEARS[1:])
    audited = _base_figures(rng, fy, "annual_report:p47")
    submitted = [ReportedFigure(f.metric, f.value_inr_crore, fy, "submitted_statement:p2") for f in audited]

    gstin = generate_plausible_gstin(rng)
    state_code = gstin[:2]
    submitted_fy, audited_fy = fy, fy
    opinion, going_concern = AuditorOpinion.UNQUALIFIED, False

    if injected == UT.FIGURE_MISMATCH:
        target = rng.choice([FinancialMetric.REVENUE, FinancialMetric.NET_PROFIT, FinancialMetric.TOTAL_ASSETS])
        factor = rng.uniform(1.05, 1.40)  # inflated: the direction that helps an applicant
        submitted = [
            ReportedFigure(f.metric, round(f.value_inr_crore * factor, 2), f.fiscal_year, f.source) if f.metric == target else f
            for f in submitted
        ]
    elif injected == UT.PERIOD_MISMATCH:
        newer = _FISCAL_YEARS[_FISCAL_YEARS.index(fy) - 1]  # audited report on file is a year older
        audited_fy = newer
        audited = [ReportedFigure(f.metric, f.value_inr_crore, newer, f.source) for f in audited]
    elif injected == UT.INVALID_GSTIN:
        gstin = _break_checksum(gstin)
    elif injected == UT.GSTIN_STATE_MISMATCH:
        other = (int(state_code) % 37) + 1
        state_code = f"{other:02d}"
    elif injected == UT.QUALIFIED_OPINION:
        if rng.random() < 0.5:
            opinion = rng.choice([AuditorOpinion.QUALIFIED, AuditorOpinion.ADVERSE, AuditorOpinion.DISCLAIMER])
        else:
            going_concern = True
    elif injected != UT.NONE:
        raise ValueError(f"{injected} is not generated synthetically -- see module docstring")

    return UnderwritingEvidence(
        case_id=f"UW-{uuid.UUID(int=rng.getrandbits(128)).hex[:10]}",
        company_name=_company_name(rng),
        registered_state_code=state_code,
        company_gstin=gstin,
        submitted_fiscal_year=submitted_fy,
        submitted_figures=submitted,
        audited_fiscal_year=audited_fy,
        audited_figures=audited,
        auditor_opinion=opinion,
        going_concern_note=going_concern,
        injected_inconsistency=injected,
    )


def generate_dataset(n_clean: int, n_per_type: int, seed: int = 7) -> list[UnderwritingEvidence]:
    rng = random.Random(seed)
    cases = [generate_case(rng, UT.NONE) for _ in range(n_clean)]
    for t in GENERATED_TYPES:
        cases.extend(generate_case(rng, t) for _ in range(n_per_type))
    return cases


# Keep compute_checksum_char importable from here for tests that build
# GSTINs by hand.
__all__ = ["GENERATED_TYPES", "generate_case", "generate_dataset", "compute_checksum_char"]
