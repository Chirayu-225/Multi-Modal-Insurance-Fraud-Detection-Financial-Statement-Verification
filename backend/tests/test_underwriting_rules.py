"""Underwriting rule pack: each generated type is caught, clean cases stay
clean, and the two model-fed rules behave on hand-built inputs."""
import random

from app.ml.underwriting_domain import (
    AuditorOpinion,
    FinancialMetric,
    ReportedFigure,
    UnderwritingEvidence,
    UnderwritingInconsistencyType as UT,
)
from app.ml.underwriting_synthetic import GENERATED_TYPES, generate_case, generate_dataset
from app.services.rules.underwriting import evaluate_underwriting, is_uw_fast_track


def test_clean_cases_raise_no_flags_and_fast_track():
    cases = generate_dataset(n_clean=300, n_per_type=0, seed=11)
    results = [evaluate_underwriting(c) for c in cases]
    assert all(not r.flags for r in results)
    assert all(r.fast_track for r in results)


def test_each_generated_type_is_detected_without_cross_contamination():
    rng = random.Random(12)
    for kind in GENERATED_TYPES:
        cases = [generate_case(rng, kind) for _ in range(150)]
        triggered = [evaluate_underwriting(c).triggered_types for c in cases]
        assert all(kind in t for t in triggered), f"{kind} missed"
        others = set(GENERATED_TYPES) - {kind}
        assert not any(t & others for t in triggered), f"{kind} cases tripped other rules"


def test_generator_is_reproducible():
    a = [c.case_id for c in generate_dataset(20, 5, seed=99)]
    b = [c.case_id for c in generate_dataset(20, 5, seed=99)]
    assert a == b


def _base(**overrides) -> UnderwritingEvidence:
    fig = [ReportedFigure(FinancialMetric.REVENUE, 100.0, "FY2025", "x")]
    kwargs = dict(case_id="UW-test", company_name="T", registered_state_code=None, company_gstin=None,
                  submitted_fiscal_year="FY2025", submitted_figures=fig, audited_fiscal_year="FY2025",
                  audited_figures=list(fig), auditor_opinion=AuditorOpinion.UNQUALIFIED)
    kwargs.update(overrides)
    return UnderwritingEvidence(**kwargs)


def test_small_rounding_difference_is_within_tolerance():
    ev = _base(submitted_figures=[ReportedFigure(FinancialMetric.REVENUE, 101.0, "FY2025", "x")])
    assert not evaluate_underwriting(ev).flags


def test_tamper_rule_fires_only_above_threshold_and_is_silent_when_not_run():
    assert UT.TAMPERED_STATEMENT in evaluate_underwriting(_base(tamper_probability=0.9)).triggered_types
    assert not evaluate_underwriting(_base(tamper_probability=0.3)).flags
    assert not evaluate_underwriting(_base(tamper_probability=None)).flags


def test_market_stress_alone_never_blocks_fast_track():
    result = evaluate_underwriting(_base(market_stress_score=0.95))
    assert UT.MARKET_STRESS in result.triggered_types
    assert is_uw_fast_track(result.risk_score)
