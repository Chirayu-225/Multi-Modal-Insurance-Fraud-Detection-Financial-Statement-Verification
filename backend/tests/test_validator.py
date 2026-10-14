import json

import pytest

from app.services.validator import numbers_in, validate_figure_extraction
from app.services.vision import SchemaViolationError

CHUNKS = {
    "annual_reports:abc": "Revenue from operations for FY2025 was 1,234.56 crore (FY2024: 1,100.00 crore).",
    "annual_reports:def": "Net profit after tax stood at 98.40 crore.",
}


def _fig(value, chunks, metric="revenue"):
    return {"metric": metric, "value": value, "fiscal_year": "FY2025", "source_chunk_ids": chunks}


def test_numbers_in_handles_indian_grouping():
    assert 1234.56 in numbers_in("Rs 1,234.56 crore") and 1234567.0 in numbers_in("12,34,567")


def test_grounded_figure_is_accepted():
    out = validate_figure_extraction(json.dumps({"figures": [_fig(1234.56, ["annual_reports:abc"])]}), CHUNKS)
    assert out.ok and len(out.accepted) == 1


def test_invented_number_is_rejected():
    out = validate_figure_extraction(json.dumps({"figures": [_fig(1500.0, ["annual_reports:abc"])]}), CHUNKS)
    assert not out.ok and "does not appear" in out.rejected[0][1]


def test_number_from_wrong_chunk_is_rejected():
    out = validate_figure_extraction(json.dumps({"figures": [_fig(98.40, ["annual_reports:abc"])]}), CHUNKS)
    assert not out.ok


def test_unknown_citation_is_rejected():
    out = validate_figure_extraction(json.dumps({"figures": [_fig(1234.56, ["annual_reports:zzz"])]}), CHUNKS)
    assert not out.ok and "never retrieved" in out.rejected[0][1]


def test_calculator_output_counts_as_grounded():
    growth = round((1234.56 - 1100.0) / 1100.0 * 100, 2)
    text = json.dumps({"figures": [_fig(growth, ["annual_reports:abc"], metric="revenue_growth_pct")]})
    assert validate_figure_extraction(text, CHUNKS, computed_values=[growth]).ok


def test_malformed_output_fails_closed():
    with pytest.raises(SchemaViolationError):
        validate_figure_extraction("Sure! Revenue was about 1,200 crore.", CHUNKS)
