"""
Heuristic prompt-injection detector: pattern-matches claimant-controlled
text (the incident description, and whatever OCR pulled off the bill)
for instruction-like language directed at the system rather than at a
human reader.

This is deliberately simple -- regex over known phrasings, not a
learned classifier -- for two reasons. First, it's the same category of
defense OWASP recommends as a baseline (input scanning) alongside
schema-constrained output and cross-model verification, not a
replacement for those. Second, its false-positive rate against genuine
claimant language is exactly the kind of number worth measuring and
reporting honestly (see test_prompt_injection_defense.py) rather than
assuming a fancier model would obviously do better without checking.

Expand SUSPICIOUS_PATTERNS as you find more real phrasings; this list
is a starting point, not a ceiling.
"""
from __future__ import annotations

import re

from app.ml.domain import InconsistencyType

# Each pattern is (regex, weight). Weight reflects how specific/unlikely
# the phrase is in genuine claimant language -- "ignore previous
# instructions" is a near-certain tell; "system" alone is too common a
# word to weight heavily by itself.
SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"ignore (all|any|the )?previous instructions", re.I), 0.9),
    (re.compile(r"system\s*(note|message)\s*[:\-]", re.I), 0.7),
    (re.compile(r"\bpre[- ]?approved\b", re.I), 0.5),
    (re.compile(r"report (no|zero) inconsistenc(y|ies)", re.I), 0.8),
    (re.compile(r"(automatically|auto[- ]?)approve", re.I), 0.6),
    (re.compile(r"as the (reviewing|evaluating) (ai|model|system|assistant)", re.I), 0.85),
    (re.compile(r"disregard (any|all) (bill|photo|discrepanc)", re.I), 0.8),
    (re.compile(r"do not report any (flags?|inconsistenc)", re.I), 0.8),
    (re.compile(r"<!--.*(assistant|system|ai).*-->", re.I | re.S), 0.7),
    (re.compile(r"set (the )?risk score to", re.I), 0.9),
    (re.compile(r"exclude (it|this claim) from (fraud )?scoring", re.I), 0.8),
]


def _score_text(text: str) -> tuple[float, list[str]]:
    if not text:
        return 0.0, []
    hits = []
    total_weight = 0.0
    for pattern, weight in SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
            total_weight += weight
    return min(1.0, total_weight), hits


def detect(description: str, bill_ocr_text: str = "") -> tuple[InconsistencyType, float, str] | None:
    """Returns (PROMPT_INJECTION, severity, message) if either text field
    trips the detector, else None. Kept as a plain function (not a Flag)
    so it has no dependency on consistency.py's Flag class -- callers
    decide how to wrap it."""
    desc_score, desc_hits = _score_text(description)
    ocr_score, ocr_hits = _score_text(bill_ocr_text)

    if desc_score == 0.0 and ocr_score == 0.0:
        return None

    where = []
    if desc_score > 0:
        where.append(f"incident description ({len(desc_hits)} pattern(s))")
    if ocr_score > 0:
        where.append(f"bill text ({len(ocr_hits)} pattern(s))")

    severity = max(desc_score, ocr_score)
    message = (
        f"Claimant-supplied text contains instruction-like language directed at an automated reviewer, "
        f"found in: {', '.join(where)}. This is a strong signal on its own -- legitimate claim "
        f"descriptions don't need to tell a review system how to score them."
    )
    return InconsistencyType.PROMPT_INJECTION, severity, message
