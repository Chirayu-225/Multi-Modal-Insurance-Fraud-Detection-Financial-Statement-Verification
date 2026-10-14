"""
Privacy guard: decides what may leave the machine, and scrubs what does.

Two layers, deliberately independent:

1. Sensitivity gate (the real control). Every piece of content carries a
   Sensitivity tag set by its *source* (app/core/case.py). RESTRICTED
   content -- anything a claimant or applicant submitted -- is refused
   outright for hosted free-tier APIs. This is what implements the
   original README's DPDP point: real personal data needs a local model
   or a contracted no-retention API, never a free tier whose terms allow
   prompts to be used to improve the provider's products.

2. Redaction (defence in depth). Text that *is* allowed out (public or
   synthetic) is still scrubbed of identifier patterns, because public
   documents can quote personal data and synthetic data can accidentally
   resemble it. Regex-only for now; an NER pass for personal names is a
   listed next step.

Status: real and unit-tested. Patterns cover common Indian identifier
formats; they are a safety net, not a guarantee -- the sensitivity gate
is what actually keeps restricted data local.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.core.case import Sensitivity


class RestrictedContentError(PermissionError):
    """Raised when restricted content is about to be sent to a hosted
    model. Callers should route to a local model instead, never catch
    this and retry against another hosted provider."""


# Order matters: GSTIN contains a PAN, and an Aadhaar-like 12-digit run
# would otherwise be eaten by the generic account-number pattern.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)|(?<!\d)\d{12}(?!\d)")),
    ("VEHICLE_REG", re.compile(r"\b[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{4}\b")),
    ("ACCOUNT_NO", re.compile(r"(?<!\d)\d{9,18}(?!\d)")),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted_anything(self) -> bool:
        return any(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return RedactionResult(text, counts)


def assert_may_leave_machine(sensitivity: Sensitivity) -> None:
    if sensitivity == Sensitivity.RESTRICTED and not settings.allow_restricted_to_hosted:
        raise RestrictedContentError(
            "Restricted (claimant/applicant-submitted) content may not be sent to a hosted model. "
            "Use a local backend for this step."
        )


def prepare_for_hosted(text: str, sensitivity: Sensitivity) -> RedactionResult:
    """The one function every hosted call goes through: gate first, then
    redact. Returns the text that may actually be sent."""
    assert_may_leave_machine(sensitivity)
    return redact(text)
