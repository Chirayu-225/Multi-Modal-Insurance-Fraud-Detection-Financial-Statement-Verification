"""
Indian GSTIN structural + checksum validation.

A GSTIN is 15 characters: a 2-digit state code, the garage's 10-character
PAN, a 1-character entity/registration code, a literal 'Z', and a final
checksum character computed from the first 14. A fabricated or mistyped
GSTIN on a repair bill fails this check deterministically -- no model,
no training data, just arithmetic -- which is exactly why it's worth
having: it's a real, explainable signal a rules engine can assert with
full confidence, unlike everything else in this project that deals in
graded severities.

Checksum algorithm (verified against the public GST Developer Network
reference implementation, not just recalled from memory): each of the
first 14 characters is mapped to a value 0-35 over the alphabet
"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", multiplied by an alternating
weight (1, 2, 1, 2, ... starting at position 0), folded back into range
with (quotient + remainder) of a division by 36 when the product is
>= 36, summed, and the final check character is
ALPHABET[(36 - (sum % 36)) % 36].

NOTE: this validates that a GSTIN is *well-formed* -- internally
consistent by construction -- not that it's *registered* or that it
belongs to the garage on the bill. Confirming real registration needs a
live lookup against the GST Network's public search API, which is a
reasonable next step but a different (networked, rate-limited, ToS-
governed) piece of work from this offline check.
"""
from __future__ import annotations

import re

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_VALUE = {ch: i for i, ch in enumerate(_ALPHABET)}

# 2 digits (state code) + 10-char PAN pattern + 1 alnum (entity code) +
# literal 'Z' + 1 alnum (checksum).
_STRUCTURE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def compute_checksum_char(gstin_first_14: str) -> str:
    total = 0
    for i, ch in enumerate(gstin_first_14):
        weight = 1 if i % 2 == 0 else 2
        product = _VALUE[ch] * weight
        total += (product // 36) + (product % 36) if product >= 36 else product
    check_value = (36 - (total % 36)) % 36
    return _ALPHABET[check_value]


def validate_gstin(gstin: str | None) -> tuple[bool, str]:
    """Returns (is_valid, reason). `gstin=None` or empty is reported as
    not-present rather than invalid -- callers decide whether "no GSTIN
    extracted at all" is itself worth flagging (a registered garage
    should have one on every bill)."""
    if not gstin:
        return False, "no GSTIN present on the bill"

    candidate = gstin.strip().upper()
    if len(candidate) != 15:
        return False, f"GSTIN must be 15 characters, got {len(candidate)}"
    if not _STRUCTURE_PATTERN.match(candidate):
        return False, "GSTIN does not match the required structure (state code + PAN + entity code + 'Z' + checksum)"
    if any(ch not in _VALUE for ch in candidate):
        return False, "GSTIN contains characters outside the valid alphanumeric set"

    expected_checksum = compute_checksum_char(candidate[:14])
    if candidate[14] != expected_checksum:
        return False, f"checksum character mismatch: expected '{expected_checksum}', got '{candidate[14]}'"

    return True, "valid"


def generate_plausible_gstin(rng) -> str:
    """A random but well-formed, checksum-valid GSTIN -- for synthetic
    data generation and tests, not a real registration. `rng` is any
    object with `.choice()` and `.randint()` (a random.Random instance)."""
    state_code = f"{rng.randint(1, 37):02d}"
    pan_letters_1 = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
    pan_digits = "".join(rng.choice("0123456789") for _ in range(4))
    pan_letter_2 = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    entity_code = rng.choice("123456789")
    first_14 = f"{state_code}{pan_letters_1}{pan_digits}{pan_letter_2}{entity_code}Z"
    return first_14 + compute_checksum_char(first_14)
