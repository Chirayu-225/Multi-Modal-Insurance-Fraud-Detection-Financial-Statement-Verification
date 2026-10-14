"""Score combination shared by every rule pack.

Noisy-OR is the same combination services/consistency.py has always used
for claims; it is pulled out here so the underwriting pack combines its
flags identically instead of inventing a second scoring rule.
"""
from __future__ import annotations

from collections.abc import Iterable


def noisy_or(severities: Iterable[float]) -> float:
    remaining = 1.0
    for s in severities:
        remaining *= (1.0 - s)
    return round(1.0 - remaining, 4)
