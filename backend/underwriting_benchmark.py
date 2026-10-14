"""
Underwriting rule pack on synthetic cases: per-type detection, false
positives on clean cases, and precision/recall at a realistic base rate.

Read the numbers this prints the way the original README reads the
injection-scanner recall: the rules and the generator were written
together, so near-perfect detection here is a consistency check that
the pack does what it says, NOT evidence it works on real submissions.
The meaningful underwriting experiment needs Phase 3: real audited
figures from public annual reports, rendered submitted statements
(including clean re-rendered forgeries), and real extraction in the
loop, where OCR and retrieval errors can actually cause misses.

Run from backend/:  python underwriting_benchmark.py
"""
from __future__ import annotations

import random

from app.ml.underwriting_domain import UnderwritingInconsistencyType as UT
from app.ml.underwriting_synthetic import GENERATED_TYPES, generate_case
from app.services.rules.underwriting import evaluate_underwriting


def per_type(n: int = 500, seed: int = 21) -> None:
    rng = random.Random(seed)
    print(f"{'type':<24}{'detected':>10}{'cross-flags':>13}")
    for kind in GENERATED_TYPES:
        results = [evaluate_underwriting(generate_case(rng, kind)) for _ in range(n)]
        detected = sum(kind in r.triggered_types for r in results) / n
        cross = sum(bool(r.triggered_types - {kind}) for r in results) / n
        print(f"{kind.value:<24}{detected:>10.1%}{cross:>13.1%}")
    clean = [evaluate_underwriting(generate_case(rng, UT.NONE)) for _ in range(n)]
    print(f"{'clean (false positive)':<24}{sum(bool(r.flags) for r in clean) / n:>10.1%}")


def realistic(n: int = 4000, base_rate: float = 0.05, seed: int = 22) -> None:
    rng = random.Random(seed)
    tp = fp = fn = tn = 0
    for _ in range(n):
        kind = rng.choice(GENERATED_TYPES) if rng.random() < base_rate else UT.NONE
        flagged = not evaluate_underwriting(generate_case(rng, kind)).fast_track
        positive = kind != UT.NONE
        tp += flagged and positive
        fp += flagged and not positive
        fn += (not flagged) and positive
        tn += (not flagged) and not positive
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nRealistic {base_rate:.0%} base rate, n={n}: positives={tp + fn}, "
          f"precision={precision:.2f}, recall={recall:.2f}, genuine fast-tracked={tn / max(tn + fp, 1):.1%}")



if __name__ == "__main__":
    per_type()
    realistic()
