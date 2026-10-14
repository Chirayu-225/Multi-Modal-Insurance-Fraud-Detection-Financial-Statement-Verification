"""
Fast-track analysis: the same risk score everything else in this project
uses to catch fraud, read in the other direction.

The framing worth taking seriously: most claims are genuine, and in
practice the biggest effect of a fraud-checking system on a normal day is
on honest claimants -- slower approvals, repeated evidence requests, a
process that treats everyone like a suspect. The risk score this project
already computes is also a precision/recall trade-off in the *other*
direction: claims with a low score and no flagged inconsistencies can go
on a fast track for quick approval, freeing an adjuster's attention for
the genuinely risky ones.

This is not new modeling work -- it's a threshold choice against the
existing realistic-base-rate benchmark (the same one train_fusion_model.py
reports on), read as "at a target miss rate on fraud, what fraction of
genuine claims can we safely wave through."

Run with: python fast_track_analysis.py
"""
from __future__ import annotations

import numpy as np

from app.ml.synthetic_data import generate_dataset
from app.services.consistency import evaluate

TARGET_MISS_RATES = [0.01, 0.02, 0.05]


def scores_and_labels(claims) -> tuple[np.ndarray, np.ndarray]:
    hashes = {c.claim_id: c.photo_hash for c in claims}
    scores = np.array([evaluate(c, other_photo_hashes=hashes).risk_score for c in claims])
    is_fraud = np.array([0 if c.injected_inconsistency.value == "none" else 1 for c in claims])
    return scores, is_fraud


def threshold_for_miss_rate(scores: np.ndarray, labels: np.ndarray, target_miss_rate: float) -> tuple[float, float]:
    """Returns (threshold, achieved_recall). A claim is flagged for
    review (NOT fast-tracked) when risk_score >= threshold. We want the
    HIGHEST threshold that still catches at least (1 - target_miss_rate)
    of the fraud in this set -- the highest, not just any, threshold,
    because a higher threshold flags fewer claims overall, which is what
    maximizes how many genuine claims get fast-tracked while still
    meeting the miss-rate target.

    Recall is monotonically non-increasing as the threshold rises
    (flagging risk_score >= t catches fewer positives as t grows), so
    scanning candidate thresholds from highest to lowest and returning
    the first one that clears the recall bar is exactly the highest
    threshold that clears it.
    """
    positives = labels.sum()
    if positives == 0:
        return float(scores.max()) + 1.0, 1.0  # nothing to catch -- flag nobody

    for t in sorted(set(scores.tolist()), reverse=True):
        flagged = scores >= t
        caught = int((flagged & (labels == 1)).sum())
        recall = caught / positives
        if recall >= 1 - target_miss_rate:
            return t, recall

    return 0.0, 1.0  # fallback: flagging every claim always achieves recall 1.0


def fast_track_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    genuine_mask = labels == 0
    fast_tracked = (scores < threshold) & genuine_mask
    return fast_tracked.sum() / genuine_mask.sum()


def main() -> None:
    print("=" * 78)
    print("Fast-track analysis: realistic 5% base rate, all fraud types, rules engine")
    print("=" * 78)
    claims = generate_dataset(4000, inconsistency_rate=0.05, seed=500, inconsistency_types=None)
    scores, labels = scores_and_labels(claims)
    print(f"  n={len(claims)}, genuine={int((labels == 0).sum())}, fraud={int(labels.sum())}")
    print()

    print(f"  {'target miss rate':18s} {'threshold':>10s} {'achieved recall':>17s} {'genuine claims fast-tracked':>28s}")
    chosen_threshold = None
    for target in TARGET_MISS_RATES:
        threshold, achieved_recall = threshold_for_miss_rate(scores, labels, target)
        ft_rate = fast_track_rate(scores, labels, threshold)
        print(f"  {target:<18.0%} {threshold:>10.3f} {achieved_recall:>16.1%}  {ft_rate:>27.1%}")
        if target == 0.02:
            chosen_threshold = threshold

    print()
    print("Read this as: at a 2% miss rate on fraud (recall >= 98%), the rules")
    print("engine's own risk score already tells you which genuine claims could")
    print("skip manual review entirely -- this is not a new model, just the")
    print("existing score used in both directions instead of one.")
    print()
    print(f"Chosen default for API's `fast_track` flag: risk_score < {chosen_threshold:.3f}")
    print("(the 2% target row above) -- set as FAST_TRACK_RISK_THRESHOLD in")
    print("app/services/consistency.py. Re-run this script and update that")
    print("constant if the generator or rules engine changes meaningfully.")


if __name__ == "__main__":
    main()
