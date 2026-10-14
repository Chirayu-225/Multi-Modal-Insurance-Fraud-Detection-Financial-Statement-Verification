"""
Trains a learned fusion model and evaluates it the way the project
review insisted on -- not just "does it work" but "how do we know it
isn't just learning the generator":

  1. HELD-OUT INCONSISTENCY TYPES. The model trains on WRONG_PANEL,
     INFLATED_BILL and EXCLUDED_ITEM claims, then is tested on
     REUSED_PHOTO and TIMESTAMP_MISMATCH claims it has never seen the
     shape of. A big drop here is an honest, reportable result, not a
     failure to hide.
  2. REALISTIC BASE RATE. Precision on a 50/50 train-style set looks
     much better than precision when inconsistency is rare. We report
     both.
  3. LEARNED MODEL VS. RULES BASELINE. The whole point of a fusion
     *model* instead of a bigger rules engine is that it should beat the
     rules on the held-out-type set. If it doesn't, that's reported too
     -- not hidden.

Run with: python train_fusion_model.py
Produces: app/ml/artifacts/fusion_model.joblib (loadable by the API when
CONSISTENCY_BACKEND=fusion) and prints the metrics tables below.

NOTE on the "hand-built adversarial test set" the review also asked for:
that has to be built by a person looking for genuinely subtle cases a
script wouldn't think of -- see adversarial_examples.py for a small
illustrative starting template (3 examples) to expand to 50-100 as
described in review. This script does not fabricate that set for you;
doing so would defeat its purpose.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from app.ml.domain import (
    BASE_REPAIR_COST_INR,
    EXPECTED_LOCATIONS,
    SEVERITY_COST_MULTIPLIER,
    ClaimEvidence,
    InconsistencyType,
)
from app.ml.synthetic_data import generate_dataset
from app.services.consistency import ADJACENT_LOCATIONS, _hamming_distance, evaluate

TRAIN_TYPES = [InconsistencyType.WRONG_PANEL, InconsistencyType.INFLATED_BILL, InconsistencyType.EXCLUDED_ITEM]
HELD_OUT_TYPES = [InconsistencyType.REUSED_PHOTO, InconsistencyType.TIMESTAMP_MISMATCH]


def featurize(claims: list[ClaimEvidence]) -> np.ndarray:
    """Engineered features -- deliberately the same signals the rules
    engine looks at, so the comparison is about *combining* signals
    better, not about the model seeing something the rules engine
    couldn't."""
    hashes = {c.claim_id: c.photo_hash for c in claims}
    rows = []
    for c in claims:
        expected = EXPECTED_LOCATIONS[c.described_incident_type]
        if c.photo_damage_location in expected:
            location_mismatch = 0.0
        else:
            adjacent = set()
            for loc in expected:
                adjacent |= ADJACENT_LOCATIONS.get(loc, set())
            location_mismatch = 0.5 if c.photo_damage_location in adjacent else 1.0

        repair_total = sum(i.cost_inr for i in c.bill_line_items if i.category.endswith("_repair"))
        expected_cost = (
            BASE_REPAIR_COST_INR[c.photo_damage_location][c.vehicle_segment]
            * SEVERITY_COST_MULTIPLIER[c.photo_damage_severity]
        )
        cost_ratio = repair_total / expected_cost if expected_cost else 1.0

        has_excluded = float(any(i.category in c.policy_excluded_categories for i in c.bill_line_items))

        best_distance = 64
        for other_id, other_hash in hashes.items():
            if other_id == c.claim_id:
                continue
            best_distance = min(best_distance, _hamming_distance(c.photo_hash, other_hash))

        ts_delta = (c.photo_capture_date - c.incident_date).days if c.photo_capture_date else 0

        rows.append([location_mismatch, cost_ratio, has_excluded, best_distance, ts_delta])
    return np.array(rows, dtype=float)


def labels(claims: list[ClaimEvidence]) -> np.ndarray:
    return np.array([0 if c.injected_inconsistency == InconsistencyType.NONE else 1 for c in claims])


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {name:28s} precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}  (n={len(y_true)}, positives={y_true.sum()})")


def rules_predictions(claims: list[ClaimEvidence], threshold: float = 0.3) -> np.ndarray:
    hashes = {c.claim_id: c.photo_hash for c in claims}
    scores = [evaluate(c, other_photo_hashes=hashes).risk_score for c in claims]
    return (np.array(scores) > threshold).astype(int)


def main() -> None:
    print("=" * 78)
    print("1. Generating held-out-type split")
    print("=" * 78)
    train_claims = generate_dataset(1200, inconsistency_rate=0.5, seed=100, inconsistency_types=TRAIN_TYPES)
    test_heldout = generate_dataset(400, inconsistency_rate=0.5, seed=200, inconsistency_types=HELD_OUT_TYPES)
    test_realistic = generate_dataset(2000, inconsistency_rate=0.05, seed=300, inconsistency_types=None)

    X_train, y_train = featurize(train_claims), labels(train_claims)
    X_heldout, y_heldout = featurize(test_heldout), labels(test_heldout)
    X_realistic, y_realistic = featurize(test_realistic), labels(test_realistic)

    print(f"  train: {len(train_claims)} claims, types={[t.value for t in TRAIN_TYPES]}")
    print(f"  held-out-type test: {len(test_heldout)} claims, types={[t.value for t in HELD_OUT_TYPES]} (NEVER in training)")
    print(f"  realistic-base-rate test: {len(test_realistic)} claims at 5% inconsistency rate, all types")

    print()
    print("=" * 78)
    print("2. Training fusion classifier (RandomForest) on TRAIN_TYPES only")
    print("=" * 78)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, class_weight="balanced")
    clf.fit(X_train, y_train)
    joblib.dump(clf, "app/ml/artifacts/fusion_model.joblib")
    print("  saved -> app/ml/artifacts/fusion_model.joblib")

    print()
    print("=" * 78)
    print("3. Results: learned fusion model vs. rules baseline")
    print("=" * 78)

    print("\n-- On types seen during training (sanity check, not the honest number) --")
    report("fusion model", y_train, clf.predict(X_train))
    report("rules baseline", y_train, rules_predictions(train_claims))

    print("\n-- On HELD-OUT types (reused_photo, timestamp_mismatch) -- THE HONEST NUMBER --")
    report("fusion model", y_heldout, clf.predict(X_heldout))
    report("rules baseline", y_heldout, rules_predictions(test_heldout))

    print("\n-- At realistic 5% base rate, all types --")
    report("fusion model", y_realistic, clf.predict(X_realistic))
    report("rules baseline", y_realistic, rules_predictions(test_realistic))

    print()
    print("Read this as: if the fusion model's held-out-type numbers collapse")
    print("while the rules baseline holds up, that tells you the rules engine's")
    print("hand-coded thresholds generalize better than a model trained only on")
    print("3 of 5 inconsistency types -- which is itself a valid, reportable")
    print("finding about when a learned model is worth it here.")


if __name__ == "__main__":
    import os
    os.makedirs("app/ml/artifacts", exist_ok=True)
    main()
