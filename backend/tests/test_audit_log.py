"""
Regression tests for the hash-chained audit log's pure logic --
compute_entry_hash, verify_entry_self_consistency, verify_full_chain.
No DB involved: entries are plain namespace objects with the same
attributes an AuditLogEntry ORM row has, which is exactly what
verify_entry_self_consistency's docstring says it accepts. This
mirrors the append_entry logic (prev_hash chains forward, hash is a
function of every stored field) without needing a running Postgres
instance, so this file alone proves the chain math is correct.
"""
from types import SimpleNamespace

from app.services.audit_log import (
    GENESIS_HASH,
    compute_entry_hash,
    make_payload_json,
    verify_entry_self_consistency,
    verify_full_chain,
)


def _make_chain(n: int) -> list[SimpleNamespace]:
    """Builds n entries the same way append_entry would: each one's
    prev_hash is the previous entry's entry_hash, starting from
    GENESIS_HASH."""
    entries = []
    prev_hash = GENESIS_HASH
    for i in range(n):
        payload_json = make_payload_json({"risk_score": 0.1 * i, "i": i})
        timestamp_iso = f"2026-09-27T00:00:{i:02d}+00:00"
        entry_hash = compute_entry_hash(prev_hash, f"CLM-{i}", "consistency_evaluated", payload_json, "rules-v1", timestamp_iso)
        entries.append(
            SimpleNamespace(
                claim_id=f"CLM-{i}",
                event_type="consistency_evaluated",
                payload_json=payload_json,
                model_version="rules-v1",
                timestamp_iso=timestamp_iso,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
        )
        prev_hash = entry_hash
    return entries


def test_compute_entry_hash_is_deterministic_and_sensitive_to_every_field():
    base = compute_entry_hash(GENESIS_HASH, "CLM-1", "consistency_evaluated", '{"a": 1}', "rules-v1", "2026-09-27T00:00:00+00:00")
    same = compute_entry_hash(GENESIS_HASH, "CLM-1", "consistency_evaluated", '{"a": 1}', "rules-v1", "2026-09-27T00:00:00+00:00")
    assert base == same

    # Changing any single field -- including prev_hash -- must change the hash.
    assert base != compute_entry_hash("1" * 64, "CLM-1", "consistency_evaluated", '{"a": 1}', "rules-v1", "2026-09-27T00:00:00+00:00")
    assert base != compute_entry_hash(GENESIS_HASH, "CLM-2", "consistency_evaluated", '{"a": 1}', "rules-v1", "2026-09-27T00:00:00+00:00")
    assert base != compute_entry_hash(GENESIS_HASH, "CLM-1", "consistency_evaluated", '{"a": 2}', "rules-v1", "2026-09-27T00:00:00+00:00")
    assert base != compute_entry_hash(GENESIS_HASH, "CLM-1", "consistency_evaluated", '{"a": 1}', "rules-v2", "2026-09-27T00:00:00+00:00")
    assert base != compute_entry_hash(GENESIS_HASH, "CLM-1", "consistency_evaluated", '{"a": 1}', "rules-v1", "2026-09-27T00:00:01+00:00")


def test_make_payload_json_is_stable_regardless_of_dict_key_order():
    a = make_payload_json({"risk_score": 0.5, "model_version": "rules-v1"})
    b = make_payload_json({"model_version": "rules-v1", "risk_score": 0.5})
    assert a == b


def test_untampered_chain_verifies_true_at_both_levels():
    entries = _make_chain(10)
    for e in entries:
        assert verify_entry_self_consistency(e)
    is_valid, message = verify_full_chain(entries)
    assert is_valid
    assert "10 entries" in message


def test_editing_an_entrys_payload_after_the_fact_is_caught_by_self_consistency():
    entries = _make_chain(5)
    tampered = entries[2]
    # Simulate someone UPDATEing the row directly in the DB, changing
    # the reported risk score without recomputing the hash.
    tampered.payload_json = make_payload_json({"risk_score": 0.0, "i": 2})

    assert not verify_entry_self_consistency(tampered)
    is_valid, message = verify_full_chain(entries)
    assert not is_valid
    assert "entry 2" in message
    assert "edited" in message


def test_deleting_an_entry_out_of_band_is_caught_by_full_chain_but_not_by_self_consistency_alone():
    entries = _make_chain(5)
    # Remove entry 2 entirely -- each remaining row is still internally
    # self-consistent (its own hash still matches its own fields), which
    # is exactly why verify_full_chain (checking prev_hash linkage), not
    # verify_entry_self_consistency alone, is needed to catch this.
    del entries[2]
    for e in entries:
        assert verify_entry_self_consistency(e)

    is_valid, message = verify_full_chain(entries)
    assert not is_valid
    assert "missing, reordered, or inserted out of band" in message


def test_reordering_entries_is_caught_by_full_chain():
    entries = _make_chain(4)
    entries[1], entries[2] = entries[2], entries[1]
    is_valid, _ = verify_full_chain(entries)
    assert not is_valid


def test_empty_chain_verifies_true_trivially():
    is_valid, message = verify_full_chain([])
    assert is_valid
    assert "0 entries" in message
