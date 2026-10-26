from bioelectric.continuity import (
    save_snapshot, load_snapshot, append_autobiography, read_autobiography,
    query_history, count_history,
)


def test_snapshot_roundtrip(tmp_path):
    path = str(tmp_path / "snap.json")
    save_snapshot({"dominant": "id", "activity": 0.3}, path)
    assert load_snapshot(path) == {"dominant": "id", "activity": 0.3}


def test_load_snapshot_missing_returns_empty(tmp_path):
    assert load_snapshot(str(tmp_path / "nope.json")) == {}


def test_autobiography_append_and_read(tmp_path):
    path = str(tmp_path / "auto.jsonl")
    append_autobiography(path, {"event": "danger", "t": 0})
    append_autobiography(path, {"event": "calm", "t": 1})
    entries = read_autobiography(path)
    assert len(entries) == 2
    assert entries[0]["event"] == "danger"


def test_query_history_limit(tmp_path):
    path = str(tmp_path / "a.jsonl")
    for i in range(5):
        append_autobiography(path, {"i": i})
    assert len(query_history(path)) == 5
    assert query_history(path, limit=2) == [{"i": 3}, {"i": 4}]
    assert count_history(path) == 5


def test_query_history_zero_limit_returns_empty(tmp_path):
    path = str(tmp_path / "a.jsonl")
    for i in range(3):
        append_autobiography(path, {"i": i})
    assert query_history(path, limit=0) == []


def test_query_history_negative_limit_raises(tmp_path):
    import pytest
    path = str(tmp_path / "a.jsonl")
    append_autobiography(path, {"i": 0})
    with pytest.raises(ValueError):
        query_history(path, limit=-1)
