from bioelectric.continuity import (
    save_snapshot, load_snapshot, append_autobiography, read_autobiography,
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
