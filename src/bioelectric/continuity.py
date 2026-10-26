from __future__ import annotations

import json
from pathlib import Path


def save_snapshot(state: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def append_autobiography(path: str, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_autobiography(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def query_history(path: str, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    entries = read_autobiography(path)
    if limit is not None:
        if limit == 0:
            return []
        return entries[-limit:]
    return entries


def count_history(path: str) -> int:
    return len(read_autobiography(path))
