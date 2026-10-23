from __future__ import annotations

import json
from pathlib import Path


class ReflectionMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = int(capacity)
        self._items: list[str] = []

    def add(self, self_description: str) -> None:
        self._items.append(self_description)
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity:]

    def recent(self, k: int = 10) -> list[str]:
        return list(self._items[-k:])

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(self._items, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path, capacity: int = 100) -> "ReflectionMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls(capacity=capacity)
        memory._items = data
        return memory
