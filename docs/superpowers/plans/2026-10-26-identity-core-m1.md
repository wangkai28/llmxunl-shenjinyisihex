# 身份核心 M1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 组装一个最小 `IdentityCore`，跑通「事件 → 状态变化 → 查询 → 保存 → 恢复」闭环。

**Architecture:** 复用 `balanced_network.py`、`multi_self.py`、`persona.py`，新建 `event.py`（事件编码）、`continuity.py`（快照+自传）、`identity_core.py`（组装四层）。

**Tech Stack:** Python 3.12、NumPy、json（标准库）。

**Spec:** [docs/superpowers/specs/2026-10-26-identity-core-design.md](../specs/2026-10-26-identity-core-design.md)

## Global Constraints

- Python >= 3.10；仅 numpy + 标准库。
- 源码 `src/bioelectric/`，pytest 通过 `pyproject.toml` 的 `pythonpath = ["src"]`。
- 所有文件 UTF-8；持久化用 JSON（人类可读）。
- TDD：每个任务先写失败测试再实现。
- 不改已有的 `balanced_network.py`/`multi_self.py`/`persona.py`（状态层和身份层复用）。

---

## File Structure

```
src/bioelectric/
├── event.py           # 新：Event dataclass + 事件类型 → 情境偏置映射
├── continuity.py      # 新：状态快照 + 自传日志
├── identity_core.py   # 新：IdentityCore 组装四层
tests/
├── test_event.py
├── test_continuity.py
└── test_identity_core.py
```

---

### Task 1: 事件编码（event.py）

**Files:**
- Create: `src/bioelectric/event.py`
- Test: `tests/test_event.py`

**Interfaces:**
- Produces:
  - `@dataclass Event`：字段 `type: str`、`intensity: float = 1.0`、`content: str = ""`
  - `EVENT_TO_CONTEXT: dict[str, dict]`：事件类型 → 情境偏置（如 `"danger"` → `{"id": 3.0, "superego": 0.2, "tao": 0.2}`）
  - `encode_event(event: Event) -> dict`：返回该事件类型对应的情境偏置（intensity 缩放主导自我）

- [ ] **Step 1: 写失败测试**

```python
from bioelectric.event import Event, EVENT_TO_CONTEXT, encode_event


def test_event_dataclass():
    e = Event(type="danger", intensity=0.8, content="x")
    assert e.type == "danger"
    assert e.intensity == 0.8


def test_encode_event_maps_type_to_context():
    ctx = encode_event(Event(type="danger", intensity=1.0))
    assert ctx["id"] > ctx["superego"]  # 危险 → 本我主导


def test_encode_event_scales_by_intensity():
    full = encode_event(Event(type="danger", intensity=1.0))
    weak = encode_event(Event(type="danger", intensity=0.3))
    assert weak["id"] < full["id"]


def test_unknown_event_type_raises():
    import pytest
    with pytest.raises(ValueError):
        encode_event(Event(type="nonexistent", intensity=1.0))
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_event.py -v
```
预期：FAIL（`No module named 'bioelectric.event'`）。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Event:
    type: str
    intensity: float = 1.0
    content: str = ""


# 事件类型 → 情境偏置（与 multi_self.set_context 的语义一致）
EVENT_TO_CONTEXT = {
    "danger": {"id": 3.0, "superego": 0.2, "tao": 0.2},
    "moral": {"id": 0.2, "superego": 3.0, "tao": 0.2},
    "calm": {"id": 0.2, "superego": 0.2, "tao": 3.0},
    "praise": {"id": 1.5, "superego": 1.2, "tao": 1.5},
    "criticism": {"id": 1.8, "superego": 2.2, "tao": 1.0},
    "threat": {"id": 3.0, "superego": 0.5, "tao": 0.5},
}


def encode_event(event: Event) -> dict:
    if event.type not in EVENT_TO_CONTEXT:
        raise ValueError(f"unknown event type {event.type!r}")
    base = dict(EVENT_TO_CONTEXT[event.type])
    # 主导自我（值最大者）的偏置随 intensity 缩放
    dominant = max(base, key=base.get)
    base[dominant] = base[dominant] * max(0.0, min(1.0, event.intensity))
    return base
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_event.py -v
```
预期：4 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/event.py tests/test_event.py
git commit -m "feat: event encoding (type -> context bias)"
```

---

### Task 2: 连续性（continuity.py）

**Files:**
- Create: `src/bioelectric/continuity.py`
- Test: `tests/test_continuity.py`

**Interfaces:**
- Produces:
  - `save_snapshot(state: dict, path: str) -> None`：写 JSON
  - `load_snapshot(path: str) -> dict`：读 JSON；文件不存在返回 `{}`
  - `append_autobiography(path: str, entry: dict) -> None`：追加一行 JSONL
  - `read_autobiography(path: str) -> list[dict]`：读全部 JSONL

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_continuity.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_continuity.py -v
```
预期：3 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/continuity.py tests/test_continuity.py
git commit -m "feat: snapshot + autobiography continuity"
```

---

### Task 3: 身份核心组装（identity_core.py）

**Files:**
- Create: `src/bioelectric/identity_core.py`
- Test: `tests/test_identity_core.py`

**Interfaces:**
- Consumes: `Event`/`encode_event`、`MultiSelf`/`set_context`/`dominant_self`/`measure_multi_self`、`continuity`
- Produces:
  - `class IdentityCore`：构造 `IdentityCore(persona="tao", state_dir="data/identity", seed=0)`
  - `process_event(event) -> dict`：处理事件（编码→设情境→驱动→返回状态）
  - `query() -> dict`：返回当前主导自我 + 各自我活动率
  - `save()` / `load()`：持久化 / 恢复

- [ ] **Step 1: 写失败测试**

```python
from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


def test_process_event_changes_dominant_self(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    # 危险事件 → 本我主导
    state = core.process_event(Event(type="danger", intensity=1.0))
    assert state["dominant"] == "id"
    # 平静事件 → 道家主导
    state = core.process_event(Event(type="calm", intensity=1.0))
    assert state["dominant"] == "tao"


def test_save_and_load_preserves_identity(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="moral", intensity=1.0))
    core.save()
    # 新建一个核心，加载后应恢复主导自我
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2.query()["dominant"] == "superego"


def test_query_returns_activity(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    state = core.query()
    assert "dominant" in state
    assert "id" in state and "superego" in state and "tao" in state
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations

import os

from .balanced_network import BalancedNetwork
from .continuity import save_snapshot, load_snapshot, append_autobiography
from .event import Event, encode_event
from .multi_self import MultiSelf, set_context, dominant_self, measure_multi_self


class IdentityCore:
    def __init__(self, *, persona: str, state_dir: str, seed: int = 0):
        self.persona = persona
        self.state_dir = state_dir
        self.seed = seed
        self._network = BalancedNetwork(
            n_units=600, k_exc=10, j_inh=-4.5,
            external_mean=0.15, external_noise=0.08, delay_max=4, seed=seed,
        )
        self._ms = MultiSelf(
            self._network,
            core_specs={
                "id": {"size": 50, "bias": 3.0},
                "superego": {"size": 50, "bias": 0.5},
                "tao": {"size": 50, "bias": 1.5},
            },
            seed=1, hebb_lr=0.02, cross_inhibition=3.0,
        )
        self._ms.form(steps=200)

    def process_event(self, event: Event) -> dict:
        ctx = encode_event(event)
        set_context(self._ms, ctx)
        # 驱动一段让情境生效
        for _ in range(50):
            dominant = max(ctx, key=ctx.get)
            from .multi_self import _drive
            _drive(self._ms, dominant)
        state = self.query()
        append_autobiography(
            os.path.join(self.state_dir, "autobiography.jsonl"),
            {"event": event.type, "intensity": event.intensity,
             "content": event.content, "dominant": state["dominant"]},
        )
        return state

    def query(self) -> dict:
        dims = measure_multi_self(self._ms, steps=50)
        names = list(self._ms.core_masks.keys())
        dominant = max(names, key=lambda n: dims[n]["activity"])
        result = {"dominant": dominant}
        for n in names:
            result[n] = dims[n]["activity"]
        return result

    def save(self):
        save_snapshot(self.query(), os.path.join(self.state_dir, "snapshot.json"))

    def load(self):
        # 恢复快照中的主导自我为当前情境
        snap = load_snapshot(os.path.join(self.state_dir, "snapshot.json"))
        if snap.get("dominant"):
            # 按主导自我设一个对应情境
            ctx = {"id": 0.2, "superego": 0.2, "tao": 0.2}
            ctx[snap["dominant"]] = 3.0
            set_context(self._ms, ctx)
            for _ in range(50):
                from .multi_self import _drive
                _drive(self._ms, snap["dominant"])
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：3 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/identity_core.py tests/test_identity_core.py
git commit -m "feat: IdentityCore assembles event/state/continuity/persona"
```

---

## Self-Review 记录

- Spec 覆盖：M1 目标（事件→状态→查询→保存→恢复）由 Task 1（事件）、Task 2（连续性）、Task 3（组装）覆盖；对应 spec §4.1/§4.2/§5。
- 占位符：无 TBD/TODO。
- 类型一致性：`encode_event(event)->dict`、`MultiSelf.set_context`、`dominant_self` 的语义在 Task 1 与 Task 3 中一致；`_drive(ms, dominant)` 是 `multi_self` 已有内部函数，Task 3 用 `from .multi_self import _drive` 导入。
