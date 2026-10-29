# 身份核心 M2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把连续性做扎实：完整状态快照（含活动率）、自传历史查询、重大事件标记，并让 persona 参数真正生效。

**Architecture:** 补 `continuity.py`（完整快照 + 历史查询），增强 `identity_core.py`（persona 生效 + 完整保存/恢复 + history + 重大事件标记）。复用 event/multi_self/persona。

**Tech Stack:** Python 3.12、NumPy、json（标准库）。

**Spec:** [docs/superpowers/specs/2026-10-26-identity-core-design.md](../specs/2026-10-26-identity-core-design.md)

## Global Constraints

- Python >= 3.10；仅 numpy + 标准库。
- 源码 `src/bioelectric/`，pytest 通过 `pyproject.toml` 的 `pythonpath = ["src"]`。
- 所有文件 UTF-8；持久化用 JSON。
- TDD。
- 不改 `balanced_network.py`/`multi_self.py`/`event.py`。

---

## File Structure

```
src/bioelectric/
├── continuity.py       # 改：快照存完整状态；加历史查询
├── identity_core.py    # 改：persona 生效 + 完整 save/load + history + 重大事件标记
tests/
├── test_continuity.py  # 改/加
└── test_identity_core.py  # 改/加
```

---

### Task 1: persona 参数生效

**Files:**
- Modify: `src/bioelectric/identity_core.py`
- Test: `tests/test_identity_core.py`

**Interfaces:**
- Consumes: `persona.PERSONAS`（`build_persona_network` 的映射）
- Produces: `IdentityCore(persona=...)` 的 `persona` 参数真正影响网络性格参数

- [ ] **Step 1: 写失败测试**

```python
def test_persona_param_changes_network(tmp_path):
    # 不同 persona 应产生不同的网络性格参数
    id_core = IdentityCore(persona="id", state_dir=str(tmp_path / "a"), seed=0)
    super_core = IdentityCore(persona="superego", state_dir=str(tmp_path / "b"), seed=0)
    # 本我应比超我有更强的外部驱动（冲动）
    assert id_core._network.external_mean > super_core._network.external_mean
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_identity_core.py::test_persona_param_changes_network -v
```
预期：FAIL（当前 external_mean 恒为 0.15，两者相等）。

- [ ] **Step 3: 实现**

在 `IdentityCore.__init__` 里，用 `PERSONAS[persona]` 的网络参数构造 `BalancedNetwork`：

```python
from .persona import PERSONAS

cfg = PERSONAS.get(persona)
if cfg is None:
    raise ValueError(f"unknown persona {persona!r}")
self._network = BalancedNetwork(
    n_units=600, k_exc=10, j_inh=cfg["j_inh"],
    external_mean=cfg["external_mean"], external_noise=cfg["external_noise"],
    delay_max=4, seed=seed,
)
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：全部通过。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/identity_core.py tests/test_identity_core.py
git commit -m "feat: wire persona param into IdentityCore network"
```

---

### Task 2: 完整状态快照 + 历史查询

**Files:**
- Modify: `src/bioelectric/continuity.py`
- Test: `tests/test_continuity.py`

**Interfaces:**
- Produces:
  - `query_history(path: str, limit: int | None = None) -> list[dict]`：读自传（按时间正序），可选 limit 返回最近 N 条
  - `count_history(path: str) -> int`：自传条数

- [ ] **Step 1: 写失败测试**

```python
def test_query_history_limit(tmp_path):
    path = str(tmp_path / "a.jsonl")
    for i in range(5):
        append_autobiography(path, {"i": i})
    assert len(query_history(path)) == 5
    assert query_history(path, limit=2) == [{"i": 3}, {"i": 4}]
    assert count_history(path) == 5
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_continuity.py::test_query_history_limit -v
```
预期：FAIL（`query_history`/`count_history` 不存在）。

- [ ] **Step 3: 实现**

```python
def query_history(path: str, limit: int | None = None) -> list[dict]:
    entries = read_autobiography(path)
    if limit is not None:
        return entries[-limit:]
    return entries


def count_history(path: str) -> int:
    return len(read_autobiography(path))
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_continuity.py -v
```
预期：全部通过。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/continuity.py tests/test_continuity.py
git commit -m "feat: history query (limit + count) for autobiography"
```

---

### Task 3: 完整 save/load + history 接口 + 重大事件标记

**Files:**
- Modify: `src/bioelectric/identity_core.py`
- Test: `tests/test_identity_core.py`

**Interfaces:**
- Produces:
  - `save()` 保存完整状态（dominant + context_bias + activities）
  - `load()` 恢复完整状态
  - `history(limit=None) -> list[dict]`：读自传历史
  - `process_event` 对高强度事件（intensity >= significant_threshold）在自传里标记 `"significant": true`

- [ ] **Step 1: 写失败测试**

```python
def test_save_persists_activities(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    core.save()
    import json, os
    snap = json.loads(open(os.path.join(str(tmp_path / "id"), "snapshot.json"), encoding="utf-8").read())
    assert "activities" in snap
    assert "tao" in snap["activities"]


def test_history_returns_entries(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="danger", intensity=1.0))
    core.process_event(Event(type="calm", intensity=1.0))
    hist = core.history()
    assert len(hist) == 2
    assert hist[0]["event"] == "danger"


def test_significant_event_marked(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0,
                        significant_threshold=0.8)
    core.process_event(Event(type="danger", intensity=0.9))
    core.process_event(Event(type="calm", intensity=0.3))
    hist = core.history()
    assert hist[0]["significant"] is True
    assert hist[1]["significant"] is False
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

`IdentityCore.__init__` 增加 `significant_threshold: float = 0.8`。

`save()`：
```python
def save(self):
    state = self.query()
    snapshot = {
        "dominant": state["dominant"],
        "context_bias": dict(self._ms.context_bias),
        "activities": {k: state[k] for k in self._ms.core_masks.keys()},
    }
    save_snapshot(snapshot, os.path.join(self.state_dir, "snapshot.json"))
```

`process_event` 记录 significant：
```python
def process_event(self, event):
    ...  # 原有逻辑
    significant = event.intensity >= self.significant_threshold
    append_autobiography(..., {..., "significant": significant})
    return state
```

`history()`：
```python
def history(self, limit=None):
    return query_history(os.path.join(self.state_dir, "autobiography.jsonl"), limit)
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：全部通过。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/identity_core.py tests/test_identity_core.py
git commit -m "feat: full snapshot + history + significant-event marking"
```

---

## Self-Review 记录

- Spec 覆盖：M2 目标（完整快照、自传历史、重大事件标记、persona 生效）由 Task 1/2/3 覆盖；对应 spec §3/§4.2/§6 里程碑 M2。
- 占位符：无 TBD/TODO。
- 类型一致性：`query_history`/`count_history`（Task 2）被 `IdentityCore.history`（Task 3）消费，签名一致；`PERSONAS`（persona.py 已有）在 Task 1 消费。
