# 身份核心 M3 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现人格演化：重大事件缓慢改变性格参数，并持久化跨重启保持。

**Architecture:** 新增 `persona_evolution.py`（演化规则，纯函数），改造 `identity_core.py`（性格参数变为可演化的持久状态，process_event 触发演化，save/load 持久化）。

**Tech Stack:** Python 3.12、NumPy、json。

**Spec:** [docs/superpowers/specs/2026-10-26-identity-core-design.md](../specs/2026-10-26-identity-core-design.md) §4.2b

## Global Constraints

- Python >= 3.10；numpy + stdlib。
- 源码 `src/bioelectric/`，pytest 通过 `pythonpath = ["src"]`。
- UTF-8；JSON 持久化。
- TDD。
- 不改 balanced_network/multi_self/event/continuity。

---

## File Structure

```
src/bioelectric/
├── persona_evolution.py   # 新：演化规则（纯函数）
├── identity_core.py       # 改：性格参数可演化 + 持久化
tests/
├── test_persona_evolution.py
└── test_identity_core.py  # 改/加
```

---

### Task 1: 演化规则（persona_evolution.py）

**Files:**
- Create: `src/bioelectric/persona_evolution.py`
- Test: `tests/test_persona_evolution.py`

**Interfaces:**
- Produces:
  - `EVOLUTION_DIRECTION: dict[str, dict]`：事件类型 → 参数演化方向（每参数 +1/-1/0 表示增/减/不变）
  - `evolve_persona(cfg: dict, event_type: str, *, lr: float = 0.05) -> dict`：返回演化后的新 cfg（不修改原 dict）

- [ ] **Step 1: 写失败测试**

```python
from bioelectric.persona_evolution import evolve_persona, EVOLUTION_DIRECTION
from bioelectric.persona import PERSONAS


def test_danger_makes_more_impulsive():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "danger", lr=0.1)
    assert new["external_mean"] > cfg["external_mean"]  # 驱动更强
    assert abs(new["j_inh"]) < abs(cfg["j_inh"])        # 抑制更弱（冲动）


def test_moral_makes_more_restrained():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "moral", lr=0.1)
    assert abs(new["j_inh"]) > abs(cfg["j_inh"])        # 抑制更强（克制）
    assert new["external_mean"] < cfg["external_mean"]  # 驱动更低


def test_evolve_does_not_mutate_input():
    cfg = dict(PERSONAS["tao"])
    original = dict(cfg)
    evolve_persona(cfg, "danger", lr=0.1)
    assert cfg == original


def test_unknown_event_no_change():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "nonexistent", lr=0.1)
    assert new == cfg
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_persona_evolution.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations


# 每个事件类型对性格参数的演化方向：+1 增加、-1 减少、0 不变
EVOLUTION_DIRECTION = {
    "danger": {"external_mean": +1, "j_inh": +1, "external_noise": +1},
    "threat": {"external_mean": +1, "j_inh": +1, "external_noise": +1},
    "moral": {"external_mean": -1, "j_inh": -1, "external_noise": -1},
    "criticism": {"external_mean": -1, "j_inh": -1, "external_noise": 0},
    "calm": {"external_mean": 0, "j_inh": 0, "external_noise": -1},
    "praise": {"external_mean": 0, "j_inh": 0, "external_noise": -1},
}

# 参数边界（防止演化跑飞）
BOUNDS = {
    "j_inh": (-8.0, -1.5),       # 注意 j_inh 是负值，越负抑制越强
    "external_mean": (0.05, 0.5),
    "external_noise": (0.02, 0.4),
}


def evolve_persona(cfg: dict, event_type: str, *, lr: float = 0.05) -> dict:
    direction = EVOLUTION_DIRECTION.get(event_type)
    if direction is None:
        return dict(cfg)
    new = dict(cfg)
    for key, sign in direction.items():
        if key not in new or sign == 0:
            continue
        delta = lr * abs(new[key]) * sign
        lo, hi = BOUNDS[key]
        new[key] = max(lo, min(hi, new[key] + delta))
    return new
```

注：`j_inh` 是负值，`direction=+1` 表示"更弱化抑制"（朝 0 靠近），配合 `delta = lr * abs(value) * sign`，`j_inh + delta` 会减小绝对值（更冲动）。`direction=-1` 则增强抑制。

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_persona_evolution.py -v
```
预期：4 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/persona_evolution.py tests/test_persona_evolution.py
git commit -m "feat: persona evolution rules (event type -> param direction)"
```

---

### Task 2: 集成演化到 IdentityCore + 持久化

**Files:**
- Modify: `src/bioelectric/identity_core.py`
- Test: `tests/test_identity_core.py`

**Interfaces:**
- Consumes: `evolve_persona`
- Produces: `IdentityCore` 在 `process_event` 里对 significant 事件调用 `evolve_persona`；`save` 持久化演化后的 cfg；`load` 恢复它

- [ ] **Step 1: 写失败测试**

```python
def test_significant_event_evolves_persona(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    before = core._persona_cfg["external_mean"]
    # 多次危险重大事件 → 本我倾向更强（external_mean 上升）
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    assert core._persona_cfg["external_mean"] > before


def test_persona_evolution_persists(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    core.save()
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2._persona_cfg["external_mean"] == core._persona_cfg["external_mean"]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：FAIL（`_persona_cfg` 不存在 / 演化未发生）。

- [ ] **Step 3: 实现**

在 `IdentityCore`：

1. `__init__` 里存 `self._persona_cfg = dict(cfg)`（可演化副本）。
2. 构造网络用 `self._persona_cfg`（而非 `PERSONAS[persona]`）。
3. `process_event` 末尾：若 `significant`，`self._persona_cfg = evolve_persona(self._persona_cfg, event.type)`。
4. `save()` 快照里加 `"persona_cfg": self._persona_cfg`。
5. `load()` 里若快照有 `persona_cfg`，则 `self._persona_cfg = snap["persona_cfg"]`。

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：全部通过。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/identity_core.py tests/test_identity_core.py
git commit -m "feat: integrate persona evolution + persistence into IdentityCore"
```

---

## Self-Review 记录

- Spec 覆盖：M3 目标（重大事件触发性格演化 + 持久化）由 Task 1（演化规则）、Task 2（集成+持久化）覆盖；对应 spec §4.2b。
- 占位符：无 TBD/TODO。
- 类型一致性：`evolve_persona(cfg, event_type, lr)` 签名在 Task 1/2 一致；`_persona_cfg` 在 IdentityCore 内部一致。
