# 五大人格（OCEAN）层 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增 OCEAN 五大人格层（E/N/C 三特质映射 + 极缓慢演化），作为性格参数之上的稳定底色。

**Architecture:** 新增 `ocean.py`（OCEAN→params 映射 + 极缓慢演化），改造 `identity_core.py`（存 `_ocean`，用其派生性格参数，重大事件演化 OCEAN 再派生）。

**Tech Stack:** Python 3.12、NumPy。

**Spec:** [docs/superpowers/specs/2026-10-28-ocean-personality-design.md](../specs/2026-10-28-ocean-personality-design.md)

## Global Constraints

- Python >= 3.10；numpy + stdlib。
- 源码 `src/bioelectric/`，pytest 通过 `pythonpath = ["src"]`。
- UTF-8；TDD。
- 不改 balanced_network/multi_self/event/continuity/persona_evolution/persona。

---

## File Structure

```
src/bioelectric/
├── ocean.py            # 新：OCEAN → params 映射 + 极缓慢演化
├── identity_core.py    # 改：存 _ocean，派生性格参数，演化 OCEAN
tests/
├── test_ocean.py
└── test_identity_core.py  # 改/加
```

---

### Task 1: OCEAN 映射与演化（ocean.py）

**Files:**
- Create: `src/bioelectric/ocean.py`
- Test: `tests/test_ocean.py`

**Interfaces:**
- Produces:
  - `DEFAULT_OCEAN: dict`（5 个特质，键 O/E/C/A/N，值 0~1）
  - `ocean_to_params(ocean: dict) -> dict`：返回 `{external_mean, external_noise, j_inh}`
  - `evolve_ocean(ocean: dict, event_type: str, *, lr: float = 0.005) -> dict`：返回新的 ocean（不改原）
  - `OCEAN_EVOLUTION_DIRECTION: dict`：事件类型 → 各特质方向

- [ ] **Step 1: 写失败测试**

```python
from bioelectric.ocean import DEFAULT_OCEAN, ocean_to_params, evolve_ocean


def test_ocean_to_params_maps_extraversion_to_activity():
    low_e = ocean_to_params({**DEFAULT_OCEAN, "E": 0.2})
    high_e = ocean_to_params({**DEFAULT_OCEAN, "E": 0.8})
    assert high_e["external_mean"] > low_e["external_mean"]


def test_ocean_to_params_maps_neuroticism_to_noise():
    low_n = ocean_to_params({**DEFAULT_OCEAN, "N": 0.2})
    high_n = ocean_to_params({**DEFAULT_OCEAN, "N": 0.8})
    assert high_n["external_noise"] > low_n["external_noise"]


def test_ocean_to_params_maps_conscientiousness_to_inhibition():
    low_c = ocean_to_params({**DEFAULT_OCEAN, "C": 0.2})
    high_c = ocean_to_params({**DEFAULT_OCEAN, "C": 0.8})
    assert abs(high_c["j_inh"]) > abs(low_c["j_inh"])  # 越尽责抑制越强


def test_evolve_ocean_does_not_mutate():
    original = dict(DEFAULT_OCEAN)
    evolve_ocean(DEFAULT_OCEAN, "danger")
    assert DEFAULT_OCEAN == original


def test_evolve_ocean_danger_raises_neuroticism():
    new = evolve_ocean(DEFAULT_OCEAN, "danger", lr=0.05)
    assert new["N"] > DEFAULT_OCEAN["N"]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_ocean.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations


DEFAULT_OCEAN = {
    "O": 0.5,  # 开放性
    "E": 0.5,  # 外向性
    "C": 0.5,  # 尽责性
    "A": 0.5,  # 宜人性
    "N": 0.5,  # 神经质
}


def ocean_to_params(ocean: dict) -> dict:
    E = float(ocean.get("E", 0.5))
    N = float(ocean.get("N", 0.5))
    C = float(ocean.get("C", 0.5))
    return {
        "external_mean": 0.05 + 0.45 * E,
        "external_noise": 0.02 + 0.38 * N,
        "j_inh": -(1.5 + 6.5 * C),
    }


# 事件类型 → OCEAN 各特质演化方向（+1 增 / -1 减 / 0 不变）
OCEAN_EVOLUTION_DIRECTION = {
    "danger": {"N": +1, "C": -1, "E": +1, "O": 0, "A": 0},
    "threat": {"N": +1, "C": -1, "E": +1, "O": 0, "A": 0},
    "moral": {"N": -1, "C": +1, "E": 0, "O": 0, "A": 0},
    "criticism": {"N": +1, "C": +1, "E": 0, "O": 0, "A": 0},
    "calm": {"N": -1, "C": 0, "E": 0, "O": 0, "A": 0},
    "praise": {"N": -1, "C": 0, "E": +1, "O": 0, "A": 0},
}


def evolve_ocean(ocean: dict, event_type: str, *, lr: float = 0.005) -> dict:
    direction = OCEAN_EVOLUTION_DIRECTION.get(event_type)
    new = dict(ocean)
    if direction is None:
        return new
    for key, sign in direction.items():
        if key not in new or sign == 0:
            continue
        delta = lr * sign
        new[key] = max(0.0, min(1.0, new[key] + delta))
    return new
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_ocean.py -v
```
预期：5 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/ocean.py tests/test_ocean.py
git commit -m "feat: OCEAN five-factor layer (mapping + slow evolution)"
```

---

### Task 2: 集成 OCEAN 到 IdentityCore

**Files:**
- Modify: `src/bioelectric/identity_core.py`
- Test: `tests/test_identity_core.py`

**Interfaces:**
- Consumes: `ocean_to_params`、`evolve_ocean`、`DEFAULT_OCEAN`
- Produces: `IdentityCore` 存 `_ocean`；构造时用 `ocean_to_params(_ocean)` 派生性格参数初始值；重大事件先演化 `_ocean` 再派生性格参数；`query()` 输出 `ocean`

- [ ] **Step 1: 写失败测试**

```python
def test_ocean_derives_initial_persona_params(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    assert hasattr(core, "_ocean")
    assert set(core._ocean.keys()) == {"O", "E", "C", "A", "N"}


def test_query_includes_ocean(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    state = core.query()
    assert "ocean" in state
    assert "O" in state["ocean"]


def test_ocean_evolves_slowly(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    before_n = core._ocean["N"]
    for _ in range(20):
        core.process_event(Event(type="danger", intensity=1.0))
    assert core._ocean["N"] > before_n  # 神经质缓慢上升
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：FAIL。

- [ ] **Step 3: 实现**

在 `IdentityCore`：

1. `__init__` 里：
   ```python
   self._ocean = dict(DEFAULT_OCEAN)
   # 性格参数初始值由 OCEAN 派生（覆盖 PERSONAS 的硬编码）
   self._persona_cfg = ocean_to_params(self._ocean)
   ```
   网络构造用 `self._persona_cfg`（已有逻辑）。

2. `process_event` 里 significant 时，先演化 OCEAN 再派生性格参数：
   ```python
   if significant:
       self._ocean = evolve_ocean(self._ocean, event.type)
       self._persona_cfg = ocean_to_params(self._ocean)
       self._apply_persona_to_network()
   ```

3. `query()` 返回的 dict 加 `"ocean": dict(self._ocean)`。

4. `save()` 快照加 `"ocean": self._ocean`；`load()` 恢复 `_ocean`（若快照有）。

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_identity_core.py -v
```
预期：全部通过。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/identity_core.py tests/test_identity_core.py
git commit -m "feat: integrate OCEAN layer into IdentityCore (derive params + slow evolution + output)"
```

---

## Self-Review 记录

- Spec 覆盖：OCEAN 层（映射 + 极缓慢演化 + 输出）由 Task 1（ocean.py）、Task 2（集成）覆盖；对应 spec §1/§2/§3/§4。
- 占位符：无 TBD/TODO。
- 类型一致性：`ocean_to_params`、`evolve_ocean`、`DEFAULT_OCEAN` 在 Task 1/2 一致。
- 注意：Task 2 用 `ocean_to_params` 覆盖 `PERSONAS[persona]` 的性格参数，意味着 persona 参数（j_inh/external_mean/external_noise）不再直接来自 PERSONAS，而来自 OCEAN。这是一个语义变化——persona 的默认 OCEAN 目前是 DEFAULT_OCEAN（全 0.5），后续可让 persona 映射到不同 OCEAN。此点已记录，M3 的 `_persona_cfg` 仍保留用于演化与持久化。
