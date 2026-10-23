# M1 最小可跑闭环 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭出一个最小可跑闭环——数十个脉冲神经元组成的网络，上面挂一个 LLM 观察者，跑通「读 → 反思 → 写回 → 再读」，并验证接口与节律正确。

**Architecture:** 底层是 LIF 脉冲网络 + Hebb 可塑性（`spiking.py`），上层是观察者回路（`observer.py`），中间以「快照压缩器（`snapshot.py`）→ LLM Provider（`llm_provider.py`）→ 反思记忆（`memory.py`）」串联。M1 用 `MockLLMProvider` 离线跑通，不依赖真实 LLM。

**Tech Stack:** Python 3.12、NumPy、pytest。

**Spec:** [docs/superpowers/specs/2026-10-22-bioelectric-emergence-design.md](../specs/2026-10-22-bioelectric-emergence-design.md)

## Global Constraints

- Python >= 3.10，依赖仅 `numpy>=2.0`（测试用 pytest）。
- 源码布局 `src/bioelectric/`，pytest 通过 `pyproject.toml` 的 `pythonpath = ["src"]` 导入。
- 所有文件 UTF-8 编码；数据与日志持久化到 D 盘（M1 尚未落盘，接口预留）。
- 每个任务以「写失败测试 → 跑红 → 最小实现 → 跑绿 → commit」顺序完成，禁止先写实现再补测试。
- 底层网络与压缩器为确定性实现（固定 seed / 纯函数），保证可复现。

---

## File Structure

```
bioelectric-emergence/
├── pyproject.toml
├── README.md
├── src/bioelectric/
│   ├── __init__.py
│   ├── spiking.py        # 脉冲网络核：LIF + Hebb + 传播
│   ├── snapshot.py       # 快照压缩器：活动摘要 -> LLM 文本
│   ├── memory.py         # 反思记录存储（内存 + 可选 JSON 持久化）
│   ├── llm_provider.py   # LLM Provider 抽象 + Mock 实现
│   └── observer.py       # 观察者回路：读 -> 反思 -> 写回 -> 再读
├── examples/
│   └── m1_loop.py        # 端到端最小演示（离线）
└── tests/
    ├── test_spiking.py
    ├── test_snapshot.py
    ├── test_memory.py
    ├── test_llm_provider.py
    └── test_observer.py
```

模块依赖方向（单向，无环）：

```text
spiking.py  ←  snapshot.py  ←  observer.py
                                 ↑
memory.py  ──  llm_provider.py ──┘
```

---

### Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/bioelectric/__init__.py`
- Create: `tests/__init__.py`（空，保证测试包可发现）

**Interfaces:**
- Consumes: 无
- Produces: 项目根、包结构、pytest 配置；后续所有任务的导入前缀 `bioelectric.*`

- [ ] **Step 1: 初始化 git 仓库**

```powershell
cd D:\CodexProjects\bioelectric-emergence
git init
```

- [ ] **Step 2: 创建 pyproject.toml**

```toml
[project]
name = "bioelectric-emergence"
version = "0.1.0"
description = "生物电涌现实验：研究自我认知是否在脉冲网络 + LLM 观察者的闭合回路中涌现"
requires-python = ">=3.10"
dependencies = ["numpy>=2.0"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: 创建 README.md**

```markdown
# 生物电涌现实验

研究「自我认知反应」能否在「生物电脉冲网络 + LLM 观察者」的闭合回路中涌现。
详见 docs/superpowers/specs/2026-10-22-bioelectric-emergence-design.md
```

- [ ] **Step 4: 创建包与测试占位文件**

创建 `src/bioelectric/__init__.py`（内容：`__version__ = "0.1.0"`）与空的 `tests/__init__.py`。

- [ ] **Step 5: 验证配置与提交**

```powershell
python -c "import sys; sys.path.insert(0, 'src'); import bioelectric; print(bioelectric.__version__)"
git add -A
git commit -m "chore: scaffold bioelectric-emergence project"
```

---

### Task 2: 脉冲网络核（LIF + Hebb）

**Files:**
- Create: `src/bioelectric/spiking.py`
- Test: `tests/test_spiking.py`

**Interfaces:**
- Consumes: 无（仅 NumPy）
- Produces:
  - `class SpikingNetwork`：构造 `SpikingNetwork(n_units, *, connectivity=0.1, threshold=1.0, refractory=2, leak=0.1, dt=1.0, weight_scale=0.5, hebb_lr=0.01, weight_decay=0.001, weight_max=2.0, seed=0)`
  - 属性：`n_units`、`W`（突触权重矩阵，`W[i,j]` 为 j→i）、`V`（膜电位）、`refrac`（不应期剩余）、`spikes`（当前脉冲 bool 向量）
  - 方法：`step(external_input=None) -> np.ndarray`：推进一个时间步，返回本步发放的 bool 向量

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from bioelectric.spiking import SpikingNetwork


def test_step_returns_boolean_spike_vector():
    net = SpikingNetwork(n_units=20, seed=0)
    spikes = net.step()
    assert spikes.shape == (20,)
    assert spikes.dtype == bool


def test_strong_input_causes_firing():
    net = SpikingNetwork(n_units=10, seed=0)
    ext = np.zeros(10)
    ext[0] = 100.0
    fired = net.step(external_input=ext)
    assert fired[0]


def test_refractory_period_blocks_refiring():
    net = SpikingNetwork(
        n_units=1, connectivity=0.0, threshold=0.5, refractory=3,
        leak=0.0, dt=1.0, seed=0,
    )
    ext = np.array([10.0])
    assert net.step(external_input=ext)[0]
    for _ in range(3):
        assert not net.step(external_input=ext)[0]


def test_hebb_strengthens_coactive_pair():
    net = SpikingNetwork(
        n_units=2, connectivity=0.0, threshold=0.5, refractory=0,
        leak=0.0, dt=1.0, hebb_lr=0.1, weight_decay=0.0, weight_max=2.0, seed=0,
    )
    before = net.W.copy()
    net.step(external_input=np.array([10.0, 10.0]))
    assert net.W[0, 1] > before[0, 1]
    assert net.W[1, 0] > before[1, 0]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_spiking.py -v
```
预期：全部 FAIL（`ModuleNotFoundError: No module named 'bioelectric'`）。

- [ ] **Step 3: 最小实现**

```python
from __future__ import annotations

import numpy as np


class SpikingNetwork:
    def __init__(
        self,
        n_units,
        *,
        connectivity=0.1,
        threshold=1.0,
        refractory=2,
        leak=0.1,
        dt=1.0,
        weight_scale=0.5,
        hebb_lr=0.01,
        weight_decay=0.001,
        weight_max=2.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.n_units = int(n_units)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.leak = float(leak)
        self.dt = float(dt)
        self.hebb_lr = float(hebb_lr)
        self.weight_decay = float(weight_decay)
        self.weight_max = float(weight_max)
        self.W = np.zeros((self.n_units, self.n_units), dtype=float)
        if connectivity > 0:
            mask = rng.random((self.n_units, self.n_units)) < connectivity
            self.W[mask] = rng.normal(0.0, weight_scale, size=int(mask.sum()))
        np.fill_diagonal(self.W, 0.0)
        self.V = rng.uniform(0.0, self.threshold, size=self.n_units)
        self.refrac = np.zeros(self.n_units, dtype=int)
        self.spikes = np.zeros(self.n_units, dtype=bool)

    def step(self, external_input=None) -> np.ndarray:
        ext = (
            np.zeros(self.n_units)
            if external_input is None
            else np.asarray(external_input, dtype=float)
        )
        synaptic = self.W @ self.spikes.astype(float)
        self.V = self.V + (synaptic + ext - self.leak * self.V) * self.dt
        fired = (self.V >= self.threshold) & (self.refrac == 0)
        self.spikes = fired
        self.V[fired] = 0.0
        self.refrac[fired] = self.refractory
        not_fired = ~fired
        self.refrac[not_fired] = np.maximum(self.refrac[not_fired] - 1, 0)
        self._hebb_update(fired)
        return fired.copy()

    def _hebb_update(self, fired):
        co = np.outer(fired.astype(float), fired.astype(float))
        np.fill_diagonal(co, 0.0)
        self.W = self.W + self.hebb_lr * co * (self.weight_max - self.W)
        self.W = self.W - self.weight_decay * self.W
        self.W = np.clip(self.W, 0.0, self.weight_max)
        np.fill_diagonal(self.W, 0.0)
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_spiking.py -v
```
预期：4 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/spiking.py tests/test_spiking.py
git commit -m "feat: add LIF spiking network with Hebbian plasticity"
```

---

### Task 3: 快照压缩器

**Files:**
- Create: `src/bioelectric/snapshot.py`
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `bioelectric.spiking.SpikingNetwork`（读 `.spikes`）
- Produces:
  - `bin_bounds(n_units: int, n_bins: int) -> list[tuple[int, int]]`：返回每个 bin 的 `[start, end)` 区间
  - `summarize(network, n_bins=8) -> dict`：返回 `{"active_count": int, "total_units": int, "bin_activity": list[int]}`
  - `to_text(summary: dict) -> str`：把 summary 转成 LLM 可读文本

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from bioelectric.spiking import SpikingNetwork
from bioelectric.snapshot import bin_bounds, summarize, to_text


def test_bin_bounds_cover_all_units():
    bounds = bin_bounds(10, 3)
    covered = []
    for start, end in bounds:
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(10))


def test_summarize_counts_active():
    net = SpikingNetwork(n_units=10, seed=0)
    net.spikes = np.array([True, True, False, False, False, False, False, False, False, False])
    summary = summarize(net, n_bins=2)
    assert summary["active_count"] == 2
    assert summary["total_units"] == 10
    assert sum(summary["bin_activity"]) == 2


def test_to_text_contains_numbers():
    text = to_text({"active_count": 2, "total_units": 10, "bin_activity": [2, 0]})
    assert "2" in text
    assert "10" in text
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_snapshot.py -v
```
预期：FAIL（`No module named 'bioelectric.snapshot'`）。

- [ ] **Step 3: 最小实现**

```python
from __future__ import annotations

import numpy as np

from .spiking import SpikingNetwork


def bin_bounds(n_units: int, n_bins: int) -> list[tuple[int, int]]:
    splits = np.array_split(np.arange(n_units), n_bins)
    return [(int(s[0]), int(s[-1]) + 1) for s in splits]


def summarize(network: SpikingNetwork, n_bins: int = 8) -> dict:
    spikes = network.spikes
    bounds = bin_bounds(len(spikes), n_bins)
    bin_activity = [int(spikes[start:end].sum()) for start, end in bounds]
    return {
        "active_count": int(spikes.sum()),
        "total_units": int(len(spikes)),
        "bin_activity": bin_activity,
    }


def to_text(summary: dict) -> str:
    return (
        f"active_count={summary['active_count']} "
        f"total_units={summary['total_units']} "
        f"bin_activity={summary['bin_activity']}"
    )
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_snapshot.py -v
```
预期：3 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/snapshot.py tests/test_snapshot.py
git commit -m "feat: add activity snapshot compressor"
```

---

### Task 4: 反思记录存储

**Files:**
- Create: `src/bioelectric/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: 无（仅标准库 json/pathlib）
- Produces:
  - `class ReflectionMemory`：构造 `ReflectionMemory(capacity=100)`
  - `add(self_description: str) -> None`：追加一条反思，超出容量淘汰最旧
  - `recent(k=10) -> list[str]`：返回最近 k 条（按时间正序）
  - `save(path) -> None` / `load(path, capacity=100) -> ReflectionMemory`：JSON 持久化

- [ ] **Step 1: 写失败测试**

```python
from bioelectric.memory import ReflectionMemory


def test_add_and_recent():
    m = ReflectionMemory(capacity=5)
    m.add("one")
    m.add("two")
    assert m.recent(10) == ["one", "two"]


def test_capacity_evicts_oldest():
    m = ReflectionMemory(capacity=2)
    m.add("a")
    m.add("b")
    m.add("c")
    assert m.recent(10) == ["b", "c"]


def test_save_and_load(tmp_path):
    m = ReflectionMemory()
    m.add("x")
    path = tmp_path / "mem.json"
    m.save(str(path))
    m2 = ReflectionMemory.load(str(path))
    assert m2.recent(10) == ["x"]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_memory.py -v
```
预期：FAIL（`No module named 'bioelectric.memory'`）。

- [ ] **Step 3: 最小实现**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_memory.py -v
```
预期：3 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/memory.py tests/test_memory.py
git commit -m "feat: add reflection memory store"
```

---

### Task 5: LLM Provider（抽象 + Mock）

**Files:**
- Create: `src/bioelectric/llm_provider.py`
- Test: `tests/test_llm_provider.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `@dataclass class Reflection`：字段 `self_description: str`、`action_type: str`（`"inject"` 或 `"none"`）、`region: int = 0`、`strength: float = 0.0`
  - `class LLMProvider`：`reflect(snapshot_text: str, history: list[str]) -> Reflection`，基类抛 `NotImplementedError`
  - `class MockLLMProvider(LLMProvider)`：构造 `MockLLMProvider(*, description="a network", action_type="none", region=0, strength=0.0)`，`reflect` 返回固定 `Reflection`

- [ ] **Step 1: 写失败测试**

```python
import pytest

from bioelectric.llm_provider import LLMProvider, MockLLMProvider, Reflection


def test_mock_returns_fixed_reflection():
    provider = MockLLMProvider(
        description="I am a network",
        action_type="inject",
        region=2,
        strength=0.5,
    )
    result = provider.reflect("snapshot", [])
    assert isinstance(result, Reflection)
    assert result.self_description == "I am a network"
    assert result.action_type == "inject"
    assert result.region == 2
    assert result.strength == 0.5


def test_base_provider_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMProvider().reflect("snapshot", [])
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_llm_provider.py -v
```
预期：FAIL（`No module named 'bioelectric.llm_provider'`）。

- [ ] **Step 3: 最小实现**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reflection:
    self_description: str
    action_type: str
    region: int = 0
    strength: float = 0.0


class LLMProvider:
    def reflect(self, snapshot_text: str, history: list[str]) -> Reflection:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        description: str = "a network",
        action_type: str = "none",
        region: int = 0,
        strength: float = 0.0,
    ):
        self.description = description
        self.action_type = action_type
        self.region = region
        self.strength = strength

    def reflect(self, snapshot_text, history) -> Reflection:
        return Reflection(
            self_description=self.description,
            action_type=self.action_type,
            region=self.region,
            strength=self.strength,
        )
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_llm_provider.py -v
```
预期：2 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/llm_provider.py tests/test_llm_provider.py
git commit -m "feat: add LLM provider abstraction and mock"
```

---

### Task 6: 观察者回路

**Files:**
- Create: `src/bioelectric/observer.py`
- Test: `tests/test_observer.py`

**Interfaces:**
- Consumes: `SpikingNetwork.step`、`bin_bounds`/`summarize`/`to_text`、`ReflectionMemory`、`LLMProvider.reflect`
- Produces:
  - `class Observer`：构造 `Observer(network, provider, memory=None, *, observe_every=10, n_bins=8, mode="readwrite", history_window=5)`
  - `tick() -> Reflection | None`：推进一个时间步；到观察节律则执行「读 → 反思 → 写回」并返回 `Reflection`，否则返回 `None`
  - `run(steps: int) -> list[Reflection]`：推进 steps 步，返回全部产生的 `Reflection`
  - `mode` 取值 `"readwrite"`（写回）或 `"readonly"`（对照组，反思但不写回）

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from bioelectric.llm_provider import MockLLMProvider
from bioelectric.observer import Observer
from bioelectric.spiking import SpikingNetwork


def test_observer_emits_reflections_at_interval():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(description="stable core")
    observer = Observer(net, provider, observe_every=5, n_bins=2, mode="readwrite")
    reflections = observer.run(12)
    assert len(reflections) == 2  # 第 5、10 步各观察一次


def test_observer_writeback_reaches_network():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=100.0, leak=0.0, dt=1.0, seed=0
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=1000.0
    )
    observer = Observer(net, provider, observe_every=1, n_bins=2, mode="readwrite")
    observer.tick()  # 观察并写入 pending input
    observer.tick()  # 下一时间步注入生效
    assert net.spikes[0] or net.spikes[1]


def test_observer_readonly_does_not_write_back():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=100.0, leak=0.0, dt=1.0, seed=0
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=1000.0
    )
    observer = Observer(net, provider, observe_every=1, n_bins=2, mode="readonly")
    observer.tick()
    observer.tick()
    assert not (net.spikes[0] or net.spikes[1])
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_observer.py -v
```
预期：FAIL（`No module named 'bioelectric.observer'`）。

- [ ] **Step 3: 最小实现**

```python
from __future__ import annotations

import numpy as np

from .llm_provider import LLMProvider, Reflection
from .memory import ReflectionMemory
from .snapshot import bin_bounds, summarize, to_text
from .spiking import SpikingNetwork


class Observer:
    def __init__(
        self,
        network: SpikingNetwork,
        provider: LLMProvider,
        memory: ReflectionMemory | None = None,
        *,
        observe_every: int = 10,
        n_bins: int = 8,
        mode: str = "readwrite",
        history_window: int = 5,
    ):
        self.network = network
        self.provider = provider
        self.memory = memory if memory is not None else ReflectionMemory()
        self.observe_every = observe_every
        self.n_bins = n_bins
        self.mode = mode
        self.history_window = history_window
        self.step_count = 0
        self.pending_input = np.zeros(network.n_units, dtype=float)
        self.bounds = bin_bounds(network.n_units, n_bins)

    def tick(self) -> Reflection | None:
        self.network.step(self.pending_input)
        self.pending_input.fill(0.0)
        self.step_count += 1
        if self.step_count % self.observe_every == 0:
            return self._observe()
        return None

    def run(self, steps: int) -> list[Reflection]:
        reflections = []
        for _ in range(steps):
            result = self.tick()
            if result is not None:
                reflections.append(result)
        return reflections

    def _observe(self) -> Reflection:
        summary = summarize(self.network, self.n_bins)
        text = to_text(summary)
        history = self.memory.recent(self.history_window)
        reflection = self.provider.reflect(text, history)
        self.memory.add(reflection.self_description)
        if self.mode == "readwrite" and reflection.action_type == "inject":
            self._writeback(reflection)
        return reflection

    def _writeback(self, reflection: Reflection) -> None:
        region = reflection.region
        if 0 <= region < len(self.bounds):
            start, end = self.bounds[region]
            self.pending_input[start:end] += reflection.strength
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_observer.py -v
```
预期：3 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/observer.py tests/test_observer.py
git commit -m "feat: add observer loop with read-reflect-writeback"
```

---

### Task 7: 端到端最小演示 + 全量验证

**Files:**
- Create: `examples/m1_loop.py`

**Interfaces:**
- Consumes: `SpikingNetwork`、`Observer`、`MockLLMProvider`
- Produces: 一个可离线运行的端到端演示，打印每次反思

- [ ] **Step 1: 写演示脚本**

```python
from bioelectric.llm_provider import MockLLMProvider
from bioelectric.observer import Observer
from bioelectric.spiking import SpikingNetwork


def main():
    network = SpikingNetwork(n_units=64, seed=42)
    provider = MockLLMProvider(
        description="a network with a stable core and a quiet periphery",
        action_type="inject",
        region=0,
        strength=2.0,
    )
    observer = Observer(network, provider, observe_every=10, n_bins=8, mode="readwrite")
    reflections = observer.run(50)
    for i, reflection in enumerate(reflections):
        print(
            f"[{i}] {reflection.self_description} | "
            f"action={reflection.action_type} region={reflection.region} "
            f"strength={reflection.strength}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行演示**

```powershell
python examples/m1_loop.py
```
预期：打印 5 条反思（第 10/20/30/40/50 步），无异常。

- [ ] **Step 3: 全量测试**

```powershell
python -m pytest -v
```
预期：全部通过（15 个测试）。

- [ ] **Step 4: 提交**

```powershell
git add examples/m1_loop.py
git commit -m "feat: add M1 end-to-end demo"
```

---

## Self-Review 记录

- **Spec 覆盖**：本计划仅实现里程碑 M1（最小可跑闭环），对应 spec §7 模块 1–5 与 §12 里程碑 M1。度量（§7 模块 6 `metrics.py`）、实验编排（模块 7 `experiment.py`）与对照组正式度量属于 M2，不在此计划内，已在里程碑边界说明。
- **占位符扫描**：无 TBD/TODO；所有实现与测试代码均已给出完整内容。
- **类型一致性**：`SpikingNetwork.step`、`bin_bounds`、`summarize`、`to_text`、`ReflectionMemory`、`Reflection`、`MockLLMProvider.reflect`、`Observer.tick/run` 的签名在各任务间一致；`Reflection` 字段名 `self_description` 与 `MockLLMProvider` 的 `description` 参数已区分，`MockLLMProvider.reflect` 内部正确映射。
