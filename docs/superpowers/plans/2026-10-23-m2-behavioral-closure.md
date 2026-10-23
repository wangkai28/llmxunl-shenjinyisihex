# M2 行为闭环实验 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接真实网关 LLM，加入行为闭环度量与无写回对照组，产出「预测误差是否收敛」的可证伪结论。

**Architecture:** 扩展 `Reflection` 增加数值预测字段，新增 `GatewayLLMProvider` 调用当前网关（qwen3.8-flash），新增 `metrics.py` 计算预测误差与收敛，新增 `experiment.py` 编排实验组/对照组。结果落 D 盘 `data/`。

**Tech Stack:** Python 3.12、NumPy、openai（OpenAI 兼容 SDK）。

**Spec:** [docs/superpowers/specs/2026-10-23-m2-behavioral-closure-design.md](../specs/2026-10-23-m2-behavioral-closure-design.md)

## Global Constraints

- Python >= 3.10；依赖 numpy、openai>=1.0（已装 2.47.0）。
- 源码布局 `src/bioelectric/`，pytest 通过 `pyproject.toml` 的 `pythonpath = ["src"]` 导入。
- LLM 走环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`，默认模型 `qwen3.8-flash`。
- LLM 解析失败必须降级（不抛异常、不中断实验）：`action_type="none"`、`predicted_bin_activity=None`。
- 所有文件 UTF-8；实验结果持久化到 D 盘 `data/`。
- TDD：每个任务先写失败测试再实现，再提交。

---

## File Structure

```
src/bioelectric/
├── llm_provider.py   # 改：Reflection 加 predicted_bin_activity 字段；新增 GatewayLLMProvider
├── metrics.py        # 新：bin_rate / prediction_error / convergence
├── experiment.py     # 新：run_experiment / ExperimentResult
examples/
├── m2_experiment.py  # 新：跑实验组 + 对照组，打印并落盘
tests/
├── test_metrics.py
├── test_gateway_provider.py
└── test_experiment.py
```

依赖方向（无环）：

```text
spiking / snapshot / memory / llm_provider  →  metrics  →  experiment
```

---

### Task 1: Reflection 扩展 + GatewayLLMProvider

**Files:**
- Modify: `src/bioelectric/llm_provider.py`
- Test: `tests/test_gateway_provider.py`

**Interfaces:**
- Consumes: 现有 `Reflection` dataclass（`self_description`, `action_type`, `region`, `strength`）
- Produces:
  - `Reflection` 新增可选字段 `predicted_bin_activity: list[float] | None = None`（向后兼容，默认 None）
  - `class GatewayLLMProvider(LLMProvider)`：构造 `GatewayLLMProvider(base_url, api_key, model, *, n_bins=8, temperature=0.0)`；`reflect(snapshot_text, history) -> Reflection`
  - `_parse_reflection(raw_text, n_bins) -> Reflection`（模块级函数，纯函数，便于测试）：从 LLM 返回文本解析 JSON，失败返回降级 Reflection

- [ ] **Step 1: 写失败测试（纯函数解析，不依赖网络）**

```python
import pytest

from bioelectric.llm_provider import (
    GatewayLLMProvider,
    _parse_reflection,
    Reflection,
)


def test_parse_reflection_valid_json():
    raw = '{"self_description": "a stable core", "predicted_bin_activity": [0.1, 0.2], "action": {"type": "inject", "region": 0, "strength": 1.0}}'
    r = _parse_reflection(raw, n_bins=8)
    assert isinstance(r, Reflection)
    assert r.self_description == "a stable core"
    assert r.predicted_bin_activity == [0.1, 0.2]
    assert r.action_type == "inject"
    assert r.region == 0
    assert r.strength == 1.0


def test_parse_reflection_missing_action_defaults_none():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1]}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity == [0.1]


def test_parse_reflection_invalid_json_degrades():
    r = _parse_reflection("not json at all", n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity is None


def test_parse_reflection_clamps_prediction_length():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1, 0.2, 0.3]}'
    r = _parse_reflection(raw, n_bins=2)
    assert r.predicted_bin_activity == [0.1, 0.2]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_gateway_provider.py -v
```
预期：FAIL（`_parse_reflection` 未定义 / `predicted_bin_activity` 字段不存在）。

- [ ] **Step 3: 实现**

在 `src/bioelectric/llm_provider.py`：

1. `Reflection` dataclass 增加字段 `predicted_bin_activity: list[float] | None = None`（放在 `strength` 之后，保证旧位置参数不变）。
2. 新增模块级 `_parse_reflection(raw_text, n_bins)`：

```python
import json
import re


def _parse_reflection(raw_text: str, n_bins: int) -> Reflection:
    def _degraded():
        return Reflection(self_description="", action_type="none",
                          region=0, strength=0.0, predicted_bin_activity=None)

    if not raw_text:
        return _degraded()
    # 容错：剥离 markdown 代码块围栏
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _degraded()
    if not isinstance(data, dict):
        return _degraded()
    action = data.get("action") or {}
    pred = data.get("predicted_bin_activity")
    if pred is not None:
        try:
            pred = [float(x) for x in pred[:n_bins]]
        except (TypeError, ValueError):
            pred = None
    return Reflection(
        self_description=str(data.get("self_description", "")),
        action_type=str(action.get("type", "none")),
        region=int(action.get("region", 0)),
        strength=float(action.get("strength", 0.0)),
        predicted_bin_activity=pred,
    )
```

3. 新增 `GatewayLLMProvider`：

```python
class GatewayLLMProvider(LLMProvider):
    def __init__(self, base_url, api_key, model, *, n_bins=8, temperature=0.0):
        from openai import OpenAI  # 延迟导入，保持 mock 场景零依赖
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.n_bins = n_bins
        self.temperature = temperature

    def reflect(self, snapshot_text, history) -> Reflection:
        prompt = self._build_prompt(snapshot_text, history)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            return Reflection(self_description="", action_type="none",
                              region=0, strength=0.0, predicted_bin_activity=None)
        return _parse_reflection(raw, self.n_bins)

    def _build_prompt(self, snapshot_text, history):
        hist = "\n".join(f"- {h}" for h in history[-5:]) or "(none)"
        return (
            "You are the self-referential observer of a spiking neural network.\n"
            "Current activity snapshot: " + snapshot_text + "\n"
            "Your recent self-reflections:\n" + hist + "\n"
            "Respond with ONLY a JSON object, no prose, with keys:\n"
            '{"self_description": string, "predicted_bin_activity": list of '
            + str(self.n_bins) + ' floats in [0,1] (predicted firing rate per bin), '
            '"action": {"type": "inject" or "none", "region": int, "strength": float}}\n'
        )
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_gateway_provider.py -v
```
预期：4 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/llm_provider.py tests/test_gateway_provider.py
git commit -m "feat: add predicted_bin_activity to Reflection and GatewayLLMProvider"
```

---

### Task 2: 度量模块

**Files:**
- Create: `src/bioelectric/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `SpikingNetwork`（读 `.spikes`）、`bin_bounds`
- Produces:
  - `bin_rate(network, bounds) -> np.ndarray`：各 bin 放电率（spike 数 / bin 大小），shape `(len(bounds),)`
  - `prediction_error(predicted, actual) -> float | None`：MAE；`predicted` 为 None 返回 None
  - `convergence(errors: list[float]) -> dict`：`{"slope": float, "head_mean": float, "tail_mean": float, "converged": bool}`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np

from bioelectric.metrics import bin_rate, convergence, prediction_error
from bioelectric.snapshot import bin_bounds
from bioelectric.spiking import SpikingNetwork


def test_bin_rate():
    net = SpikingNetwork(n_units=4, seed=0)
    net.spikes = np.array([True, True, False, False])
    bounds = bin_bounds(4, 2)
    rates = bin_rate(net, bounds)
    assert rates.shape == (2,)
    assert rates[0] == 1.0
    assert rates[1] == 0.0


def test_prediction_error_mae():
    assert prediction_error([0.2, 0.4], np.array([0.0, 0.4])) == 0.1
    assert prediction_error(None, np.array([0.0, 0.4])) is None


def test_convergence_decreasing_series():
    result = convergence([0.5, 0.4, 0.3, 0.2, 0.1])
    assert result["slope"] < 0
    assert result["converged"] is True


def test_convergence_flat_series_not_converged():
    result = convergence([0.3, 0.3, 0.3, 0.3, 0.3])
    assert result["converged"] is False


def test_convergence_too_short():
    result = convergence([0.5, 0.4])
    assert result["converged"] is False
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_metrics.py -v
```
预期：FAIL（`No module named 'bioelectric.metrics'`）。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations

import numpy as np

from .snapshot import bin_bounds
from .spiking import SpikingNetwork


def bin_rate(network: SpikingNetwork, bounds) -> np.ndarray:
    spikes = network.spikes
    return np.array([
        spikes[start:end].mean() if end > start else 0.0
        for start, end in bounds
    ], dtype=float)


def prediction_error(predicted, actual) -> float | None:
    if predicted is None:
        return None
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError("predicted and actual must have the same shape")
    return float(np.mean(np.abs(p - a)))


def convergence(errors: list[float]) -> dict:
    valid = [float(e) for e in errors if e is not None]
    if len(valid) < 4:
        return {"slope": 0.0, "head_mean": float("nan"),
                "tail_mean": float("nan"), "converged": False}
    x = np.arange(len(valid), dtype=float)
    y = np.array(valid, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    half = len(valid) // 2
    head_mean = float(np.mean(y[:half]))
    tail_mean = float(np.mean(y[half:]))
    converged = bool(slope < 0 and tail_mean < head_mean)
    return {"slope": slope, "head_mean": head_mean,
            "tail_mean": tail_mean, "converged": converged}
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_metrics.py -v
```
预期：5 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/metrics.py tests/test_metrics.py
git commit -m "feat: add behavioral-closure metrics (bin_rate, prediction_error, convergence)"
```

---

### Task 3: 实验编排

**Files:**
- Create: `src/bioelectric/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `SpikingNetwork`、`Observer`、`MockLLMProvider`、`bin_rate`、`prediction_error`、`convergence`、`bin_bounds`
- Produces:
  - `@dataclass ExperimentResult`：字段 `errors: list[float|None]`、`self_descriptions: list[str]`、`convergence: dict`
  - `run_experiment(network, provider, *, mode, steps, observe_every, n_bins) -> ExperimentResult`

- [ ] **Step 1: 写失败测试（用 mock，不依赖网络）**

```python
from bioelectric.experiment import ExperimentResult, run_experiment
from bioelectric.llm_provider import MockLLMProvider
from bioelectric.spiking import SpikingNetwork


def test_run_experiment_collects_metrics():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(
        description="stable core",
        action_type="none",
    )
    result = run_experiment(
        net, provider, mode="readwrite", steps=20, observe_every=5, n_bins=2,
    )
    assert isinstance(result, ExperimentResult)
    assert len(result.self_descriptions) == 4  # 观察点：5,10,15,20
    # Mock 不输出预测，因此误差全为 None
    assert all(e is None for e in result.errors)
    assert result.convergence["converged"] is False


def test_run_experiment_readonly_same_shape():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(description="x")
    rw = run_experiment(net, provider, mode="readwrite", steps=10, observe_every=5, n_bins=2)
    net2 = SpikingNetwork(n_units=8, seed=0)
    ro = run_experiment(net2, provider, mode="readonly", steps=10, observe_every=5, n_bins=2)
    assert len(rw.errors) == len(ro.errors)
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
python -m pytest tests/test_experiment.py -v
```
预期：FAIL（`No module named 'bioelectric.experiment'`）。

- [ ] **Step 3: 实现**

```python
from __future__ import annotations

from dataclasses import dataclass, field

from .llm_provider import LLMProvider
from .metrics import bin_rate, convergence, prediction_error
from .observer import Observer
from .snapshot import bin_bounds
from .spiking import SpikingNetwork


@dataclass
class ExperimentResult:
    errors: list = field(default_factory=list)
    self_descriptions: list = field(default_factory=list)
    convergence: dict = field(default_factory=dict)


def run_experiment(
    network: SpikingNetwork,
    provider: LLMProvider,
    *,
    mode: str,
    steps: int,
    observe_every: int,
    n_bins: int,
) -> ExperimentResult:
    observer = Observer(
        network, provider, observe_every=observe_every,
        n_bins=n_bins, mode=mode,
    )
    bounds = bin_bounds(network.n_units, n_bins)
    result = ExperimentResult()
    pending_prediction = None
    for _ in range(steps):
        reflection = observer.tick()
        if reflection is None:
            continue
        actual = bin_rate(network, bounds)
        if pending_prediction is not None:
            result.errors.append(prediction_error(pending_prediction, actual))
        result.self_descriptions.append(reflection.self_description)
        pending_prediction = reflection.predicted_bin_activity
    result.convergence = convergence(result.errors)
    return result
```

注：误差的时序——LLM 在观察点 `t` 给出的预测，要到下一个观察点 `t+1` 才用实测活动校验，因此第一个观察点的误差为空，误差序列长度 = 观察点数 - 1。这符合 spec §1.1「预测下一时刻活动」的语义。

- [ ] **Step 4: 运行测试确认通过**

```powershell
python -m pytest tests/test_experiment.py -v
```
预期：2 passed。

- [ ] **Step 5: 提交**

```powershell
git add src/bioelectric/experiment.py tests/test_experiment.py
git commit -m "feat: add experiment orchestration (run_experiment, ExperimentResult)"
```

---

### Task 4: 端到端实验脚本 + 落盘

**Files:**
- Create: `examples/m2_experiment.py`

**Interfaces:**
- Consumes: `run_experiment`、`GatewayLLMProvider`、`SpikingNetwork`
- Produces: 跑实验组/对照组，打印收敛结果并写 `data/` 下的 JSON

- [ ] **Step 1: 写脚本**

```python
import json
import os
from datetime import datetime

from bioelectric.experiment import run_experiment
from bioelectric.llm_provider import GatewayLLMProvider
from bioelectric.spiking import SpikingNetwork


def main():
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "qwen3.8-flash")
    if not base_url or not api_key:
        raise SystemExit("OPENAI_BASE_URL / OPENAI_API_KEY not configured")

    provider = GatewayLLMProvider(base_url, api_key, model, n_bins=8)
    cfg = dict(steps=60, observe_every=10, n_bins=8)

    rw = run_experiment(
        SpikingNetwork(n_units=64, seed=42), provider,
        mode="readwrite", **cfg,
    )
    ro = run_experiment(
        SpikingNetwork(n_units=64, seed=42), provider,
        mode="readonly", **cfg,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "config": cfg,
        "readwrite": {"errors": rw.errors, "convergence": rw.convergence},
        "readonly": {"errors": ro.errors, "convergence": ro.convergence},
    }
    os.makedirs("data", exist_ok=True)
    path = f"data/m2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("== M2 behavioral-closure experiment ==")
    print(f"model={model} steps={cfg['steps']} observe_every={cfg['observe_every']}")
    print(f"readwrite: converged={rw.convergence['converged']} slope={rw.convergence['slope']:.4f} errors={[round(e,3) if e is not None else None for e in rw.errors]}")
    print(f"readonly : converged={ro.convergence['converged']} slope={ro.convergence['slope']:.4f} errors={[round(e,3) if e is not None else None for e in ro.errors]}")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑全量测试**

```powershell
python -m pytest -q
```
预期：全部通过（原 24 + 新增 11 = 35 passed）。

- [ ] **Step 3: 提交**

```powershell
git add examples/m2_experiment.py
git commit -m "feat: add M2 end-to-end experiment script"
```

---

## Self-Review 记录

- **Spec 覆盖**：M2a/M2b 的 `GatewayLLMProvider`（§3.2）、`metrics`（§3.3）、`experiment`（§3.4）、误差定义（§1.1）、对照组（§2）均有对应任务；`Reflection` 扩展（§3.1）在 Task 1。
- **占位符扫描**：无 TBD/TODO；所有代码与测试给出完整内容。
- **类型一致性**：`predicted_bin_activity` 字段名在 `Reflection`、`_parse_reflection`、`GatewayLLMProvider._build_prompt`、`run_experiment` 中一致；`convergence` 返回键（slope/head_mean/tail_mean/converged）在 `metrics` 与 `experiment` 中一致。
