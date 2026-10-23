# M2 行为闭环实验设计

- 日期：2026-10-23
- 状态：架构设计（待用户评审）
- 上位依据：`2026-10-22-bioelectric-emergence-design.md` §2（行为闭环指标）、§7（模块 6/7）
- 前置状态：M1 闭环已跑通；E/I balance 地基已补齐（Dale 法则，commit b63d22c）。

## 0. 一句话

把 M1 的 `MockLLMProvider` 换成真实网关模型，让它每次观察时**对网络下一时刻的活动做数值预测**，
再对比预测与实际的偏差，观察这条偏差是否在「读 → 反思 → 写回 → 再读」的闭合回路中收敛；
用「无写回」对照组证伪。

## 1. 核心假设与度量（把自然语言拉回数值空间）

M1 遗留一个关键鸿沟：LLM 的「自我表征」是自然语言，网络「实际状态」是数值，两者不可直接比较。
M2 的解法是让 LLM 每次反思**同时输出数值预测**：

```text
LLM 反思输出 = {
  self_description: str,            # 自然语言自我描述（判据 1 的辅助观察）
  predicted_bin_activity: list[float],  # 预测下一时刻每个 bin 的放电率（[0,1]）
  action: {type, region, strength},     # 写回动作（沿用 M1 的 inject 语义）
}
```

由此定义可证伪的行为闭环度量：

1. **自我表征形成**：`self_description` 随演化保持连续（辅助观察，记录不硬判）。
2. **写回产生定向影响**：写回后网络活动朝 LLM 预测的方向移动。
3. **闭环收敛（核心判据）**：`预测误差` 随观察次数**缩小并稳定**。

### 1.1 预测误差定义

令 `p` 为 LLM 预测的 bin 放电率向量，`a` 为下一观察点实测的 bin 放电率向量（`bin_activity / bin_size`），
则单次误差：

```text
error_t = MAE(p, a) = mean(|p - a|)
```

误差序列 `E = [error_1, error_2, ...]`。收敛判定：对 `E` 做线性回归，斜率显著为负
且尾段均值低于头段均值（下降 + 稳定）。

## 2. 证伪对照（成败关键）

- **实验组**（`mode="readwrite"`）：LLM 反思 + 写回。
- **对照组**（`mode="readonly"`）：同样读、同样反思、同样预测，但**不写回**。

两者共享同一初始网络、同一随机种子、同一 LLM。唯一变量是「是否写回」。
若实验组误差序列收敛而对照组不收敛，才能归因于「闭合回路」而非「LLM 单方面的预测能力」。

## 3. 模块设计

### 3.1 `Reflection` 扩展（改 `llm_provider.py`）

新增可选字段，保持向后兼容（`MockLLMProvider` 与现有测试不受影响）：

- `predicted_bin_activity: list[float] | None = None`

### 3.2 `GatewayLLMProvider`（新，`llm_provider.py` 内）

- 构造：`GatewayLLMProvider(base_url, api_key, model, *, n_bins=8, temperature=0.0)`
- `reflect(snapshot_text, history) -> Reflection`：
  1. 组装 prompt：当前快照 + 历史反思 + 明确要求输出 JSON；
  2. 调用网关 chat.completions（openai SDK）；
  3. 解析 JSON → `Reflection`（含 `self_description`、`predicted_bin_activity`、`action`）；
  4. 解析失败 → 降级为 `action_type="none"`、`predicted_bin_activity=None`，记录告警，不抛异常。
- 依赖：`openai>=1.0`（已装 2.47.0）。

### 3.3 度量（新，`metrics.py`）

- `bin_rate(network, bounds) -> np.ndarray`：实测各 bin 放电率。
- `prediction_error(predicted, actual) -> float`：MAE；`predicted` 为 None 时返回 `None`。
- `convergence(errors: list[float]) -> dict`：返回 `slope`、`head_mean`、`tail_mean`、`converged: bool`。

### 3.4 实验编排（新，`experiment.py`）

- `run_experiment(network, provider, *, mode, steps, observe_every, n_bins, seed) -> ExperimentResult`
  - 在 `Observer` 之上编排：每个观察点取 LLM 的 `predicted_bin_activity`，到下一观察点算实测误差。
  - 返回 `errors` 序列、`self_descriptions` 序列、`convergence` 结果。
- 实验组 / 对照组用同一 `seed` 构造网络，仅 `mode` 不同。

## 4. 技术选型

| 关注面 | 选型 | 理由 |
| --- | --- | --- |
| LLM | 当前网关 `qwen3.8-flash`（env: `OPENAI_BASE_URL/KEY/MODEL`） | 已验证可调通 |
| SDK | `openai`（OpenAI 兼容） | DashScope 兼容端点，已验证 |
| 数值 | NumPy | 与 M1 一致 |
| 持久化 | 实验结果 JSON 落 D 盘 `data/` | 遵循「数据在 D 盘」约定 |

## 5. 里程碑

1. **M2a**：`GatewayLLMProvider` + `metrics` + `experiment`，跑通实验组/对照组，产出误差序列。
2. **M2b**：真实网关跑完整实验，判定「是否收敛」，给出可证伪结论。

## 6. 非目标

- 不引入 embedding / 文本相似度做「自我描述稳定性」的硬判定（判据 1 仅辅助记录）。
- 不做规模放大（M3）。
- 不做 LLM「自主决定何时观察」（仍用固定节律，保证可复现）。
