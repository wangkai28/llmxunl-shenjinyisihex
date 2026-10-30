# 生物电涌现 · AI 身份/人格核心

一个研究项目：以**脉冲神经网络为「身体」**，在其上建立分层的人格结构
（多自我 → 性格参数 → 五大人格 OCEAN），并让**人格理解与演化由神经网络学习
得到**（而非手写规则），最终形成一个可输入信息、输出意识状态、具备自我审视与
反思闭环的「身份核心」。

> 权威架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 它是什么

这不是一个可部署的产品，而是一台**研究「身份 / 人格 / 自我」的仪器** +
一套可复用的技术零件。核心主张：

- 人格不是手写的 if-else，而是**从数据里学出来的**（LLM 蒸馏 + 脉冲网络学习）。
- 人格演化不是规则表，而是**从网络学习痕迹中涌现**的。
- 不追求哲学上的「自我意识」，只追求「连续身份 + 符合人格的响应 + 自指的调节」。

## 分层架构（从快到慢）

```text
交互端口 interact()  输入文本 ⇄ 输出意识状态
  │  9 维统一人格模型（意识2 + 情绪2 + 性格5，LLM 蒸馏）
  ▼
多自我层（本我 / 超我 / 道家，情境切换、共存）
  ▼
性格参数层（从网络学习痕迹读出，涌现式演化）
  ▼
五大人格层 OCEAN（最稳定，极缓慢演化）
  ▼
脉冲神经网络身体（Brunel 平衡网络 + 稳定 Hebb）
```

## 快速上手

环境：Python ≥ 3.10，依赖 `numpy`（核心）+ `openai`（仅 LLM 蒸馏示例需要）。

```bash
pip install numpy openai
python -m pytest          # 运行全部测试（186 个）
```

最小身份核心示例（无需 LLM）：

```python
from bioelectric.identity_core import IdentityCore

core = IdentityCore(persona="tao", state_dir="data/identity")
state = core.interact("我真的很愤怒")   # → 本我主导
state = core.interact("现在很放松")      # → 道家主导
print(core.self_reflect())               # 自我审视文字自述
```

## 目录

- `src/bioelectric/`：源码（28 个模块）。
- `docs/`：架构、设计稿、实现计划、技术报告、SDD 账本。
- `examples/`：可运行的演示脚本。
- `tests/`：186 个测试。
- `data/`：实验数据与固化的人格模型权重。

## 诚实边界

- 人格理解（9 维模型）是 LLM 蒸馏的小型 MLP，字符 n-gram 特征，准确率约 92%，
  非 LLM 原生、非完美语义理解。
- 「自我审视」文本是规则模板生成，不是系统「想」出来的。
- 系统具备「自指结构」（读自己 → 调节自己），但无主观体验，离真正意识很远。

## License

[MIT](LICENSE)
