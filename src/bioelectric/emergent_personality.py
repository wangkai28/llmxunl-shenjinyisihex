from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork
from .multi_self import MultiSelf
from .codec import emotion_to_context


def _core_exc_strength(net: BalancedNetwork, mask: np.ndarray) -> float:
    """某核心区内部兴奋边的平均强度。"""
    post = net._edge_post
    pre = net._edge_pre
    edges = mask[post] & mask[pre] & (net._edge_w > 0)
    return float(net._edge_w[edges].mean()) if edges.any() else 0.0


def read_personality(net: BalancedNetwork, ms: MultiSelf) -> dict:
    """从网络的学习痕迹读出人格维度（替代手写 evolve_ocean 的产出）。"""
    # 冲动性：本我核心兴奋边强度
    impulsivity = _core_exc_strength(net, ms.core_masks["id"])
    # 克制：超我核心内部兴奋边强度（超我回路越固化越自律）
    restraint = _core_exc_strength(net, ms.core_masks["superego"])

    # 活性与稳定性：核心区活动率及其波动
    rates = []
    for _ in range(100):
        net.step(external_input=None)
        rates.append(float(net.spikes.mean()))
    rates = np.array(rates, dtype=float)
    activity = float(rates.mean())
    stability = float(1.0 / (1.0 + rates.std()))

    return {
        "impulsivity": impulsivity,
        "restraint": restraint,
        "activity": activity,
        "stability": stability,
    }


def emotion_experience(
    net: BalancedNetwork,
    ms: MultiSelf,
    *,
    valence: float,
    arousal: float,
    steps: int = 200,
):
    """用情绪坐标驱动网络学习：情绪 → 主导自我核心 → Hebb 固化。

    形成「情绪经历 → 学习 → 人格变化」的闭环。情绪坐标决定哪个核心
    被激活（复用 emotion_to_context 的映射），激活驱动 Hebb 学习。
    """
    ctx = emotion_to_context(valence, arousal)
    dominant = max(ctx, key=ctx.get)
    # 只激活主导自我核心，让 Hebb 固化该回路
    for _ in range(steps):
        inp = np.zeros(net.n_units, dtype=float)
        inp[ms.core_masks[dominant]] = ctx[dominant]
        net.step(external_input=inp)
