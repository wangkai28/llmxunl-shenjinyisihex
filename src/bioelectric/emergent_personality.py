from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork
from .multi_self import MultiSelf


def _core_exc_strength(net: BalancedNetwork, mask: np.ndarray) -> float:
    """某核心区内部兴奋边的平均强度。"""
    post = net._edge_post
    pre = net._edge_pre
    edges = mask[post] & mask[pre] & (net._edge_w > 0)
    return float(net._edge_w[edges].mean()) if edges.any() else 0.0


def _core_inh_strength(net: BalancedNetwork, mask: np.ndarray) -> float:
    """某核心区内部抑制边（或指向该核心的抑制边）的平均强度。"""
    post = net._edge_post
    pre = net._edge_pre
    edges = mask[post] & (net._edge_w < 0)
    return float(np.abs(net._edge_w[edges]).mean()) if edges.any() else 0.0


def read_personality(net: BalancedNetwork, ms: MultiSelf) -> dict:
    """从网络的学习痕迹读出人格维度（替代手写 evolve_ocean 的产出）。"""
    # 冲动性：本我核心兴奋边强度
    impulsivity = _core_exc_strength(net, ms.core_masks["id"])
    # 克制：超我核心相关的抑制强度（超我抑制越强越自律）
    restraint = _core_inh_strength(net, ms.core_masks["superego"])

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
