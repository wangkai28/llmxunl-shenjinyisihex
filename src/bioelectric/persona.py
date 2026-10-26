from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork
from .self_model import SelfModel, extract_self_dimensions


# 三个自我的人格预设（性格 → 网络参数映射）
PERSONAS = {
    # 本我：冲动、欲望驱动、兴奋强、抑制弱、活动高、不稳定
    "id": {
        "j_inh": -3.0,           # 抑制弱（克制弱）
        "external_mean": 0.25,   # 基础驱动偏高（欲望强）
        "external_noise": 0.20,  # 噪声大（不稳定、易冲动）
    },
    # 超我：克制、理想化、抑制强、活动低、稳定
    "superego": {
        "j_inh": -6.5,           # 抑制强（克制强）
        "external_mean": 0.10,   # 基础驱动低（不冲动）
        "external_noise": 0.05,  # 噪声小（稳定、自律）
    },
    # 道家：清静、顺应、柔和、协调、稳定
    "tao": {
        "j_inh": -4.5,           # 抑制适中（不强不弱，顺应）
        "external_mean": 0.15,   # 基础驱动适中
        "external_noise": 0.06,  # 噪声小（清静、稳定）
    },
}


def build_persona_network(persona: str, n_units: int, seed: int = 0) -> BalancedNetwork:
    """按人格预设构造一个平衡网络。"""
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected one of {list(PERSONAS)}")
    cfg = PERSONAS[persona]
    return BalancedNetwork(
        n_units=n_units,
        k_exc=10,
        j_exc=1.0,
        j_inh=cfg["j_inh"],
        external_mean=cfg["external_mean"],
        external_noise=cfg["external_noise"],
        delay_max=4,
        seed=seed,
    )


def measure_persona_dimensions(
    network: BalancedNetwork,
    *,
    core_size: int,
    seed: int = 0,
) -> dict:
    """度量人格网络的可标注性格维度。

    在网络上塑造一个核心自我，然后提取性格特征：
    - impulsivity（冲动性）：核心活动率（越高越冲动）
    - restraint（克制性）：核心聚焦度（自我边界清晰度，克制带来清晰边界）
    - adaptability（顺应性）：核心-外围活动率的协调度
    - stability（稳定性）：同步指数的倒数（越异步越稳定）
    """
    model = SelfModel(network, core_size=core_size, seed=seed, hebb_lr=0.02)
    model.form(steps=200)
    dims = extract_self_dimensions(model, steps=100)

    # 稳定性：活动率的时间波动（绝对标准差，越低越稳定）。
    # 不用变异系数 CV——CV 会因低活动率（分母小）而虚高，把「安静」误读成「不稳定」。
    rates = []
    for _ in range(200):
        network.step(external_input=None)
        rates.append(network.spikes.mean())
    rates = np.array(rates, dtype=float)
    std = float(rates.std())
    # 稳定性 = 1 / (1 + std)，std 越小越稳定
    stability = 1.0 / (1.0 + std)

    # 抑制控制：核心区内部抑制突触的平均幅值（越高越自律）
    post = network._edge_post
    pre = network._edge_pre
    core_edges = model.core_mask[post] & model.core_mask[pre]
    inh_edges = core_edges & (network._edge_w < 0)
    inhibitory_control = (
        float(np.abs(network._edge_w[inh_edges]).mean())
        if inh_edges.any() else 0.0
    )

    # 恢复力：施加瞬时全局脉冲后，核心活动率恢复到脉冲前的程度
    before_rates = []
    for _ in range(20):
        network.step(external_input=None)
        before_rates.append(network.spikes[model.core_mask].mean())
    before = float(np.mean(before_rates))
    # 施加瞬时强脉冲
    pulse = np.full(network.n_units, 3.0, dtype=float)
    for _ in range(5):
        network.step(external_input=pulse)
    # 恢复期
    recovered_rates = []
    for _ in range(30):
        network.step(external_input=None)
        recovered_rates.append(network.spikes[model.core_mask].mean())
    after = float(np.mean(recovered_rates))
    # 恢复力 = 1 - 恢复后偏离基线的程度
    resilience = 1.0 - abs(after - before) / (before + 1e-9)
    resilience = float(np.clip(resilience, 0.0, 1.0))

    # 冲动性 = 核心活动率
    impulsivity = dims["core_activity_rate"]
    # 克制性 = 核心聚焦度（清晰自我边界）
    restraint = dims["core_focus"]
    # 顺应性 = 核心与外围活动的协调度（两者越接近越协调）
    adaptability = 1.0 - abs(
        dims["core_activity_rate"] - dims["periphery_activity_rate"]
    )
    return {
        "impulsivity": impulsivity,
        "restraint": restraint,
        "adaptability": adaptability,
        "stability": stability,
        "resilience": resilience,
        "inhibitory_control": inhibitory_control,
    }
