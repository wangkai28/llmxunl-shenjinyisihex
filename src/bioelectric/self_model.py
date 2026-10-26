from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork


class SelfModel:
    """在网络中塑造一个「自我模式」：核心区一群单元形成自我维持的活动。

    通过持续给核心区强输入（form），让 Hebb 学习固化核心区的内部连接，
    使其在外部输入撤掉后仍能维持高于外围的活动（吸引子）。
    """

    def __init__(
        self,
        network: BalancedNetwork,
        *,
        core_size: int,
        seed: int = 0,
        hebb_lr: float = 0.0,
    ):
        self.network = network
        self.n_units = network.n_units
        self.core_size = int(core_size)
        rng = np.random.default_rng(seed)
        # 核心区 = 随机选 core_size 个单元
        self.core_idx = rng.choice(self.n_units, size=self.core_size, replace=False)
        self.core_mask = np.zeros(self.n_units, dtype=bool)
        self.core_mask[self.core_idx] = True
        self.form_strength = 2.0
        # 塑造后核心区内部连接的平均权重基线（用于算相对连接强度）
        self._baseline_core_strength = None
        if hebb_lr > 0.0:
            network.hebb_lr = hebb_lr
            self._mark_core_plastic()

    def _mark_core_plastic(self):
        """标记核心区内部的兴奋边为可塑（core→core 的边）。"""
        post = self.network._edge_post
        pre = self.network._edge_pre
        core_mask = self.core_mask
        # 可塑 = post 和 pre 都在核心区，且是兴奋边
        plastic = core_mask[post] & core_mask[pre] & (self.network._edge_w > 0)
        self.network.mark_plastic_edges(plastic)

    def _core_input(self, strength=None):
        inp = np.zeros(self.n_units, dtype=float)
        inp[self.core_mask] = self.form_strength if strength is None else strength
        return inp

    def offset_input(self, offset: float, strength=None):
        """把核心输入偏移 offset（0~1 比例），注入到偏离核心区的位置。"""
        inp = np.zeros(self.n_units, dtype=float)
        s = self.form_strength if strength is None else strength
        # 核心区索引 + 偏移量（环上移动）
        shift = int(round(offset * self.n_units))
        shifted = (self.core_idx + shift) % self.n_units
        inp[shifted] = s
        return inp

    def form(self, steps: int):
        """塑造阶段：给核心区持续强输入，跑 steps 步让 Hebb 固化核心连接。"""
        for _ in range(steps):
            self.network.step(external_input=self._core_input())
        # 记录塑造后的核心连接强度基线
        post = self.network._edge_post
        pre = self.network._edge_pre
        core_edges = self.core_mask[post] & self.core_mask[pre]
        if core_edges.any():
            self._baseline_core_strength = float(
                np.abs(self.network._edge_w[core_edges]).mean()
            )
        else:
            self._baseline_core_strength = 1e-9

    def measure_holdout(self, steps: int):
        """测试阶段：撤掉核心输入，测核心区与外围区的平均活动率。"""
        core_spikes = 0
        peri_spikes = 0
        for _ in range(steps):
            self.network.step(external_input=None)
            core_spikes += int(self.network.spikes[self.core_mask].sum())
            peri_spikes += int(self.network.spikes[~self.core_mask].sum())
        core_rate = core_spikes / (steps * self.core_size)
        peri_rate = peri_spikes / (steps * (self.n_units - self.core_size))
        return core_rate, peri_rate


def extract_self_dimensions(model: SelfModel, steps: int) -> dict:
    """从自我模式提取一组可标注的认知维度。"""
    net = model.network
    core_mask = model.core_mask

    core_spikes = []
    peri_spikes = []
    per_step_total = []
    for _ in range(steps):
        net.step(external_input=None)
        spikes = net.spikes
        core_spikes.append(int(spikes[core_mask].sum()))
        peri_spikes.append(int(spikes[~core_mask].sum()))
        per_step_total.append(int(spikes.sum()))

    core_activity_rate = float(np.mean(core_spikes)) / model.core_size
    periphery_activity_rate = float(np.mean(peri_spikes)) / (net.n_units - model.core_size)
    # 核心聚焦度：核心区活动占总活动的比例（相对核心区占比）
    core_share = float(np.mean(core_spikes)) / (float(np.mean(per_step_total)) + 1e-9)
    expected_share = model.core_size / net.n_units
    core_focus = core_share / (expected_share + 1e-9)
    # 同步指数：每步总发放数的 Fano 因子（越低越异步、越高越同步）
    total = np.array(per_step_total, dtype=float)
    sync_index = float(total.var() / total.mean()) if total.mean() > 0 else float("nan")

    # 核心连接强度：核心区内部突触的平均幅值（相对塑造基线的保留率，0~1+）
    post = net._edge_post
    pre = net._edge_pre
    core_edges = model.core_mask[post] & model.core_mask[pre]
    if core_edges.any() and model._baseline_core_strength:
        current = float(np.abs(net._edge_w[core_edges]).mean())
        core_strength = current / model._baseline_core_strength
    else:
        core_strength = 0.0

    return {
        "core_activity_rate": core_activity_rate,
        "periphery_activity_rate": periphery_activity_rate,
        "core_focus": core_focus,
        "sync_index": sync_index,
        "core_strength": core_strength,
    }
