from __future__ import annotations

import numpy as np

from .sparse import SparseMatrix


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
        inhibitory_fraction=0.2,
        background_rate=0.0,
        background_strength=0.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_units = int(n_units)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.leak = float(leak)
        self.dt = float(dt)
        self.hebb_lr = float(hebb_lr)
        self.weight_decay = float(weight_decay)
        self.weight_max = float(weight_max)
        self.inhibitory_fraction = float(inhibitory_fraction)
        self.background_rate = float(background_rate)
        self.background_strength = float(background_strength)

        # Dale 法则：每个单元要么纯兴奋(+)、要么纯抑制(-)。
        n_inhib = int(round(self.n_units * self.inhibitory_fraction))
        self.is_inhibitory = np.zeros(self.n_units, dtype=bool)
        if n_inhib > 0:
            self.is_inhibitory[:n_inhib] = True
            rng.shuffle(self.is_inhibitory)
        # 连接 j→i 的符号取决于 presynaptic 单元 j 是否抑制。
        self._presynaptic_sign = np.where(self.is_inhibitory, -1.0, 1.0)

        # 稀疏突触：仅存非零连接（pre→post）。W[i,j] 表示 j→i（post 行, pre 列）。
        self.W = SparseMatrix(self.n_units, self.n_units)
        if connectivity > 0:
            # 先按边采样：每个 (post, pre) 以 connectivity 概率存在。
            n_possible = self.n_units * self.n_units
            # 用确定性的采样：先抽边数，再抽位置，避免生成 N×N 稠密掩码。
            n_edges = rng.binomial(n_possible, connectivity)
            flat_idx = rng.choice(n_possible, size=n_edges, replace=False)
            post = flat_idx // self.n_units
            pre = flat_idx % self.n_units
            # 去掉自连接
            keep = post != pre
            post = post[keep]
            pre = pre[keep]
            magnitude = np.abs(rng.normal(0.0, weight_scale, size=len(post)))
            data = magnitude * self._presynaptic_sign[pre]
            self.W.set_coo(post, pre, data)

        self.V = rng.uniform(0.0, self.threshold, size=self.n_units)
        self.refrac = np.zeros(self.n_units, dtype=int)
        self.spikes = np.zeros(self.n_units, dtype=bool)

    def step(self, external_input=None) -> np.ndarray:
        ext = (
            np.zeros(self.n_units)
            if external_input is None
            else np.asarray(external_input, dtype=float)
        )
        was_refractory = self.refrac > 0
        synaptic = self.W.dot(self.spikes.astype(float))
        background = self._sample_background()
        integrating = ~was_refractory
        self.V[integrating] = (
            self.V[integrating]
            + (synaptic[integrating] + ext[integrating] + background[integrating]
               - self.leak * self.V[integrating]) * self.dt
        )
        self.V[was_refractory] = 0.0
        fired = (self.V >= self.threshold) & integrating
        self.spikes = fired
        self.V[fired] = 0.0
        self.refrac[fired] = self.refractory
        not_fired = ~fired
        self.refrac[not_fired] = np.maximum(self.refrac[not_fired] - 1, 0)
        self._hebb_update(fired)
        return fired.copy()

    def _sample_background(self):
        if self.background_rate <= 0.0:
            return np.zeros(self.n_units)
        spikes = self.rng.random(self.n_units) < self.background_rate
        return spikes.astype(float) * self.background_strength

    def _hebb_update(self, fired):
        """稀疏 Hebb：仅更新「已存在」的边，且只对同时发放的边增强。

        符号跟随每条边自己的 data（可正可负），Hebb 只调幅值、不吞符号。
        """
        rows, cols, data = self.W.coo_arrays()
        if len(data) == 0:
            return
        # 边 post←pre 的 post 和 pre 是否同时发放
        both = fired[rows] & fired[cols]
        # 分离符号与幅值
        sign = np.sign(data)
        mag = np.abs(data)
        # 对所有边做衰减
        mag = mag - self.weight_decay * mag
        # 对同时发放的边做增强：mag += hebb_lr * (weight_max - mag)
        mag[both] = mag[both] + self.hebb_lr * (self.weight_max - mag[both])
        mag = np.clip(mag, 0.0, self.weight_max)
        new_data = mag * sign
        self.W.set_coo(rows, cols, new_data)
