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
        # is_inhibitory[i] 表示单元 i 作为 presynaptic 时，其输出连接为负。
        n_inhib = int(round(self.n_units * self.inhibitory_fraction))
        self.is_inhibitory = np.zeros(self.n_units, dtype=bool)
        if n_inhib > 0:
            self.is_inhibitory[:n_inhib] = True
            rng.shuffle(self.is_inhibitory)
        # 符号矩阵：sign[i, j] = 连接 j→i 的符号，取决于 presynaptic 单元 j 的类型。
        # 抑制单元 j：所有输出连接为负；兴奋单元 j：所有输出连接为正。
        self._sign = np.where(self.is_inhibitory[None, :], -1.0, 1.0)
        self._sign = np.tile(self._sign, (self.n_units, 1))
        self.W = np.zeros((self.n_units, self.n_units), dtype=float)
        if connectivity > 0:
            mask = rng.random((self.n_units, self.n_units)) < connectivity
            magnitude = np.abs(rng.normal(0.0, weight_scale, size=int(mask.sum())))
            self.W[mask] = magnitude * self._sign[mask]
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
        was_refractory = self.refrac > 0
        synaptic = self.W @ self.spikes.astype(float)
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
        # 泊松背景发放：每个单元以 background_rate 的概率收到一个
        # background_strength 的兴奋脉冲。用 self.rng 保证同 seed 可复现。
        spikes = self.rng.random(self.n_units) < self.background_rate
        return spikes.astype(float) * self.background_strength

    def _hebb_update(self, fired):
        co = np.outer(fired.astype(float), fired.astype(float))
        np.fill_diagonal(co, 0.0)
        # 只调 magnitude，不吞符号：把 W 拆成 sign * |W|，对 |W| 做增强/衰减/上限。
        magnitude = np.abs(self.W)
        magnitude = magnitude + self.hebb_lr * co * (self.weight_max - magnitude)
        magnitude = magnitude - self.weight_decay * magnitude
        magnitude = np.clip(magnitude, 0.0, self.weight_max)
        self.W = magnitude * self._sign
        np.fill_diagonal(self.W, 0.0)
