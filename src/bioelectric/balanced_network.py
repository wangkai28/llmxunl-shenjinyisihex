from __future__ import annotations

import numpy as np

from .sparse import SparseMatrix


class BalancedNetwork:
    """Brunel 式平衡网络：突触强度按 1/√K 缩放，活动由涨落驱动。

    关键点（区别于普通 SpikingNetwork）：
    - 每个神经元接收固定 K 个兴奋 + K/4 个抑制突触（连接数固定，不随 N 变）。
    - 突触强度 J = j / sqrt(K)，使涨落幅度与网络规模无关。
    - 外部泊松输入提供涨落，兴奋/抑制均值平衡让膜电位悬在阈值附近。
    """

    def __init__(
        self,
        n_units,
        *,
        k_exc=100,
        j_exc=1.0,
        j_inh=-4.0,
        external_mean=0.3,
        external_noise=0.15,
        threshold=1.0,
        refractory=2,
        leak=0.1,
        dt=1.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_units = int(n_units)
        self.k_exc = int(k_exc)
        self.k_inh = int(k_exc // 4)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.leak = float(leak)
        self.dt = float(dt)
        self.external_mean = float(external_mean)
        self.external_noise = float(external_noise)

        # 抑制单元占 20%
        n_inhib = int(round(self.n_units * 0.2))
        self.is_inhibitory = np.zeros(self.n_units, dtype=bool)
        if n_inhib > 0:
            self.is_inhibitory[:n_inhib] = True
            rng.shuffle(self.is_inhibitory)

        # 突触强度按 1/√K 缩放
        scale = 1.0 / np.sqrt(self.k_exc)
        j_e = j_exc * scale
        j_i = j_inh * scale

        # 构建稀疏突触：每个 post 单元接收 k_exc 个兴奋 + k_inh 个抑制
        self.W = SparseMatrix(self.n_units, self.n_units)
        exc_src = np.flatnonzero(~self.is_inhibitory)
        inh_src = np.flatnonzero(self.is_inhibitory)

        post_parts = []
        pre_parts = []
        data_parts = []
        for i in range(self.n_units):
            # 兴奋输入：从兴奋单元中随机选 k_exc 个
            e_pre = rng.choice(exc_src, size=self.k_exc, replace=True)
            post_parts.append(np.full(self.k_exc, i, dtype=np.int64))
            pre_parts.append(e_pre)
            data_parts.append(np.full(self.k_exc, j_e, dtype=float))
            # 抑制输入：从抑制单元中随机选 k_inh 个
            i_pre = rng.choice(inh_src, size=self.k_inh, replace=True)
            post_parts.append(np.full(self.k_inh, i, dtype=np.int64))
            pre_parts.append(i_pre)
            data_parts.append(np.full(self.k_inh, j_i, dtype=float))

        post = np.concatenate(post_parts)
        pre = np.concatenate(pre_parts)
        data = np.concatenate(data_parts)
        # 去掉自连接（若有）
        keep = post != pre
        self.W.set_coo(post[keep], pre[keep], data[keep])

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
        return fired.copy()

    def _sample_background(self):
        # 大量弱外部输入的净效应：每个单元每步收到一个
        # 均值 external_mean、标准差 external_noise 的高斯涨落。
        return self.rng.normal(self.external_mean, self.external_noise, size=self.n_units)
