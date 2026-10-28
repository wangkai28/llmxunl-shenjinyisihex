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
        delay_max=1,
        inh_delay_max=None,
        threshold=1.0,
        refractory=2,
        leak=0.1,
        dt=1.0,
        hebb_lr=0.0,
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
        self.hebb_lr = float(hebb_lr)
        self.j_inh = float(j_inh)
        self.j_exc = float(j_exc)
        self.external_mean = float(external_mean)
        self.external_noise = float(external_noise)
        self.delay_max = int(delay_max)
        self.inh_delay_max = (
            int(inh_delay_max) if inh_delay_max is not None else self.delay_max
        )

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
        post = post[keep]
        pre = pre[keep]
        data = data[keep]
        # 给每条边分配固定的突触延迟。
        # 抑制突触延迟更短（快速打断兴奋雪崩），兴奋突触延迟均匀 1~delay_max。
        is_inh_edge = self.is_inhibitory[pre]
        edge_delay = np.ones(len(post), dtype=np.int64)
        max_delay = max(self.delay_max, self.inh_delay_max)
        if self.delay_max > 1:
            edge_delay[~is_inh_edge] = rng.integers(
                1, self.delay_max + 1, size=(~is_inh_edge).sum()
            )
        if self.inh_delay_max > 1:
            edge_delay[is_inh_edge] = rng.integers(
                1, self.inh_delay_max + 1, size=is_inh_edge.sum()
            )
        self._edge_post = post
        self._edge_pre = pre
        self._edge_w = data
        self._edge_delay = edge_delay
        # 标记「初始平衡抑制边」（由 j_inh 控制），区别于后来追加的交叉抑制边。
        self._base_inh_edge_mask = is_inh_edge
        # 可塑性掩码：默认全部不可塑（平衡网络），由 SelfModel 标记核心区可塑。
        self._plastic = np.zeros(len(post), dtype=bool)
        self.W.set_coo(post, pre, data)
        # 延迟缓冲：delay_buf[t] 存「t 步后要送达各单元的突触输入」
        self._delay_buf = np.zeros((max_delay, self.n_units), dtype=float)

        self.V = rng.uniform(0.0, self.threshold, size=self.n_units)
        self.refrac = np.zeros(self.n_units, dtype=int)
        self.spikes = np.zeros(self.n_units, dtype=bool)
        self.lesioned = np.zeros(self.n_units, dtype=bool)

    def step(self, external_input=None) -> np.ndarray:
        ext = (
            np.zeros(self.n_units)
            if external_input is None
            else np.asarray(external_input, dtype=float)
        )
        was_refractory = self.refrac > 0
        # 1. 本步送达的突触输入 = delay_buf[0]
        synaptic = self._delay_buf[0].copy()
        # 2. 左移缓冲
        buf_len = self._delay_buf.shape[0]
        if buf_len > 1:
            self._delay_buf[:-1] = self._delay_buf[1:]
            self._delay_buf[-1] = 0.0
        else:
            self._delay_buf[0] = 0.0
        # 3. 当前发放的 spike 按各自延迟写入缓冲
        if self.spikes.any():
            active = self.spikes[self._edge_pre]
            if active.any():
                posts = self._edge_post[active]
                weights = self._edge_w[active]
                delays = self._edge_delay[active]
                # 对每个延迟层级分别累加
                for d in range(1, buf_len + 1):
                    mask = delays == d
                    if mask.any():
                        np.add.at(
                            self._delay_buf[d - 1],
                            posts[mask],
                            weights[mask],
                        )
        background = self._sample_background()
        integrating = ~was_refractory
        self.V[integrating] = (
            self.V[integrating]
            + (synaptic[integrating] + ext[integrating] + background[integrating]
               - self.leak * self.V[integrating]) * self.dt
        )
        self.V[was_refractory] = 0.0
        fired = (self.V >= self.threshold) & integrating & ~self.lesioned
        self.spikes = fired
        self.V[fired] = 0.0
        self.refrac[fired] = self.refractory
        not_fired = ~fired
        self.refrac[not_fired] = np.maximum(self.refrac[not_fired] - 1, 0)
        self._hebb_update(fired)
        return fired.copy()

    def _sample_background(self):
        # 大量弱外部输入的净效应：每个单元每步收到一个
        # 均值 external_mean、标准差 external_noise 的高斯涨落。
        return self.rng.normal(self.external_mean, self.external_noise, size=self.n_units)

    def _hebb_update(self, fired):
        """局部 Hebb：只强化「可塑」的、且两端同时发放的边。"""
        if self.hebb_lr <= 0.0 or not self._plastic.any():
            return
        post = self._edge_post
        pre = self._edge_pre
        w = self._edge_w
        # 可塑且两端同时发放的边
        both = self._plastic & fired[post] & fired[pre]
        if not both.any():
            return
        # 对可塑边做增强：w += hebb_lr * (w_max - w)，其中 w_max 取当前最大幅值的若干倍
        w_max = np.abs(self._edge_w).max() * 5.0
        # 只增强兴奋的可塑边（抑制边不可塑，保持平衡）
        exc_plastic = both & (self._edge_w > 0)
        if exc_plastic.any():
            self._edge_w[exc_plastic] = np.clip(
                self._edge_w[exc_plastic] + self.hebb_lr * (w_max - self._edge_w[exc_plastic]),
                0.0, w_max,
            )
            # 同步更新 W（稀疏矩阵）以保持一致
            self.W.set_coo(self._edge_post, self._edge_pre, self._edge_w)

    def mark_plastic_edges(self, mask):
        """标记哪些边可塑（由 SelfModel 调用，标记核心区内部的边）。"""
        self._plastic = np.asarray(mask, dtype=bool)

    def prune_edges(self, fraction: float):
        """随机剪除 fraction 比例的边（模拟遗忘/损伤）。"""
        if fraction <= 0.0 or self._edge_w.size == 0:
            return
        n_keep = int(self._edge_w.size * (1.0 - fraction))
        keep_idx = self.rng.choice(self._edge_w.size, size=n_keep, replace=False)
        keep_idx.sort()
        self._edge_post = self._edge_post[keep_idx]
        self._edge_pre = self._edge_pre[keep_idx]
        self._edge_w = self._edge_w[keep_idx]
        self._edge_delay = self._edge_delay[keep_idx]
        self._plastic = self._plastic[keep_idx]
        self._base_inh_edge_mask = self._base_inh_edge_mask[keep_idx]
        self.W.set_coo(self._edge_post, self._edge_pre, self._edge_w)

    def add_inhibitory_edges(self, post, pre, strength: float):
        """添加一批抑制性突触（post←pre，权重为负）。"""
        post = np.asarray(post, dtype=np.int64)
        pre = np.asarray(pre, dtype=np.int64)
        n = len(post)
        if n == 0:
            return
        w = np.full(n, -abs(float(strength)), dtype=float)
        delay = np.ones(n, dtype=np.int64)
        plastic = np.zeros(n, dtype=bool)
        self._edge_post = np.concatenate([self._edge_post, post])
        self._edge_pre = np.concatenate([self._edge_pre, pre])
        self._edge_w = np.concatenate([self._edge_w, w])
        self._edge_delay = np.concatenate([self._edge_delay, delay])
        self._plastic = np.concatenate([self._plastic, plastic])
        # 追加的交叉抑制边不属于「初始平衡抑制边」，不受 set_j_inh 重缩放。
        self._base_inh_edge_mask = np.concatenate([self._base_inh_edge_mask, np.zeros(n, dtype=bool)])
        self.W.set_coo(self._edge_post, self._edge_pre, self._edge_w)

    def set_j_inh(self, new_j_inh: float):
        """动态调整抑制强度：按比例重缩放「初始平衡抑制边」，不丢状态。

        兴奋边与交叉抑制边不受影响。用于人格演化中 j_inh 的即时生效。
        """
        old_j_inh = self.j_inh
        if old_j_inh == 0.0:
            return
        ratio = float(new_j_inh) / old_j_inh
        mask = self._base_inh_edge_mask
        if mask.any():
            self._edge_w[mask] = self._edge_w[mask] * ratio
            self.W.set_coo(self._edge_post, self._edge_pre, self._edge_w)
        self.j_inh = float(new_j_inh)
