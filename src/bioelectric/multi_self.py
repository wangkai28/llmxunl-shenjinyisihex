from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork


class MultiSelf:
    """在一个网络上共存多个「自我核心」，各自有性格偏置（bias）。

    bias 是该自我在塑造时的输入强度，反映其活跃倾向（本我高、超我低、道家中）。
    """

    def __init__(
        self,
        network: BalancedNetwork,
        *,
        core_specs: dict,
        seed: int = 0,
        hebb_lr: float = 0.0,
    ):
        self.network = network
        self.n_units = network.n_units
        self.core_specs = core_specs
        rng = np.random.default_rng(seed)

        # 给每个自我分配不重叠的核心单元
        self.core_masks = {}
        self.core_sizes = {}
        self.biases = {}
        used = set()
        for name, spec in core_specs.items():
            size = int(spec["size"])
            available = np.array([i for i in range(self.n_units) if i not in used])
            chosen = rng.choice(available, size=size, replace=False)
            mask = np.zeros(self.n_units, dtype=bool)
            mask[chosen] = True
            self.core_masks[name] = mask
            self.core_sizes[name] = size
            self.biases[name] = float(spec["bias"])
            used.update(chosen.tolist())

        if hebb_lr > 0.0:
            network.hebb_lr = hebb_lr
            self._mark_all_core_plastic()

    def _mark_all_core_plastic(self):
        """标记所有自我核心区内部的兴奋边为可塑。"""
        post = self.network._edge_post
        pre = self.network._edge_pre
        plastic = np.zeros(len(post), dtype=bool)
        for mask in self.core_masks.values():
            plastic |= mask[post] & mask[pre] & (self.network._edge_w > 0)
        self.network.mark_plastic_edges(plastic)

    def form(self, steps: int):
        """同时塑造多个自我：给各核心区注入各自偏置强度的输入。"""
        for _ in range(steps):
            inp = np.zeros(self.n_units, dtype=float)
            for name, mask in self.core_masks.items():
                inp[mask] = self.biases[name]
            self.network.step(external_input=inp)


def measure_multi_self(ms: MultiSelf, steps: int) -> dict:
    """度量多个自我的活动与关系。"""
    net = ms.network
    names = list(ms.core_masks.keys())

    activity_sums = {name: 0 for name in names}
    per_step = {name: [] for name in names}
    for _ in range(steps):
        net.step(external_input=None)
        for name in names:
            act = int(net.spikes[ms.core_masks[name]].sum())
            activity_sums[name] += act
            per_step[name].append(act)

    result = {}
    for name in names:
        rate = activity_sums[name] / (steps * ms.core_sizes[name])
        result[name] = {
            "activity": rate,
            "variance": float(np.var(per_step[name])),
        }

    # 关系维度：主导性 = 最高活动自我的活动率占所有自我活动率之和的比例
    total_activity = sum(result[n]["activity"] for n in names)
    max_activity = max(result[n]["activity"] for n in names)
    dominance = max_activity / (total_activity + 1e-9) if total_activity > 0 else 0.0

    # 切换性：各自我活动率时间序列的相关性（负相关=此消彼长，即切换）
    series = np.array([per_step[n] for n in names], dtype=float)  # (names, steps)
    if series.shape[0] > 1:
        corr = np.corrcoef(series)
        # 取非对角线的平均相关
        off_diag = corr[~np.eye(len(names), dtype=bool)]
        mean_corr = float(off_diag.mean())
        # 切换性 = 1 - 平均相关（相关越低，越「此消彼长」）
        switching = 1.0 - (mean_corr + 1.0) / 2.0
    else:
        switching = 0.0

    result["dominance"] = float(dominance)
    result["switching"] = float(switching)
    return result
