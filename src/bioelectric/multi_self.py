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
        cross_inhibition: float = 0.0,
    ):
        self.network = network
        self.n_units = network.n_units
        self.core_specs = core_specs
        rng = np.random.default_rng(seed)

        # 给每个自我分配不重叠的核心单元
        self.core_masks = {}
        self.core_sizes = {}
        self.biases = {}
        self.context_bias = {}
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
            self.context_bias[name] = float(spec["bias"])
            used.update(chosen.tolist())

        if hebb_lr > 0.0:
            network.hebb_lr = hebb_lr
            self._mark_all_core_plastic()
        if cross_inhibition > 0.0:
            self._add_cross_inhibition(cross_inhibition)

    def _mark_all_core_plastic(self):
        """标记所有自我核心区内部的兴奋边为可塑。"""
        post = self.network._edge_post
        pre = self.network._edge_pre
        plastic = np.zeros(len(post), dtype=bool)
        for mask in self.core_masks.values():
            plastic |= mask[post] & mask[pre] & (self.network._edge_w > 0)
        self.network.mark_plastic_edges(plastic)

    def _add_cross_inhibition(self, strength: float):
        """在各自我核心区之间添加交叉抑制，形成互相制衡。

        每个核心区的单元向其他核心区的单元发送抑制边。
        为避免全连接，每个 pre 单元随机连到其他核心区的约 10% 单元。
        """
        net = self.network
        names = list(self.core_masks.keys())
        post_parts = []
        pre_parts = []
        for src_name in names:
            src_idx = np.flatnonzero(self.core_masks[src_name])
            for dst_name in names:
                if dst_name == src_name:
                    continue
                dst_idx = np.flatnonzero(self.core_masks[dst_name])
                # 每个 src 单元连到约 20% 的 dst 单元
                n_targets = max(1, int(len(dst_idx) * 0.2))
                for s in src_idx:
                    targets = net.rng.choice(dst_idx, size=n_targets, replace=False)
                    post_parts.append(targets)
                    pre_parts.append(np.full(n_targets, s, dtype=np.int64))
        if post_parts:
            post = np.concatenate(post_parts)
            pre = np.concatenate(pre_parts)
            net.add_inhibitory_edges(post, pre, strength)

    def form(self, steps: int):
        """同时塑造多个自我：给各核心区注入各自偏置强度的输入。"""
        for _ in range(steps):
            inp = np.zeros(self.n_units, dtype=float)
            for name, mask in self.core_masks.items():
                inp[mask] = self.context_bias[name]
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
        # 若某序列方差为 0（恒定），相关系数无定义，用 NaN 并排除
        stds = series.std(axis=1)
        valid = stds > 1e-12
        if valid.sum() > 1:
            corr = np.corrcoef(series[valid])
            off_diag = corr[~np.eye(int(valid.sum()), dtype=bool)]
            mean_corr = float(off_diag.mean())
        else:
            mean_corr = float("nan")
        # 切换性 = 1 - 平均相关（相关越低，越「此消彼长」）
        switching = 1.0 - (mean_corr + 1.0) / 2.0 if not np.isnan(mean_corr) else 0.0
    else:
        switching = 0.0

    result["dominance"] = float(dominance)
    result["switching"] = float(switching)
    return result


def lesion_one_self(ms: MultiSelf, name: str, fraction: float):
    """损伤某个自我的核心区：让该区域 fraction 比例的单元失活。"""
    if name not in ms.core_masks:
        raise ValueError(f"unknown self {name!r}")
    mask = ms.core_masks[name]
    core_idx = np.flatnonzero(mask)
    n_lesion = int(len(core_idx) * fraction)
    chosen = ms.network.rng.choice(core_idx, size=n_lesion, replace=False)
    ms.network.lesioned[chosen] = True


def set_context(ms: MultiSelf, bias_overrides: dict):
    """设置外部情境：调制各自我的有效偏置（bias），改变谁主导。"""
    for name, bias in bias_overrides.items():
        if name not in ms.core_masks:
            raise ValueError(f"unknown self {name!r}")
        ms.context_bias[name] = float(bias)


def dominant_self(ms: MultiSelf, steps: int) -> str:
    """在当前情境下跑 steps 步，返回活动率最高的自我。"""
    net = ms.network
    names = [n for n in ms.core_masks.keys()]
    activity_sums = {n: 0 for n in names}
    # 找出当前情境的主导自我（bias 最高者），对其余自我施加额外抑制
    dominant = max(names, key=lambda n: ms.context_bias[n])
    for _ in range(steps):
        _drive(ms, dominant)
        for n in names:
            activity_sums[n] += int(net.spikes[ms.core_masks[n]].sum())
    best = max(names, key=lambda n: activity_sums[n])
    return best


def _drive(ms: MultiSelf, dominant: str):
    """按当前情境驱动一步：主导自我正常激活，其余被额外抑制。"""
    net = ms.network
    names = list(ms.core_masks.keys())
    inp = np.zeros(net.n_units, dtype=float)
    for n in names:
        inp[ms.core_masks[n]] = ms.context_bias[n]
        if n != dominant:
            inp[ms.core_masks[n]] -= 3.0
    net.step(external_input=inp)


def transition_trace(
    ms: MultiSelf,
    *,
    from_context: dict,
    to_context: dict,
    settle_steps: int,
    trace_steps: int,
    sample_every: int = 10,
) -> list[dict]:
    """记录从一个情境切换到另一个情境时，各自我的活动率轨迹。

    先稳定在 from_context，再瞬间切到 to_context，逐步采样各自我活动率。
    """
    names = list(ms.core_masks.keys())

    def _dominant(ctx):
        return max(names, key=lambda n: ctx[n])

    # 稳定在 from_context
    set_context(ms, from_context)
    for _ in range(settle_steps):
        _drive(ms, _dominant(from_context))

    # 瞬间切换
    set_context(ms, to_context)
    to_dominant = _dominant(to_context)

    trace = []
    for step in range(1, trace_steps + 1):
        _drive(ms, to_dominant)
        if step % sample_every == 0:
            point = {
                n: float(ms.network.spikes[ms.core_masks[n]].mean())
                for n in names
            }
            trace.append(point)
    return trace
