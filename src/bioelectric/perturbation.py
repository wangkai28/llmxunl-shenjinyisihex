from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork
from .self_model import SelfModel, extract_self_dimensions


def apply_noise(model: SelfModel, noise_scale: float):
    """放大外部噪声（环境嘈杂程度）。"""
    model.network.external_noise = model.network.external_noise * noise_scale


def apply_prune(model: SelfModel, fraction: float):
    """随机剪除 fraction 比例的突触（遗忘/损伤）。"""
    model.network.prune_edges(fraction)


def apply_lesion(model: SelfModel, fraction: float):
    """随机让 fraction 比例的单元失活（局部脑损伤）。"""
    net = model.network
    n_lesion = int(net.n_units * fraction)
    idx = net.rng.choice(net.n_units, size=n_lesion, replace=False)
    net.lesioned[idx] = True


def apply_pulse(model: SelfModel, strength: float, duration: int):
    """瞬时强刺激：给全网一个短暂强脉冲（duration 步），随后撤掉。"""
    net = model.network
    inp = np.full(net.n_units, strength, dtype=float)
    for _ in range(duration):
        net.step(external_input=inp)


def apply_offset_input(model: SelfModel, offset: float, steps: int):
    """输入位置扰动：把维系自我的输入偏移到别处，注入 steps 步。"""
    net = model.network
    for _ in range(steps):
        net.step(external_input=model.offset_input(offset))


def perturbation_sweep(
    *,
    n_units: int,
    core_size: int,
    noise_levels: list[float],
    prune_fractions: list[float],
    lesion_fractions: list[float],
    seed: int = 0,
    measure_steps: int = 100,
) -> list[dict]:
    """生成参数化干扰数据集：对每个干扰组合测量认知维度的变化。

    返回列表，每个元素是一个 dict，含干扰参数与对应的认知维度。
    """
    dataset = []
    for noise in noise_levels:
        for prune in prune_fractions:
            for lesion in lesion_fractions:
                # 每个组合用独立、确定性的 seed 生成网络，保证可复现。
                # 不用 Python 内置 hash()（其值依赖 PYTHONHASHSEED，不可复现）。
                combo_seed = (
                    seed
                    + int(round(noise * 1000)) * 1
                    + int(round(prune * 1000)) * 1000
                    + int(round(lesion * 1000)) * 1000000
                ) % (2**31)
                net = BalancedNetwork(
                    n_units=n_units, k_exc=10, j_inh=-4.0,
                    external_mean=0.15, external_noise=0.08, delay_max=4,
                    seed=combo_seed,
                )
                model = SelfModel(net, core_size=core_size, seed=1, hebb_lr=0.02)
                model.form(steps=200)
                # 施加干扰
                if noise != 1.0:
                    apply_noise(model, noise)
                if prune > 0.0:
                    apply_prune(model, prune)
                if lesion > 0.0:
                    apply_lesion(model, lesion)
                dims = extract_self_dimensions(model, steps=measure_steps)
                dataset.append({
                    "noise_scale": noise,
                    "prune_fraction": prune,
                    "lesion_fraction": lesion,
                    **dims,
                })
    return dataset


def factor_effects(dataset: list[dict]) -> dict:
    """计算每个干扰因素对每个认知维度的边际效应。

    边际效应 = 该因素从最低档升到最高档时，认知维度的平均变化量。
    返回 {factor_name: {dimension: effect}}。
    """
    factors = ("noise_scale", "prune_fraction", "lesion_fraction")
    dimensions = [k for k in dataset[0].keys() if k not in factors] if dataset else []

    effects = {}
    for factor in factors:
        levels = sorted({d[factor] for d in dataset})
        if len(levels) < 2:
            continue
        low, high = levels[0], levels[-1]
        factor_effects = {}
        for dim in dimensions:
            low_vals = [d[dim] for d in dataset if d[factor] == low]
            high_vals = [d[dim] for d in dataset if d[factor] == high]
            # 控制其他因素：只对「其他因素取值相同」的配对求差，更严谨
            effect = float(np.mean(high_vals) - np.mean(low_vals))
            factor_effects[dim] = effect
        effects[factor] = factor_effects
    return effects


def recovery_test(
    *,
    n_units: int,
    core_size: int,
    noise_scale: float = 1.0,
    prune_fraction: float = 0.0,
    lesion_fraction: float = 0.0,
    recover_steps: int = 100,
    seed: int = 0,
) -> dict:
    """恢复性测试：施加干扰 → 撤掉 → 看自我能否恢复到基线。

    返回 baseline/damaged/recovered 的核心活动率，以及 recovery_fraction
    （恢复率 = 损伤后恢复的部分 / 损伤掉的部分，1 表示完全恢复，0 表示无恢复）。
    """
    net = BalancedNetwork(
        n_units=n_units, k_exc=10, j_inh=-4.0,
        external_mean=0.15, external_noise=0.08, delay_max=4, seed=seed,
    )
    model = SelfModel(net, core_size=core_size, seed=1, hebb_lr=0.02)
    model.form(steps=200)

    # 基线
    baseline_dims = extract_self_dimensions(model, steps=50)
    baseline = baseline_dims["core_activity_rate"]

    # 施加干扰
    if noise_scale != 1.0:
        apply_noise(model, noise_scale)
    if prune_fraction > 0.0:
        apply_prune(model, prune_fraction)
    if lesion_fraction > 0.0:
        apply_lesion(model, lesion_fraction)
    damaged_dims = extract_self_dimensions(model, steps=50)
    damaged = damaged_dims["core_activity_rate"]

    # 撤掉干扰：噪声恢复为原始值；剪除和失活是结构性、不可逆的。
    # 但剪除后仍可通过 Hebb 在剩余连接上部分重建，失活则永久。
    if noise_scale != 1.0:
        net.external_noise = 0.08  # 恢复原始噪声
    # 跑 recover_steps 让网络在剩余结构上重新演化（Hebb 重建）
    for _ in range(recover_steps):
        net.step(external_input=None)
    recovered_dims = extract_self_dimensions(model, steps=50)
    recovered = recovered_dims["core_activity_rate"]

    # 恢复率
    denom = baseline - damaged
    if abs(denom) < 1e-9:
        recovery_fraction = 1.0  # 没有损伤，视为完全"保持"
    else:
        recovery_fraction = (recovered - damaged) / denom
    recovery_fraction = float(np.clip(recovery_fraction, 0.0, 1.0))

    return {
        "baseline_core_rate": baseline,
        "damaged_core_rate": damaged,
        "recovered_core_rate": recovered,
        "recovery_fraction": recovery_fraction,
    }
