from __future__ import annotations

import numpy as np

from .balanced_network import BalancedNetwork
from .perturbation import apply_lesion, apply_prune, apply_noise
from .self_model import SelfModel, extract_self_dimensions


def persona_under_perturbation(
    network: BalancedNetwork,
    *,
    core_size: int,
    seed: int = 0,
    noise_scale: float = 1.0,
    prune_fraction: float = 0.0,
    lesion_fraction: float = 0.0,
    recover_steps: int = 200,
) -> dict:
    """把一个人格网络放入环境扰动，测量其自我核心的损伤与恢复。"""
    model = SelfModel(network, core_size=core_size, seed=seed, hebb_lr=0.02)
    model.form(steps=200)

    baseline = extract_self_dimensions(model, steps=50)["core_activity_rate"]

    # 施加扰动
    if noise_scale != 1.0:
        apply_noise(model, noise_scale)
    if prune_fraction > 0.0:
        apply_prune(model, prune_fraction)
    if lesion_fraction > 0.0:
        apply_lesion(model, lesion_fraction)
    damaged = extract_self_dimensions(model, steps=50)["core_activity_rate"]

    # 恢复期（噪声可逆，剪除/失活不可逆）
    if noise_scale != 1.0:
        network.external_noise = 0.08
    for _ in range(recover_steps):
        network.step(external_input=None)
    recovered = extract_self_dimensions(model, steps=50)["core_activity_rate"]

    denom = baseline - damaged
    if abs(denom) < 1e-9:
        recovery_fraction = 1.0
    else:
        recovery_fraction = (recovered - damaged) / denom
    recovery_fraction = float(np.clip(recovery_fraction, 0.0, 1.0))

    return {
        "baseline": baseline,
        "damaged": damaged,
        "recovered": recovered,
        "recovery_fraction": recovery_fraction,
    }
