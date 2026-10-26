import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.perturbation import (
    apply_noise,
    apply_prune,
    apply_lesion,
    perturbation_sweep,
)
from bioelectric.self_model import SelfModel, extract_self_dimensions


def _make_model(seed=0):
    net = BalancedNetwork(n_units=300, k_exc=10, j_inh=-4.0,
                          external_mean=0.15, external_noise=0.08, delay_max=4, seed=seed)
    model = SelfModel(net, core_size=40, seed=1, hebb_lr=0.02)
    model.form(steps=200)
    return model


def test_apply_noise_changes_activity():
    model = _make_model()
    base = extract_self_dimensions(model, steps=50)
    apply_noise(model, noise_scale=5.0)
    perturbed = extract_self_dimensions(model, steps=50)
    # 强噪声应改变活动率
    assert perturbed["core_activity_rate"] != base["core_activity_rate"]


def test_apply_prune_changes_focus():
    model = _make_model()
    base = extract_self_dimensions(model, steps=50)
    apply_prune(model, fraction=0.5)
    perturbed = extract_self_dimensions(model, steps=50)
    # 剪除一半突触应改变核心聚焦度或活动率
    assert (perturbed["core_focus"] != base["core_focus"]
            or perturbed["core_activity_rate"] != base["core_activity_rate"])


def test_apply_lesion_changes_activity():
    model = _make_model()
    base = extract_self_dimensions(model, steps=50)
    apply_lesion(model, fraction=0.3)
    perturbed = extract_self_dimensions(model, steps=50)
    assert perturbed["core_activity_rate"] != base["core_activity_rate"]


def test_perturbation_sweep_returns_dataset():
    dataset = perturbation_sweep(
        n_units=300, core_size=40,
        noise_levels=[1.0, 3.0],
        prune_fractions=[0.0, 0.3],
        lesion_fractions=[0.0, 0.3],
        seed=0,
    )
    # 2×2×2 = 8 个组合
    assert len(dataset) == 8
    first = dataset[0]
    for key in ("noise_scale", "prune_fraction", "lesion_fraction",
                "core_activity_rate", "core_focus", "sync_index"):
        assert key in first
