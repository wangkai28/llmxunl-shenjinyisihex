import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.self_model import SelfModel, extract_self_dimensions


def test_self_model_can_hold_pattern_without_input():
    # 塑造一个核心区活动模式后，撤掉外部输入，模式应能部分自我维持
    net = BalancedNetwork(n_units=500, k_exc=10, j_inh=-4.0,
                          external_mean=0.2, external_noise=0.1, delay_max=4, seed=0)
    model = SelfModel(net, core_size=50, seed=1, hebb_lr=0.02)
    # 塑造阶段：给核心区持续强输入，跑若干步让 Hebb 固化
    model.form(steps=200)
    # 测试阶段：撤掉核心输入，看核心区活动率是否显著高于外围（自我维持）
    core_rate, periphery_rate = model.measure_holdout(steps=100)
    assert core_rate > periphery_rate  # 核心区仍更活跃


def test_extract_self_dimensions_returns_labeled_dict():
    net = BalancedNetwork(n_units=200, k_exc=10, j_inh=-4.0,
                          external_mean=0.2, external_noise=0.1, delay_max=4, seed=0)
    model = SelfModel(net, core_size=30, seed=1)
    model.form(steps=100)
    dims = extract_self_dimensions(model, steps=50)
    # 应包含可标注的认知维度
    for key in ("core_activity_rate", "periphery_activity_rate",
                "core_focus", "sync_index"):
        assert key in dims


def test_extract_self_dimensions_includes_new_dims():
    net = BalancedNetwork(n_units=200, k_exc=10, j_inh=-4.0,
                          external_mean=0.2, external_noise=0.1, delay_max=4, seed=0)
    model = SelfModel(net, core_size=30, seed=1, hebb_lr=0.02)
    model.form(steps=100)
    dims = extract_self_dimensions(model, steps=50)
    # 新增维度：核心空间漂移、核心连接强度
    for key in ("core_drift", "core_strength"):
        assert key in dims


def test_local_plasticity_allows_core_self_sustain():
    # 核心区可塑（Hebb 强化核心连接），外围不可塑 → 撤输入后核心应自我维持
    net = BalancedNetwork(n_units=500, k_exc=10, j_inh=-4.0,
                          external_mean=0.15, external_noise=0.08, delay_max=4, seed=0)
    model = SelfModel(net, core_size=60, seed=1, hebb_lr=0.02)
    model.form(steps=300)
    core_rate, peri_rate = model.measure_holdout(steps=100)
    assert core_rate > peri_rate * 1.2  # 核心区显著更活跃（自我维持）
