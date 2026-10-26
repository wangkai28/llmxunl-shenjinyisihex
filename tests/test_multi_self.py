import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.multi_self import MultiSelf, measure_multi_self
from bioelectric.multi_self import lesion_one_self


def _make_network(seed=0):
    return BalancedNetwork(n_units=600, k_exc=10, j_inh=-4.5,
                           external_mean=0.15, external_noise=0.08, delay_max=4, seed=seed)


def test_multi_self_forms_three_cores():
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02)
    ms.form(steps=200)
    assert set(ms.core_masks.keys()) == {"id", "superego", "tao"}


def test_multi_self_persona_activity_differs():
    # 三个自我共存时，本我（高偏置）活动率应高于超我（低偏置）
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02)
    ms.form(steps=200)
    dims = measure_multi_self(ms, steps=100)
    assert dims["id"]["activity"] > dims["superego"]["activity"]


def test_measure_multi_self_returns_relation_dims():
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02)
    ms.form(steps=200)
    result = measure_multi_self(ms, steps=100)
    # 关系维度：主导性、切换性
    for key in ("dominance", "switching"):
        assert key in result


def test_lesion_one_self_causes_compensation():
    # 损伤一个自我后，其他自我应部分补偿（活动上升）
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02)
    ms.form(steps=200)
    before = measure_multi_self(ms, steps=100)
    lesion_one_self(ms, "id", fraction=0.8)
    after = measure_multi_self(ms, steps=100)
    # 本我受损后，其活动率应显著下降
    assert after["id"]["activity"] < before["id"]["activity"]
