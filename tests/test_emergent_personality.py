import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.multi_self import MultiSelf
from bioelectric.emergent_personality import read_personality


def _make_ms():
    net = BalancedNetwork(n_units=600, k_exc=10, j_inh=-4.5,
                          external_mean=0.15, external_noise=0.08, delay_max=4,
                          hebb_lr=0.02, weight_max=2.0, weight_decay=0.005, seed=0)
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02, cross_inhibition=0.1)
    ms.form(steps=200)
    return net, ms


def test_read_personality_returns_dimensions():
    net, ms = _make_ms()
    p = read_personality(net, ms)
    for key in ("impulsivity", "restraint", "activity", "stability"):
        assert key in p


def test_experience_changes_read_personality():
    net, ms = _make_ms()
    before = read_personality(net, ms)

    # 经历：持续激活本我核心（危险/应激），让 Hebb 固化本我回路
    for _ in range(300):
        inp = np.zeros(600)
        inp[ms.core_masks["id"]] = 3.0
        net.step(external_input=inp)

    after = read_personality(net, ms)
    # 经历后，本我的冲动性（核心兴奋边强度）应上升
    assert after["impulsivity"] > before["impulsivity"]
