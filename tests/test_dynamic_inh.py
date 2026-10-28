import numpy as np

from bioelectric.balanced_network import BalancedNetwork


def test_set_j_inh_rescales_inhibitory_edges():
    net = BalancedNetwork(n_units=200, k_exc=10, j_inh=-4.0,
                          external_mean=0.15, external_noise=0.08, delay_max=4, seed=0)
    # 记录初始抑制边权重
    inh_mask = net._edge_w < 0
    before = net._edge_w[inh_mask].copy()
    # 动态增强抑制（更负）
    net.set_j_inh(-6.0)
    after = net._edge_w[inh_mask]
    # 所有抑制边都应等比例变得更负（6/4 = 1.5 倍）
    assert np.allclose(after / before, 1.5, rtol=1e-9)


def test_set_j_inh_preserves_excitatory_edges():
    net = BalancedNetwork(n_units=200, k_exc=10, j_inh=-4.0,
                          external_mean=0.15, external_noise=0.08, delay_max=4, seed=0)
    exc_mask = net._edge_w > 0
    before = net._edge_w[exc_mask].copy()
    net.set_j_inh(-6.0)
    after = net._edge_w[exc_mask]
    # 兴奋边完全不变
    assert np.array_equal(before, after)


def test_set_j_inh_updates_attribute():
    net = BalancedNetwork(n_units=100, k_exc=10, j_inh=-4.0, seed=0)
    net.set_j_inh(-5.0)
    assert net.j_inh == -5.0


def test_prune_then_set_j_inh_no_error():
    net = BalancedNetwork(n_units=200, k_exc=10, j_inh=-4.0,
                          external_mean=0.15, external_noise=0.08, delay_max=4, seed=0)
    net.prune_edges(0.3)
    net.set_j_inh(-6.0)  # 不应抛 IndexError
    assert net.j_inh == -6.0
