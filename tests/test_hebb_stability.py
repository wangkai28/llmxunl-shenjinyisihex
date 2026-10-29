import numpy as np

from bioelectric.balanced_network import BalancedNetwork


def test_hebb_weights_do_not_diverge():
    # 长期反复激活同一核心，权重应稳定在绝对上限内，而非发散
    net = BalancedNetwork(n_units=300, k_exc=10, j_inh=-4.5,
                          external_mean=0.15, external_noise=0.08, delay_max=4,
                          hebb_lr=0.02, seed=0)
    # 标记一部分边可塑（核心区）
    post = net._edge_post
    pre = net._edge_pre
    core = (post < 50) & (pre < 50) & (net._edge_w > 0)
    net.mark_plastic_edges(core)

    # 反复激活核心区 500 步
    for _ in range(500):
        inp = np.zeros(300)
        inp[:50] = 3.0
        net.step(external_input=inp)

    exc = net._edge_w[net._edge_w > 0]
    # 权重应稳定在一个绝对上限内，不爆炸
    assert exc.max() < 10.0


def test_hebb_decay_reduces_unused_weights():
    # 有遗忘时，持续不激活的连接应衰减
    net = BalancedNetwork(n_units=300, k_exc=10, j_inh=-4.5,
                          external_mean=0.15, external_noise=0.08, delay_max=4,
                          hebb_lr=0.02, weight_decay=0.01, seed=0)
    post = net._edge_post
    pre = net._edge_pre
    core = (post < 50) & (pre < 50) & (net._edge_w > 0)
    net.mark_plastic_edges(core)

    # 激活一次让核心边增强
    inp = np.zeros(300)
    inp[:50] = 3.0
    net.step(external_input=inp)
    w_after_activation = net._edge_w[core].copy()

    # 然后不激活，跑很多步，遗忘应让权重回落
    for _ in range(200):
        net.step(external_input=None)
    w_after_decay = net._edge_w[core]
    assert w_after_decay.mean() < w_after_activation.mean()
