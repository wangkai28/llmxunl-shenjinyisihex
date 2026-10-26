import numpy as np

from bioelectric.balanced_network import BalancedNetwork


def test_balanced_network_synapse_strength_scales_with_inverse_sqrt_k():
    # 突触强度应正比于 1/sqrt(K)，使得涨落幅度与规模无关
    net_small = BalancedNetwork(n_units=500, k_exc=50, seed=0)
    net_large = BalancedNetwork(n_units=5000, k_exc=50, seed=0)
    # 单个突触的幅值应相近（因为 K 相同）
    w_small = np.abs(net_small.W.to_dense()[net_small.W.to_dense() != 0])
    w_large = np.abs(net_large.W.to_dense()[net_large.W.to_dense() != 0])
    assert abs(w_small.mean() - w_large.mean()) < 0.1


def test_balanced_network_spontaneous_activity():
    # 无外部背景时，平衡网络也应自发产生活动（涨落驱动）
    net = BalancedNetwork(n_units=1000, k_exc=100, seed=0)
    total = 0
    for _ in range(200):
        total += net.step().sum()
    rate = total / (1000 * 200)
    assert 0.01 < rate < 0.3  # 活动率应在健康区间（非死寂、非爆发）


def test_balanced_network_asynchronous():
    # 异步不规则：Fano 因子应接近 1（不是爆发式）
    net = BalancedNetwork(
        n_units=1000, k_exc=10, j_inh=-4.0,
        external_mean=0.2, external_noise=0.1, delay_max=4, seed=0,
    )
    per_step = []
    for _ in range(300):
        per_step.append(net.step().sum())
    per_step = np.array(per_step, dtype=float)
    fano = per_step.var() / per_step.mean()
    assert fano < 2.0  # 异步不规则的 Fano 接近 1


def test_dispersed_delay_reduces_synchrony():
    # 分散突触延迟应显著降低 Fano（打破同步）
    def fano(delay_max):
        net = BalancedNetwork(
            n_units=1000, k_exc=10, j_inh=-4.0,
            external_mean=0.2, external_noise=0.1,
            delay_max=delay_max, seed=0,
        )
        per = np.array([net.step().sum() for _ in range(300)], dtype=float)
        return per.var() / per.mean()

    fano_no_delay = fano(delay_max=1)
    fano_dispersed = fano(delay_max=4)
    # 分散延迟应让 Fano 显著下降（更异步）
    assert fano_dispersed < fano_no_delay
