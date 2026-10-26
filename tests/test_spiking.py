import numpy as np
from bioelectric.spiking import SpikingNetwork


def test_step_returns_boolean_spike_vector():
    net = SpikingNetwork(n_units=20, seed=0)
    spikes = net.step()
    assert spikes.shape == (20,)
    assert spikes.dtype == bool


def test_strong_input_causes_firing():
    net = SpikingNetwork(n_units=10, seed=0)
    ext = np.zeros(10)
    ext[0] = 100.0
    fired = net.step(external_input=ext)
    assert fired[0]


def test_refractory_period_blocks_refiring():
    net = SpikingNetwork(
        n_units=1, connectivity=0.0, threshold=0.5, refractory=3,
        leak=0.0, dt=1.0, seed=0,
    )
    ext = np.array([10.0])
    assert net.step(external_input=ext)[0]
    for _ in range(3):
        assert not net.step(external_input=ext)[0]


def test_hebb_strengthens_coactive_pair():
    net = SpikingNetwork(
        n_units=2, connectivity=0.0, threshold=0.5, refractory=0,
        leak=0.0, dt=1.0, hebb_lr=0.1, weight_decay=0.0, weight_max=2.0, seed=0,
    )
    # 先手动建立两条边，Hebb 应在已存在的边上增强
    net.W[0, 1] = 0.5
    net.W[1, 0] = 0.5
    before01 = net.W[0, 1]
    before10 = net.W[1, 0]
    net.step(external_input=np.array([10.0, 10.0]))
    assert net.W[0, 1] > before01
    assert net.W[1, 0] > before10


def test_refractory_clamps_membrane_potential():
    net = SpikingNetwork(
        n_units=1, connectivity=0.0, threshold=0.5, refractory=3,
        leak=0.0, dt=1.0, seed=0,
    )
    assert net.step(external_input=np.array([10.0]))[0]
    assert not net.step(external_input=np.array([10.0]))[0]
    assert net.V[0] == 0.0
    for _ in range(4):
        assert not net.step(external_input=np.array([0.0]))[0]
        assert net.V[0] == 0.0


def test_inhibitory_units_have_negative_output_connections():
    net = SpikingNetwork(
        n_units=20, connectivity=0.2, inhibitory_fraction=0.2, seed=0,
    )
    inhib = net.is_inhibitory
    assert inhib.sum() == 4  # 20 * 0.2
    # 抑制单元作为 presynaptic 的输出连接（W[:, inhib]）全部非正
    assert (net.W[:, inhib] <= 0).all()
    # 且至少存在一条真实的负连接
    assert (net.W[:, inhib] < 0).any()


def test_hebb_preserves_inhibitory_sign():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=0.5, refractory=0,
        leak=0.0, dt=1.0, hebb_lr=0.1, weight_decay=0.0, weight_max=2.0,
        inhibitory_fraction=0.25, seed=0,
    )
    inhib_idx = int(np.flatnonzero(net.is_inhibitory)[0])
    exc_idx = int(np.flatnonzero(~net.is_inhibitory)[0])
    net.W[exc_idx, inhib_idx] = -0.5
    before = net.W[exc_idx, inhib_idx]
    ext = np.zeros(4)
    ext[exc_idx] = 10.0
    ext[inhib_idx] = 10.0
    net.step(external_input=ext)
    after = net.W[exc_idx, inhib_idx]
    assert after < before      # magnitude 增强
    assert after < 0.0         # 符号仍为负（抑制未被吞掉）


def test_hebb_preserves_excitatory_sign():
    net = SpikingNetwork(
        n_units=2, connectivity=0.0, threshold=0.5, refractory=0,
        leak=0.0, dt=1.0, hebb_lr=0.1, weight_decay=0.0, weight_max=2.0,
        inhibitory_fraction=0.0, seed=0,
    )
    net.W[0, 1] = 0.5
    before = net.W[0, 1]
    net.step(external_input=np.array([10.0, 10.0]))
    after = net.W[0, 1]
    assert after > before      # magnitude 增强
    assert after > 0.0         # 符号仍为正


def test_background_causes_spontaneous_firing():
    net = SpikingNetwork(
        n_units=64, connectivity=0.0, threshold=1.0, leak=0.1, dt=1.0,
        background_rate=0.2, background_strength=0.5, seed=0,
    )
    total = 0
    for _ in range(500):
        total += net.step().sum()
    assert total > 0  # 有背景活动后，网络能自发发放


def test_no_background_stays_silent_without_input():
    net = SpikingNetwork(n_units=64, connectivity=0.0, seed=0)
    total = 0
    for _ in range(200):
        total += net.step().sum()
    assert total == 0  # 默认无背景、无输入时保持静默（向后兼容）


def test_background_is_deterministic_given_seed():
    def run():
        net = SpikingNetwork(
            n_units=32, connectivity=0.0, threshold=1.0, leak=0.1, dt=1.0,
            background_rate=0.2, background_strength=0.5, seed=7,
        )
        trace = []
        for _ in range(50):
            trace.append(net.step().sum())
        return trace
    assert run() == run()  # 同 seed 背景序列可复现


def test_larger_network_runs_fast_enough():
    import time
    net = SpikingNetwork(
        n_units=20000, connectivity=0.02, threshold=1.0, leak=0.1, dt=1.0,
        background_rate=0.02, background_strength=0.6, seed=0,
    )
    t0 = time.time()
    for _ in range(20):
        net.step()
    dt = time.time() - t0
    # 20000 单元、20 步，应远快于 O(N²)（旧实现约 90 秒）。
    # 稀疏优化后约 6 秒（800 万条边），阈值取 15 秒留余量。
    assert dt < 15.0, f"too slow: {dt:.2f}s for 20 steps at n=20000"
