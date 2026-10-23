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
    before = net.W.copy()
    net.step(external_input=np.array([10.0, 10.0]))
    assert net.W[0, 1] > before[0, 1]
    assert net.W[1, 0] > before[1, 0]


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
