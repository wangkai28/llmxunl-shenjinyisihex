import numpy as np
import pytest

from bioelectric.llm_provider import MockLLMProvider
from bioelectric.observer import Observer
from bioelectric.spiking import SpikingNetwork


def test_observer_emits_reflections_at_interval():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(description="stable core")
    observer = Observer(net, provider, observe_every=5, n_bins=2, mode="readwrite")
    reflections = observer.run(12)
    assert len(reflections) == 2  # 第 5、10 步各观察一次


def test_observer_writeback_reaches_network():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=100.0, leak=0.0, dt=1.0, seed=0
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=1000.0
    )
    observer = Observer(net, provider, observe_every=1, n_bins=2, mode="readwrite")
    observer.tick()  # 观察并写入 pending input
    observer.tick()  # 下一时间步注入生效
    assert net.spikes[0] or net.spikes[1]


def test_observer_readonly_does_not_write_back():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=100.0, leak=0.0, dt=1.0, seed=0
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=1000.0
    )
    observer = Observer(net, provider, observe_every=1, n_bins=2, mode="readonly")
    observer.tick()
    observer.tick()
    assert not (net.spikes[0] or net.spikes[1])


def test_observer_rejects_zero_observe_every():
    net = SpikingNetwork(n_units=4, seed=0)
    with pytest.raises(ValueError):
        Observer(net, MockLLMProvider(), observe_every=0)
