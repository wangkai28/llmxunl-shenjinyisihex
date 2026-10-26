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


def test_observer_rejects_invalid_mode():
    net = SpikingNetwork(n_units=4, seed=0)
    with pytest.raises(ValueError):
        Observer(net, MockLLMProvider(), mode="ReadWrite")


def test_observer_rejects_zero_n_bins():
    net = SpikingNetwork(n_units=4, seed=0)
    with pytest.raises(ValueError):
        Observer(net, MockLLMProvider(), n_bins=0)


def test_writeback_is_sustained_across_multiple_steps():
    # 注入应持续多个时间步，而不是只喂给下一个时间步就消失
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=0.5, leak=0.0, dt=1.0, seed=0,
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=0.4,
    )
    observer = Observer(
        net, provider, observe_every=2, n_bins=2, mode="readwrite",
        writeback_duration=3, writeback_gain=1.0,
    )
    observer.tick()  # 第 1 步：无观察
    observer.tick()  # 第 2 步：观察并注入（持续 3 步）
    fired_any = False
    for _ in range(4):
        observer.tick()
        if net.spikes.any():
            fired_any = True
    assert fired_any  # 持续注入期间网络应能累积到阈值并发放


def test_writeback_gain_scales_strength():
    net = SpikingNetwork(
        n_units=2, connectivity=0.0, threshold=0.5, leak=0.0, dt=1.0, seed=0,
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=0.2,
    )
    # gain=5 让 0.2 的意图变成 1.0 的物理强度，足以跨越阈值
    observer = Observer(
        net, provider, observe_every=1, n_bins=1, mode="readwrite",
        writeback_duration=1, writeback_gain=5.0,
    )
    observer.tick()  # 观察并注入
    observer.tick()  # 下一时间步注入生效
    assert net.spikes[0] or net.spikes[1]


def test_error_driven_policy_writes_back_only_when_error_large():
    net = SpikingNetwork(
        n_units=4, connectivity=0.0, threshold=0.5, leak=0.0, dt=1.0, seed=0,
    )
    provider = MockLLMProvider(
        description="x", action_type="inject", region=0, strength=0.4,
    )
    observer = Observer(
        net, provider, observe_every=1, n_bins=2, mode="readwrite",
        writeback_gain=2.0, writeback_duration=1,
        writeback_policy="error-driven", error_threshold=0.1,
    )
    # 误差小 → 不写回
    observer.feedback_error(0.01)
    observer.tick()
    observer.tick()
    assert not (net.spikes[0] or net.spikes[1])
    # 误差大 → 写回
    observer.feedback_error(0.5)
    observer.tick()
    observer.tick()
    assert net.spikes[0] or net.spikes[1]
