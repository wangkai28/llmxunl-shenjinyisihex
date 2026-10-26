import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.multi_self import MultiSelf, measure_multi_self
from bioelectric.multi_self import lesion_one_self
from bioelectric.multi_self import set_context, dominant_self
from bioelectric.multi_self import transition_trace


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


def test_cross_inhibition_causes_compensation():
    # 加入交叉抑制后，损伤本我 → 超我/道家应补偿性上升
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02, cross_inhibition=3.0)
    ms.form(steps=200)
    before = measure_multi_self(ms, steps=100)
    lesion_one_self(ms, "id", fraction=0.8)
    after = measure_multi_self(ms, steps=100)
    # 本我受损，对超我的抑制解除，超我应上升
    assert after["superego"]["activity"] > before["superego"]["activity"]


def test_context_switches_dominant_self():
    # 不同情境应让不同的自我主导
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02, cross_inhibition=3.0)
    ms.form(steps=200)

    # 危险情境：本我主导
    set_context(ms, {"id": 3.0, "superego": 0.2, "tao": 0.2})
    assert dominant_self(ms, steps=50) == "id"

    # 道德情境：超我主导
    set_context(ms, {"id": 0.2, "superego": 3.0, "tao": 0.2})
    assert dominant_self(ms, steps=50) == "superego"


def test_transition_trace_returns_activity_over_time():
    net = _make_network()
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02, cross_inhibition=3.0)
    ms.form(steps=200)
    trace = transition_trace(
        ms,
        from_context={"id": 3.0, "superego": 0.2, "tao": 0.2},
        to_context={"id": 0.2, "superego": 3.0, "tao": 0.2},
        settle_steps=50, trace_steps=100, sample_every=10,
    )
    # trace 应包含每个采样点的三个自我活动率
    assert len(trace) == 10  # 100/10 = 10 个采样点
    for key in ("id", "superego", "tao"):
        assert key in trace[0]
