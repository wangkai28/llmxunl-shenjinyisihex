import numpy as np

from bioelectric.persona import PERSONAS, build_persona_network, measure_persona_dimensions


def test_build_persona_network_accepts_all_personas():
    for name in ("id", "superego", "tao"):
        net = build_persona_network(name, n_units=300, seed=0)
        assert net is not None
        assert net.n_units == 300


def test_measure_persona_dimensions_returns_character_traits():
    net = build_persona_network("id", n_units=300, seed=0)
    dims = measure_persona_dimensions(net, core_size=40, seed=1)
    for key in ("impulsivity", "restraint", "adaptability", "stability"):
        assert key in dims


def test_id_is_more_impulsive_than_superego():
    # 本我应比超我更冲动（更高的核心活动率/兴奋性）
    id_net = build_persona_network("id", n_units=300, seed=0)
    super_net = build_persona_network("superego", n_units=300, seed=0)
    id_dims = measure_persona_dimensions(id_net, core_size=40, seed=1)
    super_dims = measure_persona_dimensions(super_net, core_size=40, seed=1)
    assert id_dims["impulsivity"] > super_dims["impulsivity"]


def test_superego_is_more_stable_than_id():
    id_net = build_persona_network("id", n_units=300, seed=0)
    super_net = build_persona_network("superego", n_units=300, seed=0)
    id_dims = measure_persona_dimensions(id_net, core_size=40, seed=1)
    super_dims = measure_persona_dimensions(super_net, core_size=40, seed=1)
    assert super_dims["stability"] > id_dims["stability"]


def test_persona_dimensions_include_resilience_and_inhibitory_control():
    net = build_persona_network("tao", n_units=300, seed=0)
    dims = measure_persona_dimensions(net, core_size=40, seed=1)
    for key in ("resilience", "inhibitory_control"):
        assert key in dims


def test_superego_has_higher_inhibitory_control_than_id():
    # 超我应比本我有更强的抑制控制（更高 j_inh → 核心区抑制突触更强）
    id_net = build_persona_network("id", n_units=300, seed=0)
    super_net = build_persona_network("superego", n_units=300, seed=0)
    id_dims = measure_persona_dimensions(id_net, core_size=40, seed=1)
    super_dims = measure_persona_dimensions(super_net, core_size=40, seed=1)
    assert super_dims["inhibitory_control"] > id_dims["inhibitory_control"]
