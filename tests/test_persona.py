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
