from bioelectric.persona_evolution import evolve_persona, EVOLUTION_DIRECTION
from bioelectric.persona import PERSONAS


def test_danger_makes_more_impulsive():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "danger", lr=0.1)
    assert new["external_mean"] > cfg["external_mean"]  # 驱动更强
    assert abs(new["j_inh"]) < abs(cfg["j_inh"])        # 抑制更弱（冲动）


def test_moral_makes_more_restrained():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "moral", lr=0.1)
    assert abs(new["j_inh"]) > abs(cfg["j_inh"])        # 抑制更强（克制）
    assert new["external_mean"] < cfg["external_mean"]  # 驱动更低


def test_evolve_does_not_mutate_input():
    cfg = dict(PERSONAS["tao"])
    original = dict(cfg)
    evolve_persona(cfg, "danger", lr=0.1)
    assert cfg == original


def test_unknown_event_no_change():
    cfg = dict(PERSONAS["tao"])
    new = evolve_persona(cfg, "nonexistent", lr=0.1)
    assert new == cfg
