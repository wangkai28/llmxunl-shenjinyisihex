import numpy as np

from bioelectric.persona import build_persona_network
from bioelectric.persona_environment import persona_under_perturbation


def test_persona_under_perturbation_returns_recovery():
    net = build_persona_network("tao", n_units=300, seed=0)
    result = persona_under_perturbation(net, core_size=40, seed=1,
                                        lesion_fraction=0.3)
    for key in ("baseline", "damaged", "recovered", "recovery_fraction"):
        assert key in result


def test_tao_recovers_better_than_id_under_lesion():
    # 道家应比本我更抗失活损伤（恢复力更高）
    id_net = build_persona_network("id", n_units=300, seed=0)
    tao_net = build_persona_network("tao", n_units=300, seed=0)
    id_result = persona_under_perturbation(id_net, core_size=40, seed=1,
                                           lesion_fraction=0.3)
    tao_result = persona_under_perturbation(tao_net, core_size=40, seed=1,
                                            lesion_fraction=0.3)
    assert tao_result["recovery_fraction"] > id_result["recovery_fraction"]
