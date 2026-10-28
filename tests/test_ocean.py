from bioelectric.ocean import DEFAULT_OCEAN, ocean_to_params, evolve_ocean


def test_ocean_to_params_maps_extraversion_to_activity():
    low_e = ocean_to_params({**DEFAULT_OCEAN, "E": 0.2})
    high_e = ocean_to_params({**DEFAULT_OCEAN, "E": 0.8})
    assert high_e["external_mean"] > low_e["external_mean"]


def test_ocean_to_params_maps_neuroticism_to_noise():
    low_n = ocean_to_params({**DEFAULT_OCEAN, "N": 0.2})
    high_n = ocean_to_params({**DEFAULT_OCEAN, "N": 0.8})
    assert high_n["external_noise"] > low_n["external_noise"]


def test_ocean_to_params_maps_conscientiousness_to_inhibition():
    low_c = ocean_to_params({**DEFAULT_OCEAN, "C": 0.2})
    high_c = ocean_to_params({**DEFAULT_OCEAN, "C": 0.8})
    assert abs(high_c["j_inh"]) > abs(low_c["j_inh"])  # 越尽责抑制越强


def test_evolve_ocean_does_not_mutate():
    original = dict(DEFAULT_OCEAN)
    evolve_ocean(DEFAULT_OCEAN, "danger")
    assert DEFAULT_OCEAN == original


def test_evolve_ocean_danger_raises_neuroticism():
    new = evolve_ocean(DEFAULT_OCEAN, "danger", lr=0.05)
    assert new["N"] > DEFAULT_OCEAN["N"]
