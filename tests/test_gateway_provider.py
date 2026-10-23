import pytest

from bioelectric.llm_provider import (
    GatewayLLMProvider,
    _parse_reflection,
    Reflection,
)


def test_parse_reflection_valid_json():
    raw = '{"self_description": "a stable core", "predicted_bin_activity": [0.1, 0.2], "action": {"type": "inject", "region": 0, "strength": 1.0}}'
    r = _parse_reflection(raw, n_bins=8)
    assert isinstance(r, Reflection)
    assert r.self_description == "a stable core"
    assert r.predicted_bin_activity == [0.1, 0.2]
    assert r.action_type == "inject"
    assert r.region == 0
    assert r.strength == 1.0


def test_parse_reflection_missing_action_defaults_none():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1]}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity == [0.1]


def test_parse_reflection_invalid_json_degrades():
    r = _parse_reflection("not json at all", n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity is None


def test_parse_reflection_clamps_prediction_length():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1, 0.2, 0.3]}'
    r = _parse_reflection(raw, n_bins=2)
    assert r.predicted_bin_activity == [0.1, 0.2]
