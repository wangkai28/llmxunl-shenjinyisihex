import pytest

from bioelectric.llm_provider import (
    GatewayLLMProvider,
    _parse_reflection,
    Reflection,
)


def test_parse_reflection_valid_json():
    raw = '{"self_description": "a stable core", "predicted_bin_activity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "action": {"type": "inject", "region": 0, "strength": 1.0}}'
    r = _parse_reflection(raw, n_bins=8)
    assert isinstance(r, Reflection)
    assert r.self_description == "a stable core"
    assert r.predicted_bin_activity == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert r.action_type == "inject"
    assert r.region == 0
    assert r.strength == 1.0


def test_parse_reflection_missing_action_defaults_none():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def test_parse_reflection_invalid_json_degrades():
    r = _parse_reflection("not json at all", n_bins=8)
    assert r.action_type == "none"
    assert r.predicted_bin_activity is None


def test_parse_reflection_clamps_prediction_length():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1, 0.2, 0.3]}'
    r = _parse_reflection(raw, n_bins=2)
    assert r.predicted_bin_activity == [0.1, 0.2]


def test_parse_reflection_non_dict_action_degrades():
    raw = '{"self_description": "x", "action": "inject"}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"


def test_parse_reflection_non_numeric_strength_degrades():
    raw = '{"self_description": "x", "action": {"type": "inject", "region": 0, "strength": "high"}}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"


def test_parse_reflection_null_action_type_normalizes():
    raw = '{"self_description": "x", "action": {"type": null, "region": 0, "strength": 0.0}}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"


def test_parse_reflection_overflow_number_degrades():
    raw = '{"self_description": "x", "action": {"type": "inject", "region": 1e400, "strength": 0.0}}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.action_type == "none"


def test_parse_reflection_short_prediction_becomes_none():
    raw = '{"self_description": "x", "predicted_bin_activity": [0.1, 0.2]}'
    r = _parse_reflection(raw, n_bins=4)
    assert r.predicted_bin_activity is None


def test_build_prompt_encodes_active_shaper_role():
    provider = GatewayLLMProvider(
        base_url="http://unused", api_key="unused", model="unused", n_bins=8,
    )
    prompt = provider._build_prompt("snapshot", [])
    # 方案 A：prompt 必须明确让 LLM 用注入来塑造网络、对齐自我描述
    assert "inject" in prompt.lower()
    assert "shape" in prompt.lower() or "steer" in prompt.lower()
    assert "self_description" in prompt


def test_parse_reflection_identity_continuity():
    raw = '{"self_description": "I am the same observer", "identity_continuity": 0.9}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.identity_continuity == 0.9


def test_parse_reflection_missing_identity_continuity_defaults_none():
    raw = '{"self_description": "x"}'
    r = _parse_reflection(raw, n_bins=8)
    assert r.identity_continuity is None


def test_build_prompt_requests_identity_continuity():
    provider = GatewayLLMProvider(
        base_url="http://unused", api_key="unused", model="unused", n_bins=8,
    )
    prompt = provider._build_prompt("snapshot", ["I am an observer"])
    assert "identity_continuity" in prompt
    assert "same observer" in prompt.lower() or "still myself" in prompt.lower()
