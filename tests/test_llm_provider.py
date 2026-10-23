import pytest

from bioelectric.llm_provider import LLMProvider, MockLLMProvider, Reflection


def test_mock_returns_fixed_reflection():
    provider = MockLLMProvider(
        description="I am a network",
        action_type="inject",
        region=2,
        strength=0.5,
    )
    result = provider.reflect("snapshot", [])
    assert isinstance(result, Reflection)
    assert result.self_description == "I am a network"
    assert result.action_type == "inject"
    assert result.region == 2
    assert result.strength == 0.5


def test_base_provider_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMProvider().reflect("snapshot", [])
