from bioelectric.llm_annotate import LLMAnnotator


def test_parse_plain_json():
    r = LLMAnnotator._parse('{"valence": 0.8, "arousal": -0.7}')
    assert r == (0.8, -0.7)


def test_parse_markdown_fenced():
    r = LLMAnnotator._parse('```json\n{"valence": -0.5, "arousal": 0.9}\n```')
    assert r == (-0.5, 0.9)


def test_parse_clamps_out_of_range():
    r = LLMAnnotator._parse('{"valence": 1.5, "arousal": -2.0}')
    assert r == (1.0, -1.0)


def test_parse_invalid_returns_none():
    assert LLMAnnotator._parse("not json") is None
