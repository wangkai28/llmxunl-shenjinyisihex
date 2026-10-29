import numpy as np

from bioelectric.unified_personality import UnifiedPersonalityMLP, DIMENSIONS
from bioelectric.llm_annotate import LLMAnnotator


def test_dimensions_are_nine():
    assert len(DIMENSIONS) == 9


def test_unified_mlp_predicts_nine_dims():
    model = UnifiedPersonalityMLP(input_dim=100, hidden=16)
    x = np.zeros(100)
    out = model.predict(x)
    assert out.shape == (9,)


def test_llm_annotator_parses_unified_json():
    raw = '{"valence":0.65,"arousal":0.35,"openness":0.6,"extraversion":0.35,"conscientiousness":0.65,"agreeableness":0.5,"neuroticism":0.25,"self_clarity":0.85,"agency":0.7}'
    result = LLMAnnotator._parse_unified(raw)
    assert result is not None
    assert len(result) == 9
    assert result[7] == 0.85  # self_clarity
