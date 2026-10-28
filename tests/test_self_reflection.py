from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


def _core(tmp_path):
    return IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)


def test_self_reflection_returns_text(tmp_path):
    core = _core(tmp_path)
    core.process_event(Event(type="calm", intensity=1.0))
    text = core.self_reflect()
    assert isinstance(text, str)
    assert len(text) > 0


def test_self_reflection_mentions_dominant_self(tmp_path):
    core = _core(tmp_path)
    core.process_event(Event(type="danger", intensity=1.0))
    text = core.self_reflect()
    # 危险 → 本我主导，文本应提到「本我」或相关情绪
    assert "本我" in text or "冲动" in text or "亢奋" in text


def test_self_reflection_mentions_ocean(tmp_path):
    core = _core(tmp_path)
    text = core.self_reflect()
    # 应包含人格底色的描述
    assert "开放性" in text or "尽责" in text or "外向" in text or "神经质" in text


def test_self_reflection_mentions_history(tmp_path):
    core = _core(tmp_path)
    core.process_event(Event(type="danger", intensity=1.0))
    core.process_event(Event(type="calm", intensity=1.0))
    text = core.self_reflect()
    # 应提到最近的经历
    assert "经历" in text or "最近" in text or "危险" in text or "平静" in text
