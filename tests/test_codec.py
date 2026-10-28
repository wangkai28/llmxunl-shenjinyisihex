from bioelectric.codec import encode_input, decode_state
from bioelectric.event import Event


def test_encode_input_passthrough_event():
    ev = Event(type="danger", intensity=0.9, content="x")
    assert encode_input(ev) == ev


def test_encode_input_from_dict():
    ev = encode_input({"type": "calm", "intensity": 0.5, "content": "x"})
    assert isinstance(ev, Event)
    assert ev.type == "calm"
    assert ev.intensity == 0.5


def test_encode_input_from_text_rule():
    # 规则版文本编码：关键词 → 事件类型
    ev = encode_input("危险！有威胁")
    assert ev.type in ("danger", "threat")


def test_encode_input_unknown_text_defaults_calm():
    ev = encode_input("今天天气不错")
    assert ev.type == "calm"


def test_decode_state_returns_dict():
    state = {"dominant": "id", "id": 0.3, "superego": 0.1, "tao": 0.2,
             "ocean": {"O": 0.5, "E": 0.5, "C": 0.5, "A": 0.5, "N": 0.5}}
    out = decode_state(state)
    assert isinstance(out, dict)
    assert "dominant" in out
