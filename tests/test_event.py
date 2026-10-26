from bioelectric.event import Event, EVENT_TO_CONTEXT, encode_event


def test_event_dataclass():
    e = Event(type="danger", intensity=0.8, content="x")
    assert e.type == "danger"
    assert e.intensity == 0.8


def test_encode_event_maps_type_to_context():
    ctx = encode_event(Event(type="danger", intensity=1.0))
    assert ctx["id"] > ctx["superego"]  # 危险 → 本我主导


def test_encode_event_scales_by_intensity():
    full = encode_event(Event(type="danger", intensity=1.0))
    weak = encode_event(Event(type="danger", intensity=0.3))
    assert weak["id"] < full["id"]


def test_unknown_event_type_raises():
    import pytest
    with pytest.raises(ValueError):
        encode_event(Event(type="nonexistent", intensity=1.0))
