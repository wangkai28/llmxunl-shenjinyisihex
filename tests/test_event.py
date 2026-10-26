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


def test_dominant_invariant_to_intensity():
    for event_type in EVENT_TO_CONTEXT:
        full = encode_event(Event(type=event_type, intensity=1.0))
        weak = encode_event(Event(type=event_type, intensity=0.3))
        zero = encode_event(Event(type=event_type, intensity=0.0))
        assert max(full, key=full.get) == max(weak, key=weak.get) == max(zero, key=zero.get)


def test_danger_zero_intensity_still_id_dominant():
    ctx = encode_event(Event(type="danger", intensity=0.0))
    assert max(ctx, key=ctx.get) == "id"


def test_praise_has_strict_dominant():
    full = encode_event(Event(type="praise", intensity=1.0))
    half = encode_event(Event(type="praise", intensity=0.5))
    assert max(full, key=full.get) == "id"
    assert max(half, key=half.get) == "id"
