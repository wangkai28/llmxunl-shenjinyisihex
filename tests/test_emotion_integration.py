from bioelectric.codec import emotion_to_context


def test_anger_maps_to_id():
    ctx = emotion_to_context(-0.8, 0.9)  # 愤怒/恐惧
    assert max(ctx, key=ctx.get) == "id"


def test_calm_maps_to_tao():
    ctx = emotion_to_context(0.6, -0.7)  # 平静
    assert max(ctx, key=ctx.get) == "tao"


def test_joy_maps_to_id():
    ctx = emotion_to_context(0.8, 0.8)  # 兴奋开心
    assert max(ctx, key=ctx.get) == "id"


def test_sadness_maps_to_id():
    ctx = emotion_to_context(-0.7, -0.5)  # 低落/委屈
    assert max(ctx, key=ctx.get) == "id"


def test_intensity_scales_with_arousal():
    high = emotion_to_context(0.5, 0.9)
    low = emotion_to_context(0.5, 0.1)
    # 高唤醒 → 更强的主导 bias
    assert max(high.values()) > max(low.values())
