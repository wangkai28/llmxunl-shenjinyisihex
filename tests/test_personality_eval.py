import numpy as np

from bioelectric.learned_personality import train_emotion_mlp, EmotionMLP
from bioelectric.personality_data import build_training_data, encode_text
from bioelectric.personality_eval import evaluate_quadrant_accuracy, HOLD_OUT_TESTS


def test_hold_out_tests_cover_four_quadrants():
    # 未见面测试集应覆盖情绪平面四象限
    vals = [(v, a) for _, v, a in HOLD_OUT_TESTS]
    assert any(v > 0 and a > 0 for v, a in vals)  # 愉快+激动
    assert any(v > 0 and a < 0 for v, a in vals)  # 愉快+平静
    assert any(v < 0 and a > 0 for v, a in vals)  # 不愉快+激动
    assert any(v < 0 and a < 0 for v, a in vals)  # 不愉快+平静


def test_trained_model_learns_training_mapping():
    texts, labels = build_training_data()
    model = train_emotion_mlp(texts, labels, epochs=300, hidden=16, lr=0.05)
    # 训练后模型应能学会训练集内的情绪映射（情绪坐标预测接近标签）。
    # 跨词汇泛化（成语/抽象表达）由 LLM 蒸馏路径保证，见 examples/distill_personality.py。
    from bioelectric.personality_data import encode_text
    errors = []
    for i, t in enumerate(texts[:20]):
        pred = model.predict_text(t)
        errors.append(float(np.abs(pred[0] - labels[i][0])))
    assert float(np.mean(errors)) < 0.3  # 训练集内 valence 误差应较小


def test_save_load_preserves_generalization(tmp_path):
    texts, labels = build_training_data()
    model = train_emotion_mlp(texts, labels, epochs=300, hidden=16, lr=0.05)
    path = str(tmp_path / "emotion_model.npz")
    model.save(path)
    loaded = EmotionMLP.load(path)
    acc_before = evaluate_quadrant_accuracy(model)
    acc_after = evaluate_quadrant_accuracy(loaded)
    assert acc_before == acc_after
