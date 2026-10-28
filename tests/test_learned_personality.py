import numpy as np

from bioelectric.learned_personality import EmotionMLP, train_emotion_mlp
from bioelectric.personality_data import build_training_data, encode_text


def test_build_training_data_returns_texts_and_labels():
    texts, labels = build_training_data()
    assert len(texts) == len(labels)
    assert len(texts) >= 50  # 至少几十条
    # 标签是 (valence, arousal)，范围 [-1, 1]
    labels = np.asarray(labels)
    assert labels.shape[1] == 2
    assert labels.min() >= -1.0
    assert labels.max() <= 1.0


def test_encode_text_returns_fixed_dim_vector():
    x = encode_text("今天很开心")
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    assert len(x) > 0


def test_train_emotion_mlp_learns_mapping():
    texts, labels = build_training_data()
    model = train_emotion_mlp(texts, labels, epochs=200, hidden=16, lr=0.05)
    # 训练后应能大致预测训练集（开心句 valence 高）
    happy_idx = None
    for i, t in enumerate(texts):
        if "开心" in t or "高兴" in t or "快乐" in t:
            happy_idx = i
            break
    if happy_idx is not None:
        pred = model.predict(encode_text(texts[happy_idx]))
        assert pred[0] > 0.0  # valence 应为正（愉快）


def test_model_save_load(tmp_path):
    texts, labels = build_training_data()
    model = train_emotion_mlp(texts, labels, epochs=50, hidden=8, lr=0.05)
    path = str(tmp_path / "model.npz")
    model.save(path)
    loaded = EmotionMLP.load(path)
    # 同输入应给出相近输出
    x = encode_text("今天很开心")
    assert np.allclose(model.predict(x), loaded.predict(x), atol=1e-6)
