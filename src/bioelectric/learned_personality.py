from __future__ import annotations

import numpy as np

from .personality_data import encode_text, build_feature_matrix


class EmotionMLP:
    """小型 MLP：BOW 文本特征 → 情绪坐标 (valence, arousal)。

    结构：input → hidden(relu) → output(2, tanh)。输出经 tanh 压到 [-1,1]。
    """

    def __init__(self, input_dim: int, hidden: int):
        rng = np.random.default_rng(0)
        self.input_dim = input_dim
        self.hidden = hidden
        self.vocab = None  # 词表（predict 时编码用）
        self.W1 = rng.normal(0.0, 0.5, size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, 0.5, size=(hidden, 2))
        self.b2 = np.zeros(2)

    def _forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)  # relu
        z2 = a1 @ self.W2 + self.b2
        a2 = np.tanh(z2)  # 输出 [-1, 1]
        return a1, a2

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, out = self._forward(x)
        return out

    def predict_text(self, text: str) -> np.ndarray:
        """预测一句文本的情绪坐标（用模型持有的词表编码）。"""
        if self.vocab is None:
            raise RuntimeError("model has no vocab; it must be trained first")
        vec = np.zeros(len(self.vocab), dtype=float)
        for g, c in encode_text(text).items():
            if g in self.vocab:
                vec[self.vocab[g]] = c
        return self.predict(vec)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        a1, a2 = self._forward(x)
        # MSE 损失 + 反向传播
        error = a2 - y  # (2,)
        # 输出层 tanh 导数
        dz2 = error * (1.0 - a2 ** 2)
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        # 隐层 relu 导数
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (a1 > 0).astype(float)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        # 更新
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return float(np.mean(error ** 2))

    def save(self, path: str):
        vocab = self.vocab if self.vocab is not None else {}
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 input_dim=self.input_dim, hidden=self.hidden,
                 vocab_keys=list(vocab.keys()), vocab_vals=list(vocab.values()))

    @classmethod
    def load(cls, path: str) -> "EmotionMLP":
        data = np.load(path)
        model = cls(int(data["input_dim"]), int(data["hidden"]))
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        if "vocab_keys" in data:
            model.vocab = {k: int(v) for k, v in zip(data["vocab_keys"], data["vocab_vals"])}
        return model


def train_emotion_mlp(texts, labels, *, epochs=200, hidden=16, lr=0.05):
    """训练 EmotionMLP：texts 是语句列表，labels 是 (valence, arousal) 数组。"""
    X, vocab = build_feature_matrix(texts)
    Y = np.asarray(labels, dtype=float)
    model = EmotionMLP(input_dim=X.shape[1], hidden=hidden)
    model.vocab = vocab
    for _ in range(epochs):
        for i in range(X.shape[0]):
            model.train_step(X[i], Y[i], lr)
    return model
