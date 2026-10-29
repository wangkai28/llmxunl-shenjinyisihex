from __future__ import annotations

import numpy as np

from .personality_data import encode_text, build_feature_matrix


# 9 维：情绪(2) + 性格(5) + 意识(2)
DIMENSIONS = [
    "valence", "arousal",
    "openness", "extraversion", "conscientiousness",
    "agreeableness", "neuroticism",
    "self_clarity", "agency",
]


class UnifiedPersonalityMLP:
    """9 维统一人格模型：文本 → (情绪2, 性格5, 意识2)。

    输出分三组：情绪用 tanh（[-1,1]），性格与意识用 sigmoid（[0,1]）。
    """

    def __init__(self, input_dim: int, hidden: int):
        rng = np.random.default_rng(0)
        self.input_dim = input_dim
        self.hidden = hidden
        self.vocab = None
        self.W1 = rng.normal(0.0, 0.5, size=(input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, 0.5, size=(hidden, 9))
        self.b2 = np.zeros(9)

    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        out = np.zeros(9)
        out[0:2] = np.tanh(z2[0:2])          # 情绪 [-1,1]
        out[2:9] = 1.0 / (1.0 + np.exp(-z2[2:9]))  # 性格+意识 [0,1]
        return a1, z2, out

    def predict(self, x):
        _, _, out = self._forward(x)
        return out

    def predict_text(self, text):
        if self.vocab is None:
            raise RuntimeError("model has no vocab; must be trained first")
        vec = np.zeros(len(self.vocab), dtype=float)
        for g, c in encode_text(text).items():
            if g in self.vocab:
                vec[self.vocab[g]] = c
        return self.predict(vec)

    def train_step(self, x, y, lr):
        a1, z2, out = self._forward(x)
        error = out - y
        # 输出层：情绪 tanh 导数，其余 sigmoid 导数
        dz2 = np.zeros(9)
        dz2[0:2] = error[0:2] * (1.0 - out[0:2] ** 2)
        dz2[2:9] = error[2:9] * out[2:9] * (1.0 - out[2:9])
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (a1 > 0).astype(float)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return float(np.mean(error ** 2))

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 input_dim=self.input_dim, hidden=self.hidden,
                 vocab_keys=list(self.vocab.keys()) if self.vocab else [],
                 vocab_vals=list(self.vocab.values()) if self.vocab else [])

    @classmethod
    def load(cls, path):
        data = np.load(path)
        m = cls(int(data["input_dim"]), int(data["hidden"]))
        m.W1 = data["W1"]; m.b1 = data["b1"]
        m.W2 = data["W2"]; m.b2 = data["b2"]
        if "vocab_keys" in data and len(data["vocab_keys"]) > 0:
            m.vocab = {k: int(v) for k, v in zip(data["vocab_keys"], data["vocab_vals"])}
        return m


def train_unified_mlp(texts, labels, *, epochs=300, hidden=32, lr=0.05):
    X, vocab = build_feature_matrix(texts)
    Y = np.asarray(labels, dtype=float)
    model = UnifiedPersonalityMLP(input_dim=X.shape[1], hidden=hidden)
    model.vocab = vocab
    for _ in range(epochs):
        for i in range(X.shape[0]):
            model.train_step(X[i], Y[i], lr)
    return model
