from __future__ import annotations

import numpy as np

from .personality_data import encode_text


# 未见面测试集：训练数据里没有的语句，覆盖情绪四象限。
# 每项 (语句, valence, arousal)
HOLD_OUT_TESTS = [
    # 愉快 + 激动
    ("真让人振奋", 0.8, 0.75),
    ("太令人惊喜了", 0.85, 0.8),
    ("这真是天大的好事", 0.8, 0.7),
    # 愉快 + 平静
    ("心里很踏实", 0.65, -0.4),
    ("一切都好起来了", 0.7, -0.3),
    ("感觉挺安稳的", 0.6, -0.5),
    # 不愉快 + 激动
    ("气得浑身发抖", -0.8, 0.9),
    ("简直要气疯了", -0.8, 0.9),
    ("吓死我了", -0.7, 0.85),
    # 不愉快 + 平静
    ("心里空落落的", -0.6, -0.5),
    ("整个人没精神", -0.55, -0.6),
    ("感觉被掏空了", -0.65, -0.5),
]


def _quadrant(v, a):
    """返回情绪所在象限：0=愉快+激动, 1=愉快+平静, 2=不愉快+激动, 3=不愉快+平静"""
    if v >= 0 and a >= 0:
        return 0
    if v >= 0 and a < 0:
        return 1
    if v < 0 and a >= 0:
        return 2
    return 3


def evaluate_quadrant_accuracy(model) -> float:
    """在未见面测试集上评估四象限分类准确率（0~1）。"""
    correct = 0
    total = 0
    for text, v, a in HOLD_OUT_TESTS:
        pred = model.predict_text(text)
        if _quadrant(pred[0], pred[1]) == _quadrant(v, a):
            correct += 1
        total += 1
    return correct / total if total else 0.0
