import numpy as np

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.multi_self import MultiSelf
from bioelectric.emergent_personality import read_personality, emotion_experience


def _make_ms():
    net = BalancedNetwork(n_units=600, k_exc=10, j_inh=-4.5,
                          external_mean=0.15, external_noise=0.08, delay_max=4,
                          hebb_lr=0.02, weight_max=2.0, weight_decay=0.005, seed=0)
    ms = MultiSelf(net, core_specs={
        "id": {"size": 50, "bias": 3.0},
        "superego": {"size": 50, "bias": 0.5},
        "tao": {"size": 50, "bias": 1.5},
    }, seed=1, hebb_lr=0.02, cross_inhibition=0.1)
    ms.form(steps=200)
    return net, ms


def test_read_personality_returns_dimensions():
    net, ms = _make_ms()
    p = read_personality(net, ms)
    for key in ("impulsivity", "restraint", "activity", "stability"):
        assert key in p


def test_experience_changes_read_personality():
    net, ms = _make_ms()
    before = read_personality(net, ms)

    # 经历：持续激活本我核心（危险/应激），让 Hebb 固化本我回路
    for _ in range(300):
        inp = np.zeros(600)
        inp[ms.core_masks["id"]] = 3.0
        net.step(external_input=inp)

    after = read_personality(net, ms)
    # 经历后，本我的冲动性（核心兴奋边强度）应上升
    assert after["impulsivity"] > before["impulsivity"]


def test_emotion_experience_closes_loop():
    # 情绪坐标驱动核心学习，进而改变读出的人格
    net, ms = _make_ms()
    before = read_personality(net, ms)
    # 愤怒情绪（负效价 + 高唤醒）→ 强化本我 → 冲动性上升
    emotion_experience(net, ms, valence=-0.8, arousal=0.9, steps=300)
    after = read_personality(net, ms)
    assert after["impulsivity"] > before["impulsivity"]


def test_calm_emotion_reduces_impulsivity():
    net, ms = _make_ms()
    # 先愤怒让冲动性上升
    emotion_experience(net, ms, valence=-0.8, arousal=0.9, steps=200)
    angry = read_personality(net, ms)
    # 再平静让冲动性回落
    emotion_experience(net, ms, valence=0.6, arousal=-0.7, steps=400)
    calm = read_personality(net, ms)
    assert calm["impulsivity"] < angry["impulsivity"]


def test_moral_experience_raises_restraint():
    # 道德/克制经历应让「克制」维度上升（抑制边增强）
    net, ms = _make_ms()
    before = read_personality(net, ms)
    # 持续激活超我核心（道德/克制）
    for _ in range(300):
        inp = np.zeros(600)
        inp[ms.core_masks["superego"]] = 3.0
        net.step(external_input=inp)
    after = read_personality(net, ms)
    assert after["restraint"] > before["restraint"]
