from __future__ import annotations


SELF_LABELS = {"id": "本我", "superego": "超我", "tao": "道家"}

SELF_EMOTION = {
    "id": "亢奋、冲动、欲望驱动",
    "superego": "克制、严肃、道德自律",
    "tao": "平和、从容、顺应自然",
}

EVENT_LABELS = {
    "danger": "危险",
    "threat": "威胁",
    "moral": "道德困境",
    "criticism": "批评",
    "praise": "表扬",
    "calm": "平静",
}


def _describe_ocean(ocean: dict) -> str:
    """把 OCEAN 数值转成简短文字描述。"""
    parts = []
    o, e, c, a, n = ocean.get("O"), ocean.get("E"), ocean.get("C"), ocean.get("A"), ocean.get("N")

    def level(v, low_label, high_label):
        return low_label if v < 0.4 else (high_label if v > 0.6 else "适中")

    parts.append(f"开放性{level(o, '偏低', '偏高')}")
    parts.append(f"外向性{level(e, '偏低', '偏高')}")
    parts.append(f"尽责性{level(c, '偏低', '偏高')}")
    parts.append(f"宜人性{level(a, '偏低', '偏高')}")
    parts.append(f"神经质{level(n, '偏低', '偏高')}")
    return "、".join(parts)


def _describe_history(history: list[dict], limit: int = 3) -> str:
    """把最近经历简述为文字。"""
    if not history:
        return "还没有值得记住的经历"
    recent = history[-limit:]
    items = []
    for entry in recent:
        ev = EVENT_LABELS.get(entry.get("event", ""), entry.get("event", "?"))
        dom = SELF_LABELS.get(entry.get("dominant", ""), "")
        items.append(f"{ev}{('(' + dom + '主导)') if dom else ''}")
    return "、".join(items)


def render_self_reflection(state: dict, ocean: dict, history: list[dict]) -> str:
    """生成一段自我审视的文字描述。"""
    dominant = state.get("dominant", "tao")
    dom_label = SELF_LABELS.get(dominant, dominant)
    emotion = SELF_EMOTION.get(dominant, "")

    # 各自我活动率
    acts = []
    for key, label in SELF_LABELS.items():
        v = state.get(key, 0.0)
        acts.append(f"{label}{v:.2f}")

    lines = [
        f"【自我审视】我当前由{dom_label}主导，情绪倾向：{emotion}。",
        f"我的内心活动——" + "、".join(acts) + "。",
        f"我的稳定人格底色——" + _describe_ocean(ocean) + "。",
        f"我最近的经历——" + _describe_history(history) + "。",
    ]
    return "\n".join(lines)
