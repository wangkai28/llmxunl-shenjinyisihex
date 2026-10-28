from __future__ import annotations

from .event import Event


# 规则版文本编码：关键词 → 事件类型（可插拔，后续可换 LLM）
_KEYWORD_RULES = [
    (("危险", "威胁", "攻击", "恐惧", "害怕"), "danger"),
    (("道德", "良心", "应该", "责任", "愧疚"), "moral"),
    (("批评", "指责", "错误", "失败"), "criticism"),
    (("表扬", "夸", "赞", "优秀", "成功"), "praise"),
    (("安静", "平静", "放松", "自然", "休息"), "calm"),
]


def _text_to_type(text: str) -> str:
    for keywords, event_type in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return event_type
    return "calm"  # 默认平静


def encode_input(raw) -> Event:
    """把外界输入编码成 Event（可插拔入口）。

    接受三种形态：
    - Event：直通
    - dict：{"type": str, "intensity": float, "content": str}
    - str：规则版文本 → 事件类型（关键词匹配）
    """
    if isinstance(raw, Event):
        return raw
    if isinstance(raw, dict):
        return Event(
            type=raw["type"],
            intensity=float(raw.get("intensity", 1.0)),
            content=str(raw.get("content", "")),
        )
    if isinstance(raw, str):
        return Event(type=_text_to_type(raw), intensity=1.0, content=raw)
    raise TypeError(f"cannot encode input of type {type(raw).__name__}")


def decode_state(state: dict) -> dict:
    """把意识状态解码为外界可读的结构化 dict（可插拔，可换自然语言版）。"""
    return dict(state)


def emotion_to_context(valence: float, arousal: float) -> dict:
    """把情绪坐标 (valence, arousal) 映射到多自我情境 bias。

    用学到的情绪坐标替代关键词规则，更语义化、更细腻地驱动自我层：
    - 高唤醒（激动）→ 本我主导（应激/兴奋外放）
    - 正效价 + 低唤醒（平静满足）→ 道家主导（从容）
    - 低唤醒 + 负效价（低落）→ 本我主导（压抑）
    超我在纯情绪中体现较弱，保持中等。
    """
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    intensity = 0.5 + 0.5 * abs(arousal)  # 唤醒度越高，主导越强

    if arousal >= 0.3:  # 高唤醒：激动/应激 → 本我主导
        dominant = "id"
    elif valence >= 0.0:  # 正效价 + 低唤醒：平静满足 → 道家
        dominant = "tao"
    else:  # 负效价 + 低唤醒：低落压抑 → 本我
        dominant = "id"

    # 生成 bias：主导自我按 intensity 定高，其余按中等
    ctx = {"id": 0.5, "superego": 0.5, "tao": 0.5}
    for key in ctx:
        if key == dominant:
            ctx[key] = 0.5 + 2.5 * intensity  # 主导偏置
        else:
            ctx[key] = 0.5 * (1.0 - 0.3 * intensity)  # 非主导略降
    return ctx
