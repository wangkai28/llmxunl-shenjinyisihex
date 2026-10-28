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
