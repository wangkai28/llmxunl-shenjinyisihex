from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Event:
    type: str
    intensity: float = 1.0
    content: str = ""


# 事件类型 → 情境偏置（与 multi_self.set_context 的语义一致）
EVENT_TO_CONTEXT = {
    "danger": {"id": 3.0, "superego": 0.2, "tao": 0.2},
    "moral": {"id": 0.2, "superego": 3.0, "tao": 0.2},
    "calm": {"id": 0.2, "superego": 0.2, "tao": 3.0},
    "praise": {"id": 1.5, "superego": 1.2, "tao": 1.5},
    "criticism": {"id": 1.8, "superego": 2.2, "tao": 1.0},
    "threat": {"id": 3.0, "superego": 0.5, "tao": 0.5},
}


def encode_event(event: Event) -> dict:
    if event.type not in EVENT_TO_CONTEXT:
        raise ValueError(f"unknown event type {event.type!r}")
    base = dict(EVENT_TO_CONTEXT[event.type])
    # 主导自我（值最大者）的偏置随 intensity 缩放
    dominant = max(base, key=base.get)
    base[dominant] = base[dominant] * max(0.0, min(1.0, event.intensity))
    return base
