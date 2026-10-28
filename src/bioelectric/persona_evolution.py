from __future__ import annotations


# 每个事件类型对性格参数的演化方向：+1 增加、-1 减少、0 不变
EVOLUTION_DIRECTION = {
    "danger": {"external_mean": +1, "j_inh": +1, "external_noise": +1},
    "threat": {"external_mean": +1, "j_inh": +1, "external_noise": +1},
    "moral": {"external_mean": -1, "j_inh": -1, "external_noise": -1},
    "criticism": {"external_mean": -1, "j_inh": -1, "external_noise": 0},
    "calm": {"external_mean": 0, "j_inh": 0, "external_noise": -1},
    "praise": {"external_mean": 0, "j_inh": 0, "external_noise": -1},
}

# 参数边界（防止演化跑飞）
BOUNDS = {
    "j_inh": (-8.0, -1.5),       # 注意 j_inh 是负值，越负抑制越强
    "external_mean": (0.05, 0.5),
    "external_noise": (0.02, 0.4),
}


def evolve_persona(cfg: dict, event_type: str, *, lr: float = 0.05) -> dict:
    direction = EVOLUTION_DIRECTION.get(event_type)
    if direction is None:
        return dict(cfg)
    new = dict(cfg)
    for key, sign in direction.items():
        if key not in new or sign == 0:
            continue
        delta = lr * abs(new[key]) * sign
        lo, hi = BOUNDS[key]
        new[key] = max(lo, min(hi, new[key] + delta))
    return new
