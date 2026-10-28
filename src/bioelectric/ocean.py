from __future__ import annotations


DEFAULT_OCEAN = {
    "O": 0.5,  # 开放性
    "E": 0.5,  # 外向性
    "C": 0.5,  # 尽责性
    "A": 0.5,  # 宜人性
    "N": 0.5,  # 神经质
}


def ocean_to_params(ocean: dict) -> dict:
    E = float(ocean.get("E", 0.5))
    N = float(ocean.get("N", 0.5))
    C = float(ocean.get("C", 0.5))
    return {
        "external_mean": 0.05 + 0.45 * E,
        "external_noise": 0.02 + 0.38 * N,
        "j_inh": -(1.5 + 6.5 * C),
    }


# 事件类型 → OCEAN 各特质演化方向（+1 增 / -1 减 / 0 不变）
OCEAN_EVOLUTION_DIRECTION = {
    "danger": {"N": +1, "C": -1, "E": +1, "O": 0, "A": 0},
    "threat": {"N": +1, "C": -1, "E": +1, "O": 0, "A": 0},
    "moral": {"N": -1, "C": +1, "E": 0, "O": 0, "A": 0},
    "criticism": {"N": +1, "C": +1, "E": 0, "O": 0, "A": 0},
    "calm": {"N": -1, "C": 0, "E": 0, "O": 0, "A": 0},
    "praise": {"N": -1, "C": 0, "E": +1, "O": 0, "A": 0},
}


def evolve_ocean(ocean: dict, event_type: str, *, lr: float = 0.005) -> dict:
    direction = OCEAN_EVOLUTION_DIRECTION.get(event_type)
    new = dict(ocean)
    if direction is None:
        return new
    for key, sign in direction.items():
        if key not in new or sign == 0:
            continue
        delta = lr * sign
        new[key] = max(0.0, min(1.0, new[key] + delta))
    return new
