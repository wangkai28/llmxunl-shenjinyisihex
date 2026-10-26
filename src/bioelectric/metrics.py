from __future__ import annotations

import numpy as np

from .snapshot import bin_bounds
from .spiking import SpikingNetwork


def bin_rate(network: SpikingNetwork, bounds) -> np.ndarray:
    spikes = network.spikes
    return np.array([
        spikes[start:end].mean() if end > start else 0.0
        for start, end in bounds
    ], dtype=float)


def prediction_error(predicted, actual) -> float | None:
    if predicted is None:
        return None
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        return None
    return float(np.mean(np.abs(p - a)))


def convergence(errors: list[float], seed: int = 0) -> dict:
    valid = [float(e) for e in errors if e is not None]
    if len(valid) < 4:
        return {
            "slope": 0.0, "head_mean": float("nan"), "tail_mean": float("nan"),
            "p_value": float("nan"), "slope_ci": [float("nan"), float("nan")],
            "converged": False,
        }
    x = np.arange(len(valid), dtype=float)
    y = np.array(valid, dtype=float)
    slope = _least_squares_slope(x, y)
    half = len(valid) // 2
    head_mean = float(np.mean(y[:half]))
    tail_mean = float(np.mean(y[half:]))

    # Bootstrap 显著性：对误差序列重采样，估计「斜率」的抽样分布。
    # 收敛判定 = 斜率显著为负（单侧 p < 0.05）且尾段均值低于头段均值。
    rng = np.random.default_rng(seed)
    n_boot = 2000
    slopes = np.empty(n_boot, dtype=float)
    xs = np.arange(len(valid), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(valid), size=len(valid))
        ys = y[idx]
        # 配对重采样：x 与 y 用同一 idx 一起采样，保持配对关系。
        slopes[i] = _least_squares_slope(xs[idx], ys)
    # 单侧 p 值：重采样斜率 >= 0 的比例（越小越说明「下降」是稳定的）
    p_value = float(np.mean(slopes >= 0.0))
    ci_low = float(np.percentile(slopes, 2.5))
    ci_high = float(np.percentile(slopes, 97.5))
    significant = p_value < 0.05
    converged = bool(significant and tail_mean < head_mean)
    return {
        "slope": slope,
        "head_mean": head_mean,
        "tail_mean": tail_mean,
        "p_value": p_value,
        "slope_ci": [ci_low, ci_high],
        "converged": converged,
    }


def _least_squares_slope(x: np.ndarray, y: np.ndarray) -> float:
    """一元线性回归斜率的一元闭式解，避免 polyfit 的病态警告。"""
    x_mean = x.mean()
    y_mean = y.mean()
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator == 0.0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denominator)


def narrative_coherence(identity_continuities: list[float | None]) -> dict:
    """度量「自我叙事一致性」：观察者身份连续性的均值与稳定性。

    identity_continuity 是 LLM 每次反思时自评的「我还是不是同一个我」，
    范围 [0,1]。均值高且标准差小 → 观察者始终认为自己是同一个我（高一致性）。
    """
    valid = [float(v) for v in identity_continuities if v is not None]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"), "coherent": False}
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    # 一致性 = 均值高（始终是同一个我）且波动小（身份稳定）
    coherent = bool(mean >= 0.8 and std <= 0.15)
    return {"mean": mean, "std": std, "coherent": coherent}
