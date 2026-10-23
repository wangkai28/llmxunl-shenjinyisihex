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


def convergence(errors: list[float]) -> dict:
    valid = [float(e) for e in errors if e is not None]
    if len(valid) < 4:
        return {"slope": 0.0, "head_mean": float("nan"),
                "tail_mean": float("nan"), "converged": False}
    x = np.arange(len(valid), dtype=float)
    y = np.array(valid, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    half = len(valid) // 2
    head_mean = float(np.mean(y[:half]))
    tail_mean = float(np.mean(y[half:]))
    converged = bool(slope < 0 and tail_mean < head_mean)
    return {"slope": slope, "head_mean": head_mean,
            "tail_mean": tail_mean, "converged": converged}
