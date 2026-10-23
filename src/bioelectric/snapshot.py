from __future__ import annotations

import numpy as np

from .spiking import SpikingNetwork


def bin_bounds(n_units: int, n_bins: int) -> list[tuple[int, int]]:
    splits = np.array_split(np.arange(n_units), n_bins)
    return [(int(s[0]), int(s[-1]) + 1) for s in splits]


def summarize(network: SpikingNetwork, n_bins: int = 8) -> dict:
    spikes = network.spikes
    bounds = bin_bounds(len(spikes), n_bins)
    bin_activity = [int(spikes[start:end].sum()) for start, end in bounds]
    return {
        "active_count": int(spikes.sum()),
        "total_units": int(len(spikes)),
        "bin_activity": bin_activity,
    }


def to_text(summary: dict) -> str:
    return (
        f"active_count={summary['active_count']} "
        f"total_units={summary['total_units']} "
        f"bin_activity={summary['bin_activity']}"
    )
