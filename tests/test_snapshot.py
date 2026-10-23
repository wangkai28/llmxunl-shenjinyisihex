import numpy as np
from bioelectric.spiking import SpikingNetwork
from bioelectric.snapshot import bin_bounds, summarize, to_text


def test_bin_bounds_cover_all_units():
    bounds = bin_bounds(10, 3)
    covered = []
    for start, end in bounds:
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(10))


def test_summarize_counts_active():
    net = SpikingNetwork(n_units=10, seed=0)
    net.spikes = np.array([True, True, False, False, False, False, False, False, False, False])
    summary = summarize(net, n_bins=2)
    assert summary["active_count"] == 2
    assert summary["total_units"] == 10
    assert sum(summary["bin_activity"]) == 2


def test_to_text_contains_numbers():
    text = to_text({"active_count": 2, "total_units": 10, "bin_activity": [2, 0]})
    assert "2" in text
    assert "10" in text
