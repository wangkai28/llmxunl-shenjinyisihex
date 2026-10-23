import numpy as np

from bioelectric.metrics import bin_rate, convergence, prediction_error
from bioelectric.snapshot import bin_bounds
from bioelectric.spiking import SpikingNetwork


def test_bin_rate():
    net = SpikingNetwork(n_units=4, seed=0)
    net.spikes = np.array([True, True, False, False])
    bounds = bin_bounds(4, 2)
    rates = bin_rate(net, bounds)
    assert rates.shape == (2,)
    assert rates[0] == 1.0
    assert rates[1] == 0.0


def test_prediction_error_mae():
    assert prediction_error([0.2, 0.4], np.array([0.0, 0.4])) == 0.1
    assert prediction_error(None, np.array([0.0, 0.4])) is None


def test_convergence_decreasing_series():
    result = convergence([0.5, 0.4, 0.3, 0.2, 0.1])
    assert result["slope"] < 0
    assert result["converged"] is True


def test_convergence_flat_series_not_converged():
    result = convergence([0.3, 0.3, 0.3, 0.3, 0.3])
    assert result["converged"] is False


def test_convergence_too_short():
    result = convergence([0.5, 0.4])
    assert result["converged"] is False
