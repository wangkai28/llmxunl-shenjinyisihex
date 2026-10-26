import numpy as np

from bioelectric.metrics import bin_rate, convergence, prediction_error
from bioelectric.metrics import narrative_coherence
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


def test_prediction_error_shape_mismatch_returns_none():
    assert prediction_error([0.1, 0.2], np.array([0.1, 0.2, 0.3])) is None


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


def test_convergence_has_significance_fields():
    # 明显下降的序列应有负斜率、低 p 值、且收敛
    result = convergence([0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15])
    assert result["slope"] < 0
    assert "p_value" in result
    assert "slope_ci" in result
    assert result["converged"] is True


def test_convergence_noisy_flat_not_significant():
    # 无趋势的抖动序列不应被误判为收敛
    result = convergence([0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.2])
    assert result["converged"] is False


def test_convergence_deterministic_given_seed():
    errors = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
    a = convergence(errors, seed=42)
    b = convergence(errors, seed=42)
    assert a["p_value"] == b["p_value"]


def test_narrative_coherence_high_and_stable():
    # 身份连续性始终高且稳定 → 高叙事一致性
    result = narrative_coherence([0.95, 0.92, 0.94, 0.93, 0.95])
    assert result["mean"] > 0.9
    assert result["std"] < 0.1


def test_narrative_coherence_low_or_fluctuating():
    # 身份连续性低、或剧烈波动 → 低叙事一致性
    result = narrative_coherence([0.3, 0.8, 0.2, 0.9, 0.4])
    assert result["mean"] < 0.9


def test_narrative_coherence_ignores_none():
    result = narrative_coherence([0.9, None, 0.91, None, 0.92])
    assert result["mean"] > 0.9
