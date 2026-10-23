from bioelectric.experiment import ExperimentResult, run_experiment
from bioelectric.llm_provider import MockLLMProvider
from bioelectric.spiking import SpikingNetwork


def test_run_experiment_collects_metrics():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(
        description="stable core",
        action_type="none",
    )
    result = run_experiment(
        net, provider, mode="readwrite", steps=20, observe_every=5, n_bins=2,
    )
    assert isinstance(result, ExperimentResult)
    assert len(result.self_descriptions) == 4  # 观察点：5,10,15,20
    # Mock 不输出预测，因此误差全为 None
    assert all(e is None for e in result.errors)
    assert result.convergence["converged"] is False


def test_run_experiment_readonly_same_shape():
    net = SpikingNetwork(n_units=8, seed=0)
    provider = MockLLMProvider(description="x")
    rw = run_experiment(net, provider, mode="readwrite", steps=10, observe_every=5, n_bins=2)
    net2 = SpikingNetwork(n_units=8, seed=0)
    ro = run_experiment(net2, provider, mode="readonly", steps=10, observe_every=5, n_bins=2)
    assert len(rw.errors) == len(ro.errors)
