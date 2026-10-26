from __future__ import annotations

from dataclasses import dataclass, field

from .llm_provider import LLMProvider
from .metrics import bin_rate, convergence, prediction_error
from .observer import Observer
from .snapshot import bin_bounds
from .spiking import SpikingNetwork


@dataclass
class ExperimentResult:
    errors: list = field(default_factory=list)
    self_descriptions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    convergence: dict = field(default_factory=dict)


def run_experiment(
    network: SpikingNetwork,
    provider: LLMProvider,
    *,
    mode: str,
    steps: int,
    observe_every: int,
    n_bins: int,
) -> ExperimentResult:
    observer = Observer(
        network, provider, observe_every=observe_every,
        n_bins=n_bins, mode=mode,
    )
    bounds = bin_bounds(network.n_units, n_bins)
    result = ExperimentResult()
    pending_prediction = None
    for _ in range(steps):
        reflection = observer.tick()
        if reflection is None:
            continue
        actual = bin_rate(network, bounds)
        if pending_prediction is not None:
            result.errors.append(prediction_error(pending_prediction, actual))
        result.self_descriptions.append(reflection.self_description)
        result.actions.append({
            "type": reflection.action_type,
            "region": reflection.region,
            "strength": reflection.strength,
        })
        pending_prediction = reflection.predicted_bin_activity
    result.convergence = convergence(result.errors)
    return result
