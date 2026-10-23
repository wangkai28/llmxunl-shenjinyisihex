from __future__ import annotations

import numpy as np

from .llm_provider import LLMProvider, Reflection
from .memory import ReflectionMemory
from .snapshot import bin_bounds, summarize, to_text
from .spiking import SpikingNetwork


class Observer:
    def __init__(
        self,
        network: SpikingNetwork,
        provider: LLMProvider,
        memory: ReflectionMemory | None = None,
        *,
        observe_every: int = 10,
        n_bins: int = 8,
        mode: str = "readwrite",
        history_window: int = 5,
    ):
        self.network = network
        self.provider = provider
        self.memory = memory if memory is not None else ReflectionMemory()
        self.observe_every = observe_every
        self.n_bins = n_bins
        self.mode = mode
        self.history_window = history_window
        self.step_count = 0
        self.pending_input = np.zeros(network.n_units, dtype=float)
        self.bounds = bin_bounds(network.n_units, n_bins)

    def tick(self) -> Reflection | None:
        self.network.step(self.pending_input)
        self.pending_input.fill(0.0)
        self.step_count += 1
        if self.step_count % self.observe_every == 0:
            return self._observe()
        return None

    def run(self, steps: int) -> list[Reflection]:
        reflections = []
        for _ in range(steps):
            result = self.tick()
            if result is not None:
                reflections.append(result)
        return reflections

    def _observe(self) -> Reflection:
        summary = summarize(self.network, self.n_bins)
        text = to_text(summary)
        history = self.memory.recent(self.history_window)
        reflection = self.provider.reflect(text, history)
        self.memory.add(reflection.self_description)
        if self.mode == "readwrite" and reflection.action_type == "inject":
            self._writeback(reflection)
        return reflection

    def _writeback(self, reflection: Reflection) -> None:
        region = reflection.region
        if 0 <= region < len(self.bounds):
            start, end = self.bounds[region]
            self.pending_input[start:end] += reflection.strength
