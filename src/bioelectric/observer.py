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
        writeback_duration: int | None = None,
        writeback_gain: float = 1.0,
        writeback_policy: str = "llm",
        error_threshold: float = 0.1,
    ):
        if observe_every < 1:
            raise ValueError("observe_every must be >= 1")
        if mode not in ("readwrite", "readonly"):
            raise ValueError(f"invalid mode {mode!r}; expected 'readwrite' or 'readonly'")
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if writeback_duration is not None and writeback_duration < 1:
            raise ValueError("writeback_duration must be >= 1")
        if writeback_gain <= 0.0:
            raise ValueError("writeback_gain must be > 0")
        if writeback_policy not in ("llm", "error-driven"):
            raise ValueError(
                f"invalid writeback_policy {writeback_policy!r}; "
                "expected 'llm' or 'error-driven'"
            )
        self.network = network
        self.provider = provider
        self.memory = memory if memory is not None else ReflectionMemory()
        self.observe_every = observe_every
        self.n_bins = n_bins
        self.mode = mode
        self.history_window = history_window
        self.writeback_duration = (
            observe_every if writeback_duration is None else writeback_duration
        )
        self.writeback_gain = writeback_gain
        self.writeback_policy = writeback_policy
        self.error_threshold = error_threshold
        self._last_error = None
        self.step_count = 0
        self._active_input = np.zeros(network.n_units, dtype=float)
        self._active_remaining = 0
        self.bounds = bin_bounds(network.n_units, n_bins)

    def tick(self) -> Reflection | None:
        self.network.step(self._active_input)
        if self._active_remaining > 0:
            self._active_remaining -= 1
            if self._active_remaining == 0:
                self._active_input.fill(0.0)
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
        if self.mode == "readwrite":
            if self.writeback_policy == "error-driven":
                if self._last_error is not None and self._last_error > self.error_threshold:
                    self._writeback(reflection)
            elif reflection.action_type == "inject":
                self._writeback(reflection)
        return reflection

    def _writeback(self, reflection: Reflection) -> None:
        region = reflection.region
        if 0 <= region < len(self.bounds):
            start, end = self.bounds[region]
            self._active_input[start:end] += reflection.strength * self.writeback_gain
            self._active_remaining = self.writeback_duration

    def feedback_error(self, error: float | None) -> None:
        """由上层（experiment）在算出预测误差后喂回，供 error-driven 策略使用。"""
        self._last_error = error
