from __future__ import annotations

import numpy as np


class SpikingNetwork:
    def __init__(
        self,
        n_units,
        *,
        connectivity=0.1,
        threshold=1.0,
        refractory=2,
        leak=0.1,
        dt=1.0,
        weight_scale=0.5,
        hebb_lr=0.01,
        weight_decay=0.001,
        weight_max=2.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.n_units = int(n_units)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.leak = float(leak)
        self.dt = float(dt)
        self.hebb_lr = float(hebb_lr)
        self.weight_decay = float(weight_decay)
        self.weight_max = float(weight_max)
        self.W = np.zeros((self.n_units, self.n_units), dtype=float)
        if connectivity > 0:
            mask = rng.random((self.n_units, self.n_units)) < connectivity
            self.W[mask] = rng.normal(0.0, weight_scale, size=int(mask.sum()))
        np.fill_diagonal(self.W, 0.0)
        self.V = rng.uniform(0.0, self.threshold, size=self.n_units)
        self.refrac = np.zeros(self.n_units, dtype=int)
        self.spikes = np.zeros(self.n_units, dtype=bool)

    def step(self, external_input=None) -> np.ndarray:
        ext = (
            np.zeros(self.n_units)
            if external_input is None
            else np.asarray(external_input, dtype=float)
        )
        synaptic = self.W @ self.spikes.astype(float)
        self.V = self.V + (synaptic + ext - self.leak * self.V) * self.dt
        fired = (self.V >= self.threshold) & (self.refrac == 0)
        self.spikes = fired
        self.V[fired] = 0.0
        self.refrac[fired] = self.refractory
        not_fired = ~fired
        self.refrac[not_fired] = np.maximum(self.refrac[not_fired] - 1, 0)
        self._hebb_update(fired)
        return fired.copy()

    def _hebb_update(self, fired):
        co = np.outer(fired.astype(float), fired.astype(float))
        np.fill_diagonal(co, 0.0)
        self.W = self.W + self.hebb_lr * co * (self.weight_max - self.W)
        self.W = self.W - self.weight_decay * self.W
        self.W = np.clip(self.W, 0.0, self.weight_max)
        np.fill_diagonal(self.W, 0.0)
