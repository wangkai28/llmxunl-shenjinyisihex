from __future__ import annotations

import os

from .balanced_network import BalancedNetwork
from .continuity import save_snapshot, load_snapshot, append_autobiography
from .event import Event, encode_event
from .multi_self import MultiSelf, set_context, dominant_self, measure_multi_self


class IdentityCore:
    def __init__(self, *, persona: str, state_dir: str, seed: int = 0):
        self.persona = persona
        self.state_dir = state_dir
        self.seed = seed
        self._network = BalancedNetwork(
            n_units=600, k_exc=10, j_inh=-4.5,
            external_mean=0.15, external_noise=0.08, delay_max=4, seed=seed,
        )
        self._ms = MultiSelf(
            self._network,
            core_specs={
                "id": {"size": 50, "bias": 3.0},
                "superego": {"size": 50, "bias": 0.5},
                "tao": {"size": 50, "bias": 1.5},
            },
            seed=1, hebb_lr=0.02, cross_inhibition=3.0,
        )
        self._ms.form(steps=200)

    def process_event(self, event: Event) -> dict:
        ctx = encode_event(event)
        set_context(self._ms, ctx)
        # 驱动一段让情境生效
        for _ in range(50):
            dominant = max(ctx, key=ctx.get)
            from .multi_self import _drive
            _drive(self._ms, dominant)
        state = self.query()
        append_autobiography(
            os.path.join(self.state_dir, "autobiography.jsonl"),
            {"event": event.type, "intensity": event.intensity,
             "content": event.content, "dominant": state["dominant"]},
        )
        return state

    def query(self) -> dict:
        dominant = dominant_self(self._ms, steps=50)
        dims = measure_multi_self(self._ms, steps=50)
        names = list(self._ms.core_masks.keys())
        result = {"dominant": dominant}
        for n in names:
            result[n] = dims[n]["activity"]
        return result

    def save(self):
        save_snapshot(self.query(), os.path.join(self.state_dir, "snapshot.json"))

    def load(self):
        # 恢复快照中的主导自我为当前情境
        snap = load_snapshot(os.path.join(self.state_dir, "snapshot.json"))
        if snap.get("dominant"):
            # 按主导自我设一个对应情境
            ctx = {"id": 0.2, "superego": 0.2, "tao": 0.2}
            ctx[snap["dominant"]] = 3.0
            set_context(self._ms, ctx)
            for _ in range(50):
                from .multi_self import _drive
                _drive(self._ms, snap["dominant"])
