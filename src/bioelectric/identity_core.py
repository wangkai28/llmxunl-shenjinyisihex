from __future__ import annotations

import os

from .balanced_network import BalancedNetwork
from .continuity import save_snapshot, load_snapshot, append_autobiography, query_history
from .event import Event, encode_event
from .multi_self import MultiSelf, set_context, _drive
from .ocean import PERSONA_OCEAN, ocean_to_params, evolve_ocean
from .codec import encode_input, decode_state


def _query_state(ms: MultiSelf, steps: int):
    """按当前 context_bias 驱动网络，返回 (主导自我, 各自我活动率)。

    主导自我与活动率来自同一次情境驱动：主导 = argmax(context_bias)，
    活动率 = 该驱动过程中各核心区发放率的累积。不调用 measure_multi_self
    （它用 external_input=None 的自由演化，会丢失当前情境）。
    """
    names = list(ms.core_masks.keys())
    dominant = max(names, key=lambda n: ms.context_bias[n])
    sums = {n: 0 for n in names}
    net = ms.network
    for _ in range(steps):
        _drive(ms, dominant)
        for n in names:
            sums[n] += int(net.spikes[ms.core_masks[n]].sum())
    activity = {n: sums[n] / (steps * ms.core_sizes[n]) for n in names}
    return dominant, activity

class IdentityCore:
    def __init__(self, *, persona: str, state_dir: str, seed: int = 0,
                 significant_threshold: float = 0.8):
        self.persona = persona
        self.state_dir = state_dir
        self.seed = seed
        self.significant_threshold = significant_threshold
        # 性格参数初始值由 OCEAN 派生（覆盖 PERSONAS 的硬编码）
        if persona not in PERSONA_OCEAN:
            raise ValueError(f"unknown persona {persona!r}")
        self._ocean = dict(PERSONA_OCEAN[persona])
        self._persona_cfg = ocean_to_params(self._ocean)
        self._network = BalancedNetwork(
            n_units=600, k_exc=10, j_inh=self._persona_cfg["j_inh"],
            external_mean=self._persona_cfg["external_mean"],
            external_noise=self._persona_cfg["external_noise"],
            delay_max=4, seed=seed,
        )
        self._ms = MultiSelf(
            self._network,
            core_specs={
                "id": {"size": 50, "bias": 3.0},
                "superego": {"size": 50, "bias": 0.5},
                "tao": {"size": 50, "bias": 1.5},
            },
            seed=1, hebb_lr=0.02, cross_inhibition=0.1,
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
        significant = event.intensity >= self.significant_threshold
        if significant:
            self._ocean = evolve_ocean(self._ocean, event.type)
            self._persona_cfg = ocean_to_params(self._ocean)
            self._apply_persona_to_network()
        append_autobiography(
            os.path.join(self.state_dir, "autobiography.jsonl"),
            {"event": event.type, "intensity": event.intensity,
             "content": event.content, "dominant": state["dominant"],
             "significant": significant},
        )
        return state

    def interact(self, raw) -> dict:
        """交互端口：输入外界信息（文本/事件/信号），输出解码后的意识状态。"""
        event = encode_input(raw)
        state = self.process_event(event)
        return decode_state(state)

    def _apply_persona_to_network(self):
        """把演化的性格参数即时应用到活网络（不丢 Hebb 状态）。"""
        self._network.external_mean = self._persona_cfg["external_mean"]
        self._network.external_noise = self._persona_cfg["external_noise"]
        self._network.set_j_inh(self._persona_cfg["j_inh"])

    def query(self) -> dict:
        dominant, activity = _query_state(self._ms, steps=50)
        result = {"dominant": dominant}
        result.update(activity)
        result["ocean"] = dict(self._ocean)
        return result

    def history(self, limit=None):
        return query_history(os.path.join(self.state_dir, "autobiography.jsonl"), limit)

    def save(self):
        state = self.query()
        snapshot = {
            "dominant": state["dominant"],
            "context_bias": dict(self._ms.context_bias),
            "activities": {k: state[k] for k in self._ms.core_masks.keys()},
            "persona_cfg": self._persona_cfg,
            "ocean": self._ocean,
        }
        save_snapshot(snapshot, os.path.join(self.state_dir, "snapshot.json"))

    def load(self):
        # 从快照恢复完整情境偏置（含主导自我），真实还原身份。
        snap = load_snapshot(os.path.join(self.state_dir, "snapshot.json"))
        if not snap:
            return
        ctx = snap.get("context_bias")
        if not ctx:
            return
        set_context(self._ms, ctx)
        if snap.get("ocean"):
            self._ocean = dict(snap["ocean"])
        if snap.get("persona_cfg"):
            self._persona_cfg = snap["persona_cfg"]
        else:
            self._persona_cfg = ocean_to_params(self._ocean)
        self._apply_persona_to_network()
