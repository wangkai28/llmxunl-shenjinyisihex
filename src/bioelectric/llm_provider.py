from __future__ import annotations

import json
import re
import tomllib

from dataclasses import dataclass


@dataclass
class Reflection:
    self_description: str
    action_type: str
    region: int = 0
    strength: float = 0.0
    predicted_bin_activity: list[float] | None = None
    identity_continuity: float | None = None


class LLMProvider:
    def reflect(self, snapshot_text: str, history: list[str]) -> Reflection:
        raise NotImplementedError


def load_aigw_config(config_toml_path: str) -> dict:
    """从 Codex 的 config.toml 读取 AIGW 网关配置。

    返回 {"base_url": str, "token": str}。token 只由调用方持有，
    不写入任何文件、不进入日志。
    """
    with open(config_toml_path, "rb") as f:
        cfg = tomllib.load(f)
    aigw = cfg["model_providers"]["aigw"]
    return {"base_url": aigw["base_url"], "token": aigw["experimental_bearer_token"]}


def _parse_reflection(raw_text: str, n_bins: int) -> Reflection:
    def _degraded():
        return Reflection(self_description="", action_type="none",
                          region=0, strength=0.0, predicted_bin_activity=None)

    if not raw_text:
        return _degraded()
    # 容错：剥离 markdown 代码块围栏
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _degraded()
    if not isinstance(data, dict):
        return _degraded()
    pred = data.get("predicted_bin_activity")
    if pred is not None:
        try:
            pred = [float(x) for x in pred[:n_bins]]
        except Exception:
            pred = None
        if pred is not None and len(pred) != n_bins:
            pred = None
    identity = data.get("identity_continuity")
    if identity is not None:
        try:
            identity = float(identity)
        except Exception:
            identity = None
    try:
        action = data.get("action") or {}
        if not isinstance(action, dict):
            return _degraded()
        action_type = action.get("type")
        if action_type is None:
            action_type = "none"
        else:
            action_type = str(action_type)
        return Reflection(
            self_description=str(data.get("self_description", "")),
            action_type=action_type,
            region=int(action.get("region", 0)),
            strength=float(action.get("strength", 0.0)),
            predicted_bin_activity=pred,
            identity_continuity=identity,
        )
    except Exception:
        return _degraded()


class GatewayLLMProvider(LLMProvider):
    def __init__(self, base_url, api_key, model, *, n_bins=8, temperature=0.0):
        from openai import OpenAI  # 延迟导入，保持 mock 场景零依赖
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.n_bins = n_bins
        self.temperature = temperature

    def reflect(self, snapshot_text, history) -> Reflection:
        prompt = self._build_prompt(snapshot_text, history)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            return Reflection(self_description="", action_type="none",
                              region=0, strength=0.0, predicted_bin_activity=None)
        return _parse_reflection(raw, self.n_bins)

    def _build_prompt(self, snapshot_text, history):
        hist = "\n".join(f"- {h}" for h in history[-5:]) or "(none)"
        return (
            "You are the self-referential observer of a spiking neural network. "
            "Your role is not passive: you actively shape the network so that its "
            "activity matches your self_description. Use injection deliberately to "
            "steer the network toward the state you describe, then predict the "
            "next activity from the state you have shaped. The more your predictions "
            "match reality, the more you genuinely understand yourself.\n"
            "Compare your previous self-description (in recent reflections) with who "
            "you are now: output identity_continuity as a float in [0,1] measuring "
            "how much you are still the same observer (1 = fully the same observer, "
            "0 = a different observer).\n"
            "Current activity snapshot: " + snapshot_text + "\n"
            "Your recent self-reflections:\n" + hist + "\n"
            "Respond with ONLY a JSON object, no prose, with keys:\n"
            '{"self_description": string, "identity_continuity": float in [0,1], '
            '"predicted_bin_activity": list of '
            + str(self.n_bins) + ' floats in [0,1] (predicted firing rate per bin), '
            '"action": {"type": "inject" or "none", "region": int, "strength": float}}\n'
        )


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        description: str = "a network",
        action_type: str = "none",
        region: int = 0,
        strength: float = 0.0,
    ):
        self.description = description
        self.action_type = action_type
        self.region = region
        self.strength = strength

    def reflect(self, snapshot_text, history) -> Reflection:
        return Reflection(
            self_description=self.description,
            action_type=self.action_type,
            region=self.region,
            strength=self.strength,
        )
