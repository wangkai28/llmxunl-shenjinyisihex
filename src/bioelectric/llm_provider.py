from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reflection:
    self_description: str
    action_type: str
    region: int = 0
    strength: float = 0.0


class LLMProvider:
    def reflect(self, snapshot_text: str, history: list[str]) -> Reflection:
        raise NotImplementedError


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
