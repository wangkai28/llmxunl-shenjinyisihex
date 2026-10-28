from __future__ import annotations

import json
import re
import tomllib

from openai import OpenAI


_PROMPT_TEMPLATE = (
    "请把下面这句话的情绪，量化成两个连续数值（Russell 情绪模型）：\n"
    "- valence 效价：从 -1（很不愉快）到 +1（很愉快）\n"
    "- arousal 唤醒度：从 -1（很平静）到 +1（很激动）\n"
    "只输出一个 JSON 对象，不要任何其他文字、不要 markdown，格式：\n"
    '{{"valence": 0.8, "arousal": 0.7}}\n\n'
    "句子：{text}"
)


def _load_aigw():
    cfg = tomllib.load(open("~/.codex/config.toml", "rb"))
    aigw = cfg["model_providers"]["aigw"]
    return aigw["base_url"], aigw["experimental_bearer_token"]


class LLMAnnotator:
    """用 LLM（aigw 网关）把语句标注成 (valence, arousal) 情绪坐标。"""

    def __init__(self, model: str = "qwen3.7-plus"):
        base_url, api_key = _load_aigw()
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def annotate(self, text: str):
        prompt = _PROMPT_TEMPLATE.format(text=text)
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        raw = r.choices[0].message.content or ""
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str):
        text = raw.strip()
        # 剥离可能的 markdown 围栏
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
            v = float(data["valence"])
            a = float(data["arousal"])
            v = max(-1.0, min(1.0, v))
            a = max(-1.0, min(1.0, a))
            return (v, a)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def annotate_batch(self, texts):
        """批量标注，返回 list[(valence, arousal)]，失败项为 None。"""
        return [self.annotate(t) for t in texts]
