"""LLM gateway. One interface, two backends:

* OpenAICompatLLM: any OpenAI-compatible endpoint (DeepSeek, Qwen/DashScope,
  Moonshot, vLLM, OpenAI). Configured by LLM_BASE_URL / LLM_API_KEY /
  LLM_MODEL_STRONG / LLM_MODEL_LIGHT.
* OfflineLLM: no network. `online` is False and every agent falls back to its
  deterministic template path. Prompts are still built so token accounting and
  context-isolation measurements stay meaningful.

LLMs never compute financial numbers in either mode.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

PRICE_PER_1K = {"strong": (0.0005, 0.002), "light": (0.0001, 0.0004)}  # USD in/out, configurable estimate


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cjk / 1.5 + (len(text) - cjk) / 4) + 1


@dataclass
class LLMResult:
    text: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class BaseLLM:
    online = False

    def chat(self, system: str, user: str, tier: str = "strong", json_mode: bool = False) -> LLMResult:
        raise NotImplementedError


class OfflineLLM(BaseLLM):
    online = False

    def chat(self, system: str, user: str, tier: str = "strong", json_mode: bool = False) -> LLMResult:
        return LLMResult(None, estimate_tokens(system + user), 0, 0.0, f"offline-{tier}")


class OpenAICompatLLM(BaseLLM):
    online = True

    def __init__(self, base_url: str, api_key: str, strong: str, light: str, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.models = {"strong": strong, "light": light}
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, system: str, user: str, tier: str = "strong", json_mode: bool = False) -> LLMResult:
        body: dict[str, Any] = {"model": self.models[tier], "temperature": 0.1,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {})
                text = data["choices"][0]["message"]["content"]
                return LLMResult(text, usage.get("prompt_tokens", estimate_tokens(system + user)),
                                 usage.get("completion_tokens", estimate_tokens(text)),
                                 (time.perf_counter() - t0) * 1000, self.models[tier])
            except (httpx.HTTPError, KeyError, ValueError):
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
        return LLMResult(None, estimate_tokens(system + user), 0, (time.perf_counter() - t0) * 1000, self.models[tier])


def parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def get_llm() -> BaseLLM:
    url, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if url and key and os.environ.get("LLM_OFFLINE", "0") != "1":
        return OpenAICompatLLM(url, key, os.environ.get("LLM_MODEL_STRONG", "deepseek-chat"),
                               os.environ.get("LLM_MODEL_LIGHT", os.environ.get("LLM_MODEL_STRONG", "deepseek-chat")))
    return OfflineLLM()


def cost_usd(tier: str, inp: int, out: int) -> float:
    pi, po = PRICE_PER_1K.get(tier, PRICE_PER_1K["strong"])
    return inp / 1000 * pi + out / 1000 * po


def call_json(llm: BaseLLM, tracer, node: str, system: str, user: str, tier: str,
              validate: Callable[[dict], bool] = lambda d: True) -> Optional[dict]:
    """Ask for JSON; return None when offline, on failure, or when validation fails."""
    res = llm.chat(system, user, tier=tier, json_mode=True)
    tracer.llm(node, res, tier)
    data = parse_json(res.text)
    if data is not None and validate(data):
        return data
    return None
