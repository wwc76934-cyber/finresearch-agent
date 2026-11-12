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
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

# 元 per 1M tokens (input, output). Defaults are placeholders; override in .env with your provider's prices.
PRICE_PER_M = {"strong": (2.0, 8.0), "light": (0.0, 0.0)}


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

    fatal_error: Optional[str] = None

    def chat(self, system: str, user: str, tier: str = "strong", json_mode: bool = False) -> LLMResult:
        if self.fatal_error:  # fail fast once the provider has said no
            return LLMResult(None, estimate_tokens(system + user), 0, 0.0, self.models[tier])
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
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code in (401, 402, 403):  # bad key / no balance / forbidden: retrying cannot help
                    self.fatal_error = f"HTTP {code}: {exc.response.text[:200]}"
                    break
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
            except (httpx.HTTPError, KeyError, ValueError):
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
        return LLMResult(None, estimate_tokens(system + user), 0, (time.perf_counter() - t0) * 1000, self.models[tier])


def parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models (DeepSeek-R1 / Qwen3)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader. Real environment variables always win over the file."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v


def get_llm() -> BaseLLM:
    load_dotenv()
    url, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if url and key and os.environ.get("LLM_OFFLINE", "0") != "1":
        return OpenAICompatLLM(url, key, os.environ.get("LLM_MODEL_STRONG", "deepseek-chat"),
                               os.environ.get("LLM_MODEL_LIGHT", os.environ.get("LLM_MODEL_STRONG", "deepseek-chat")))
    return OfflineLLM()


def cost_cny(tier: str, inp: int, out: int) -> float:
    """Price in CNY. Set LLM_PRICE_{STRONG|LIGHT}_{IN|OUT} (元 per 1M tokens) from your provider's price page."""
    pi = float(os.environ.get(f"LLM_PRICE_{tier.upper()}_IN", PRICE_PER_M[tier][0]))
    po = float(os.environ.get(f"LLM_PRICE_{tier.upper()}_OUT", PRICE_PER_M[tier][1]))
    return inp / 1e6 * pi + out / 1e6 * po


def run_cost_cny(spans: list[dict]) -> float:
    """Only real, successful API calls are billed; context-size estimates are not."""
    return sum(cost_cny(c["tier"], c["in"], c["out"]) for s in spans for c in s.get("llm_calls", [])
               if c.get("online") and c.get("model") != "context")


def run_parallel(fns: list) -> list:
    """Run independent callables (LLM calls) concurrently, results in input order. Each runs in a copy of the
    caller's context so trace spans still attach to the right node. Exceptions are returned, not raised."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    if len(fns) <= 1:
        out = []
        for f in fns:
            try:
                out.append(f())
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
        return out

    def wrap(f):
        ctx = contextvars.copy_context()

        def go():
            try:
                return ctx.run(f)
            except Exception as exc:  # noqa: BLE001
                return exc
        return go

    with ThreadPoolExecutor(max_workers=min(len(fns), 6)) as ex:
        return list(ex.map(lambda g: g(), [wrap(f) for f in fns]))


def call_json(llm: BaseLLM, tracer, node: str, system: str, user: str, tier: str,
              validate: Callable[[dict], bool] = lambda d: True) -> Optional[dict]:
    """Ask for JSON; return None when offline, on failure, or when validation fails."""
    res = llm.chat(system, user, tier=tier, json_mode=True)
    tracer.llm(node, res, tier)
    data = parse_json(res.text)
    if data is not None and validate(data):
        return data
    return None
