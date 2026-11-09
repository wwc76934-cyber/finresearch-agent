"""Check the LLM gateway configured in .env.

  python scripts/check_llm.py          # one JSON-mode test call per tier
  python scripts/check_llm.py --list   # list model ids the key can use
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

import httpx  # noqa: E402

from agent_runtime.llm import get_llm, load_dotenv, parse_json  # noqa: E402

if __name__ == "__main__":
    load_dotenv()
    url, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if not (url and key):
        sys.exit("LLM_BASE_URL / LLM_API_KEY not set. Copy .env.example to .env and fill in LLM_API_KEY.")
    if "--list" in sys.argv:
        r = httpx.get(f"{url.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        for m in sorted(x["id"] for x in r.json()["data"]):
            print(m)
        sys.exit(0)
    llm = get_llm()
    print(f"backend={type(llm).__name__} models={getattr(llm, 'models', None)}")
    for tier in ("strong", "light"):
        res = llm.chat("只输出JSON。", '返回 {"ok": true, "model": "<你的模型名>"}', tier=tier, json_mode=True)
        data = parse_json(res.text)
        status = "OK" if data else "FAILED"
        print(f"[{tier}] {status} {res.model} in={res.input_tokens} out={res.output_tokens} "
              f"{res.latency_ms:.0f}ms -> {json.dumps(data, ensure_ascii=False) if data else res.text}")
