"""LLM judges that are independent of the generator.

* The judge model is a different family from the generator (default Qwen2.5-72B
  judging DeepSeek-V3 output), configured by LLM_MODEL_JUDGE.
* Pairwise comparison: two reports for the same task, judged in both orders to
  cancel position bias. A side wins a dimension only if it wins in both orders;
  otherwise it is a tie. This has far more resolution than absolute 0-1 scores,
  which in our runs clustered at 0.76 regardless of the report.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from agent_runtime.llm import OpenAICompatLLM, load_dotenv, parse_json

DIMS = ["基本面深度", "归因逻辑", "反证质量", "证伪条件明确性", "风险覆盖"]


def judge_llm() -> Optional[OpenAICompatLLM]:
    load_dotenv()
    url, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if not (url and key) or os.environ.get("LLM_OFFLINE") == "1":
        return None
    m = os.environ.get("LLM_MODEL_JUDGE", "Qwen/Qwen2.5-72B-Instruct")
    return OpenAICompatLLM(url, key, m, m)


def _clip(md: str, n: int = 9000) -> str:
    # drop the citation list: it is long and identical in structure for both sides
    return md.split("## 引用")[0][:n]


SYS = ("你是买方研究总监，盲评两份针对同一问题的投资研究报告 A 与 B。逐维度判断哪份更好："
       + "、".join(DIMS) + "。只比较分析质量，不因篇幅长短加分；数值都已经过程序核验，不必核对数字。"
       '只输出JSON {"winners":{"基本面深度":"A|B|tie",...},"overall":"A|B|tie","reason":"不超过80字"}')


def _ask(llm, question: str, a: str, b: str) -> Optional[dict]:
    res = llm.chat(SYS, json.dumps({"question": question, "A": a, "B": b}, ensure_ascii=False), json_mode=True)
    d = parse_json(res.text)
    if not d or not isinstance(d.get("winners"), dict):
        return None
    d["_tokens"] = res.input_tokens + res.output_tokens
    return d


def pairwise(question: str, rep_x: dict, rep_y: dict, llm=None) -> Optional[dict]:
    """Return per-dimension winner in {'x','y','tie'} after judging both orders."""
    llm = llm or judge_llm()
    if llm is None:
        return None
    x, y = _clip(rep_x["markdown"]), _clip(rep_y["markdown"])
    d1 = _ask(llm, question, x, y)   # x is A
    d2 = _ask(llm, question, y, x)   # x is B
    if not d1 or not d2:
        return None
    out = {}
    for dim in DIMS + ["overall"]:
        w1 = (d1["winners"].get(dim) if dim != "overall" else d1.get("overall")) or "tie"
        w2 = (d2["winners"].get(dim) if dim != "overall" else d2.get("overall")) or "tie"
        x_wins = w1 == "A" and w2 == "B"
        y_wins = w1 == "B" and w2 == "A"
        out[dim] = "x" if x_wins else ("y" if y_wins else "tie")
    out["reasons"] = [d1.get("reason", ""), d2.get("reason", "")]
    out["consistent"] = sum(1 for k in DIMS + ["overall"] if out[k] != "tie")
    return out
