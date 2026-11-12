# Step 4: faster and cheaper B4 (v1.2 PRD)

All numbers come from `reports/eval/online_s4_summary.json` (`python scripts/s4_summary.py`). Same 75 online tasks as Step 3
(golden 30, blind v1 30, multidoc 15). The sealed blind v2 set was not touched.

## What changed

| Change | Where | Effect |
|---|---|---|
| ComplexityRouter | `agents/router.py`, `planner.py`, `graph.py`, `collectors.py`, `verifier.py`, `report.py` | Decides `fast` / `standard` / `full` from the question text, requested mode and rule-parsed asks (no ticker, task id or answer is read). |
| `fast` route | lookups and refusals | Rule-parsed plan (no planner LLM), no re-planning, debate, risk matrix, semantic probe, cross-check or LLM judge; no LLM executive summary. Tier-1, PIT checks and the deterministic fallacy checks still run on every claim. |
| `standard` route | single-topic anomaly / notes questions | Keeps collection, re-planning, probe and the full verifier. Drops debate and risk matrix. Rule-parsed plan. |
| `full` route | valuation, bull/bear, peers, open-ended research, and anything not recognised | Unchanged pipeline. Unrecognised wording defaults here. |
| Parallel verifier | `verifier.tier2_batch`, `llm.run_parallel` | Claims judged in batches of 6, batches concurrent; reason limited to 20 characters. |
| Parallel debate | `debate._llm_rounds` | Bull round 1 and Bear's search queries run together (Bear sees the existing bull points, not the new ones). |
| Subgraph recap | `collectors.f_summarize` | Joined from the first claims; the model call (3-5 s on every subgraph's critical path) is off unless `LLM_SUBGRAPH_SUMMARY=1`. No claim depends on the recap. |
| Table fallback | `collectors.explain_anomaly` | The quoted table rows are data rows, not header rows. |
| Config | `ArchConfig.complexity_router`, arch `B4-no_router` | Ablation switch. |

Fact Lock: already enforced in `debate._accept_args` (arguments with digits or without a valid PIT-safe evidence id are
rejected; numbers are attached by code from the cited MetricRecords). `tests/test_fact_lock.py` now pins this with a scripted
model that tries to write a number and cite a made-up evidence id.

Not done from the PRD list: a fast/reasoning model split. Subgraph recaps moved to the light
model first, then were removed. The verifier and debate still use the strong model.

## Route mix on the 75 tasks (rules only, offline)

| Route | Tasks |
|---|---|
| fast | 18 (12 blind metric, 6 refusal) |
| standard | 26 |
| full | 31 |

The `OPEN_MARKERS` / `STANDARD_MARKERS` word lists were written after reading the golden and blind v1 questions (dev data).
Blind v2 questions were not read.

## Online result (B4, 75 tasks)

| | Before (Step 3, r2+r3) | After (1 run) |
|---|---|---|
| Pass rate | 88.0% | 89.3% (67/75) |
| P50 latency, 75 tasks | 114.5 s | **42.8 s** |
| P90 latency | 135.7 s | 102.5 s |
| P50 latency, golden + blind (60 tasks) | 113.3 s | 62.0 s |
| Cost, 75 tasks | ¥5.31 | ¥2.88 |
| Cost, golden + blind (60 tasks) | ¥4.06 | ¥2.33 |

PRD targets: P50 ≤ 38 s and ¥1.50 per 60 tasks. **Not met.** The 75-task P50 is 4.8 s over, the 60-task P50 24 s over, the 60-task
cost ¥0.83 over.

| Category | n | Pass before | Pass after | P50 before | P50 after |
|---|---|---|---|---|---|
| blind metric (fast) | 12 | 100% | 100% | 96.6 s | 0.2 s |
| blind refusal (fast) | 7 | 100% | 100% | 109.9 s | 0.1 s |
| golden anomaly (standard) | 10 | 80% | 70% | 127.3 s | 27.9 s |
| multidoc (standard) | 15 | 86.7% | 100% | 117.8 s | 37.4 s |
| golden deep research (full) | 10 | 100% | 100% | 126.5 s | 92.6 s |
| golden events/peers (full) | 10 | 100% | 100% | 104.3 s | 80.9 s |
| blind deep research (full) | 5 | 100% | 100% | 118.3 s | 132.7 s |
| blind anomaly | 6 | 16.7% | 16.7% | 113.8 s | 100.5 s |

Reading:
* Fast-path hit 18 of 75 tasks and made zero LLM calls (P50 0.1-0.2 s). Fast-path P50 target of 18 s is met.
* The full route is the bottleneck: 92 s, 81 s and 133 s. Its latency is bound by serial LLM calls (planner, debate rounds,
  manager, verifier passes). Parallel batches cut the verifier and one debate stage but not the chain.
* One run only. The multidoc +13pp and golden anomaly -10pp are one or two tasks each, inside run-to-run noise
  (Step 3 B4 had std 0.0 on pass rate, but with n=3 the range was 79-94% on the Wilson interval).

## Known costs of the change

* `standard` drops debate and risk, so those reports have no bull/bear section and no risk matrix.
* The fast route judges document claims with the offline n-gram heuristic instead of an LLM.
* Golden G011 changed from pass (3/3 earlier runs) to fail. Its anomaly explanation is a table-row quote that the LLM judge now
  rejects as unrelated evidence. The earlier pass may have come from a more lenient judge before the Tier-2 prompt change;
  that was not isolated. The header-row fix recovered G014.
* Over-budget tests use `token_budget=100`: the leaner pipeline no longer crosses 1000 tokens at the re-plan decision point.

## Not yet done

* n=3: only one valid run exists. Runs r2 and r3 were interrupted by HTTP 402 (balance); partial files were removed. The
  r3 file contained 16 tasks with failed LLM calls and was discarded. Resume with
  `python scripts/run_online_eval.py --arch B4 --workers 4 --name online_v12s4_B4_r2 --datasets golden,blind,multidoc`
  (and r3), then `python scripts/s4_summary.py`.
* Full-route latency: planner on the rule parse, a combined debate/manager call, or a faster model would each shorten it;
  none was tried, because each changes report content and needs a pairwise quality check first.
