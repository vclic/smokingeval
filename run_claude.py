#!/usr/bin/env python3
"""Run a Claude model over one condition of the benchmark, using structured outputs.

    python run_claude.py --condition basic --model claude-sonnet-5 --effort low
    python run_claude.py --condition basic --model claude-haiku-4-5

Writes predictions.jsonl (one record per note, in the shared output shape) and run_meta.json
(model version, tokens, cost, wall-clock time, latencies) to the output folder. A run can be
interrupted and restarted: notes already in predictions.jsonl are skipped.

Needs ANTHROPIC_API_KEY in the environment (SMOKING_BENCH_ANTHROPIC_API_KEY is used first if it
is set, with SMOKING_BENCH_ANTHROPIC_WORKSPACE_ID for an organization-level key).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import time
from pathlib import Path

import anthropic

from common import (CODEBOOK, CONDITIONS, ClaudeExtractionV2, append_jsonl, benchmark_dir, done_ids,
                    load_env, load_notes)

MODEL = "claude-sonnet-5"
# USD per token, Anthropic list prices as of the study
PRICES = {"claude-opus-5": (5.00 / 1e6, 25.00 / 1e6), "claude-sonnet-5": (2.00 / 1e6, 10.00 / 1e6),
          "claude-haiku-4-5": (1.00 / 1e6, 5.00 / 1e6), "claude-fable-5-1": (10.00 / 1e6, 50.00 / 1e6)}


def cost(model: str, usage) -> float:
    p_in, p_out = PRICES[model]
    cache_write = (usage.cache_creation_input_tokens or 0) * p_in * 1.25
    cache_read = (usage.cache_read_input_tokens or 0) * p_in * 0.1
    return usage.input_tokens * p_in + usage.output_tokens * p_out + cache_write + cache_read


async def run(benchmark: Path, condition: str, out: Path, model: str, effort: str | None,
              concurrency: int, limit: int | None, ids: list[str] | None):
    load_env()
    notes = load_notes(benchmark, condition, limit, ids)
    pred_path = out / "predictions.jsonl"
    skip = done_ids(pred_path)
    todo = [n for n in notes if n["patient_id"] not in skip]
    print(f"{len(todo)} notes to run ({len(skip)} already done) -> {pred_path}")

    extra = {"output_config": {"effort": effort}} if effort else {}
    api_key = os.environ.get("SMOKING_BENCH_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    ws = os.environ.get("SMOKING_BENCH_ANTHROPIC_WORKSPACE_ID")  # organization-level keys need a workspace
    headers = {"anthropic-workspace-id": ws} if ws else None

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    async with anthropic.AsyncAnthropic(api_key=api_key, max_retries=6, default_headers=headers) as client:
        async def one(n):
            async with sem:
                t = time.time()
                try:
                    resp = await client.messages.parse(
                        model=model,
                        max_tokens=16000,
                        system=CODEBOOK,
                        messages=[{"role": "user", "content": f"NOTE:\n{n['note_text']}"}],
                        output_format=ClaudeExtractionV2,
                        **extra,
                    )
                    latency = time.time() - t
                    rec = {"patient_id": n["patient_id"], "model": resp.model, "request_id": resp._request_id,
                           "stop_reason": resp.stop_reason, "input_tokens": resp.usage.input_tokens,
                           "output_tokens": resp.usage.output_tokens, "cost_usd": round(cost(model, resp.usage), 6),
                           "latency_s": round(latency, 3)}
                    if resp.stop_reason == "refusal" or resp.parsed_output is None:
                        rec["error"] = f"no parsed output (stop_reason={resp.stop_reason})"
                    else:
                        out_ = resp.parsed_output
                        # only the three shared fields are scored; the evidence fields are kept for error analysis
                        rec.update(out_.final().model_dump())
                        rec["full_output"] = out_.model_dump()
                except anthropic.APIStatusError as e:
                    rec = {"patient_id": n["patient_id"], "error": f"{type(e).__name__} {e.status_code}: {e.message}",
                           "latency_s": round(time.time() - t, 3)}
                except Exception as e:  # e.g. output that failed schema validation
                    rec = {"patient_id": n["patient_id"], "error": f"{type(e).__name__}: {e}",
                           "latency_s": round(time.time() - t, 3)}
            append_jsonl(pred_path, rec)
            return rec

        recs = await asyncio.gather(*(one(n) for n in todo))

    wall = time.time() - t0
    ok = [r for r in recs if "error" not in r]
    billed = [r for r in recs if "input_tokens" in r]
    meta = {"system": "claude", "model_requested": model, "models_served": sorted({r["model"] for r in billed}),
            "sdk": f"anthropic {anthropic.__version__}", "condition": condition,
            "structured_outputs": "messages.parse(output_format=ClaudeExtractionV2); scored fields = SmokingExtraction",
            "effort": effort or "model default", "max_tokens": 16000,
            "started_utc": started, "wall_clock_s": round(wall, 2), "concurrency": concurrency,
            "n_requested": len(todo), "n_ok": len(ok), "n_errors": len(recs) - len(ok),
            "input_tokens": sum(r["input_tokens"] for r in billed),
            "output_tokens": sum(r["output_tokens"] for r in billed),
            "cost_usd": round(sum(r["cost_usd"] for r in billed), 4),
            "pricing": f"${PRICES[model][0] * 1e6:.2f} / ${PRICES[model][1] * 1e6:.2f} per 1M input/output tokens "
                       "(Anthropic list price)",
            "median_latency_s": round(sorted(r["latency_s"] for r in ok)[len(ok) // 2], 3) if ok else None}
    if ok:
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="smokingbenchmark", help="path to a clone of vclic/smokingbenchmark")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--model", default=MODEL, choices=sorted(PRICES))
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                    help="how much the model reasons before answering; the study used low for Sonnet 5 "
                         "and the default (no extended thinking) for Haiku 4.5")
    ap.add_argument("--out", default=None, help="default: results/<condition>/<model>")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, help="run only the first N notes (for a quick check)")
    ap.add_argument("--ids", nargs="*", help="run only these patient IDs")
    a = ap.parse_args()
    bench = benchmark_dir(a.benchmark)
    out = Path(a.out) if a.out else Path("results") / a.condition / a.model
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(bench, a.condition, out, a.model, a.effort, a.concurrency, a.limit, a.ids))


if __name__ == "__main__":
    main()
