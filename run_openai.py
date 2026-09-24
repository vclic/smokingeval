#!/usr/bin/env python3
"""Run an OpenAI model over one condition of the benchmark, using structured outputs.

    python run_openai.py --condition basic --model gpt-6-sol --effort low
    python run_openai.py --condition basic --model gpt-6-luna --effort none

The system prompt and the output schema are the same ones the Claude runs used, so the only
difference between the two vendors is the model.

Writes predictions.jsonl (one record per note, in the shared output shape) and run_meta.json
(model version, tokens including reasoning tokens, cost, wall-clock time, latencies) to the output
folder. A run can be interrupted and restarted: notes already in predictions.jsonl are skipped.

Needs OPENAI_API_KEY in the environment (SMOKING_BENCH_OPENAI_API_KEY is used first if set).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import ssl
import time
from pathlib import Path

try:  # the OpenAI SDK ships with httpx2; fall back to httpx for older installs
    import httpx2 as httpx
except ImportError:
    import httpx
import openai

from common import (CODEBOOK, CONDITIONS, ClaudeExtractionV2, append_jsonl, benchmark_dir, done_ids,
                    load_env, load_notes)

MODEL = "gpt-6-sol"
# USD per token, OpenAI list prices as of the study
PRICES = {"gpt-6-astra": (10.00 / 1e6, 50.00 / 1e6), "gpt-6-sol": (2.00 / 1e6, 10.00 / 1e6),
          "gpt-6-luna": (0.10 / 1e6, 0.50 / 1e6), "gpt-5.4-mini": (0.75 / 1e6, 4.50 / 1e6),
          "gpt-5-mini": (0.25 / 1e6, 2.00 / 1e6)}


def http_client() -> httpx.AsyncClient | None:
    """Honor a custom CA bundle, for networks that inspect TLS traffic.

    Set SSL_CERT_FILE (or SMOKING_BENCH_CA_BUNDLE) to a bundle that includes the organization's
    certificate authority. Partial-chain verification is enabled because such CAs are usually
    installed as trust anchors even though they are cross-signed by another root.
    """
    bundle = os.environ.get("SMOKING_BENCH_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if not bundle:
        return None
    ctx = ssl.create_default_context(cafile=bundle)
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return httpx.AsyncClient(verify=ctx, timeout=300.0)


def cost(model: str, usage) -> float:
    """Reasoning tokens are billed as output tokens and are already included in output_tokens."""
    p_in, p_out = PRICES[model]
    cached = (usage.input_tokens_details.cached_tokens or 0) if usage.input_tokens_details else 0
    return (usage.input_tokens - cached) * p_in + cached * p_in * 0.1 + usage.output_tokens * p_out


async def run(benchmark: Path, condition: str, out: Path, model: str, effort: str | None,
              concurrency: int, limit: int | None, ids: list[str] | None):
    load_env()
    notes = load_notes(benchmark, condition, limit, ids)
    pred_path = out / "predictions.jsonl"
    skip = done_ids(pred_path)
    todo = [n for n in notes if n["patient_id"] not in skip]
    print(f"{len(todo)} notes to run ({len(skip)} already done) -> {pred_path}")

    extra = {"reasoning": {"effort": effort}} if effort else {}
    api_key = os.environ.get("SMOKING_BENCH_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    hc = http_client()
    async with openai.AsyncOpenAI(api_key=api_key, max_retries=6, timeout=300.0,
                                  **({"http_client": hc} if hc else {})) as client:
        async def one(n):
            async with sem:
                t = time.time()
                try:
                    resp = await client.responses.parse(
                        model=model,
                        max_output_tokens=16000,
                        instructions=CODEBOOK,
                        input=[{"role": "user", "content": f"NOTE:\n{n['note_text']}"}],
                        text_format=ClaudeExtractionV2,
                        **extra,
                    )
                    latency = time.time() - t
                    u = resp.usage
                    rec = {"patient_id": n["patient_id"], "model": resp.model, "response_id": resp.id,
                           "status": resp.status, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                           "reasoning_tokens": u.output_tokens_details.reasoning_tokens if u.output_tokens_details else None,
                           "cost_usd": round(cost(model, u), 6), "latency_s": round(latency, 3)}
                    if resp.output_parsed is None:
                        rec["error"] = f"no parsed output (status={resp.status}, {resp.incomplete_details})"
                    else:
                        # only the three shared fields are scored; the evidence fields are kept for error analysis
                        rec.update(resp.output_parsed.final().model_dump())
                        rec["full_output"] = resp.output_parsed.model_dump()
                except openai.APIStatusError as e:
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
    meta = {"system": "openai", "model_requested": model, "models_served": sorted({r["model"] for r in billed}),
            "sdk": f"openai {openai.__version__}", "condition": condition,
            "structured_outputs": "responses.parse(text_format=ClaudeExtractionV2); scored fields = SmokingExtraction",
            "effort": effort or "model default", "max_output_tokens": 16000,
            "started_utc": started, "wall_clock_s": round(wall, 2), "concurrency": concurrency,
            "n_requested": len(todo), "n_ok": len(ok), "n_errors": len(recs) - len(ok),
            "input_tokens": sum(r["input_tokens"] for r in billed),
            "output_tokens": sum(r["output_tokens"] for r in billed),
            "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in billed),
            "cost_usd": round(sum(r["cost_usd"] for r in billed), 4),
            "pricing": f"${PRICES[model][0] * 1e6:.2f} / ${PRICES[model][1] * 1e6:.2f} per 1M input/output tokens "
                       "(OpenAI list price)",
            "median_latency_s": round(sorted(r["latency_s"] for r in ok)[len(ok) // 2], 3) if ok else None}
    if ok:
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="smokingbenchmark", help="path to a clone of vclic/smokingbenchmark")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--model", default=MODEL, choices=sorted(PRICES))
    ap.add_argument("--effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default=None,
                    help="reasoning effort; the study used low for the mid-size model and none for the small one")
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
