#!/usr/bin/env python3
"""Run TypeSafe Jev over one condition of the benchmark.

    python run_typesafe.py --condition basic

Writes predictions.jsonl (one record per note, in the shared output shape) and run_meta.json
(model version, tokens, cost, wall-clock time, latencies) to the output folder. A run can be
interrupted and restarted: notes already in predictions.jsonl are skipped.

Needs TYPESAFE_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import time
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk._version import __version__ as SDK_VERSION

from common import CONDITIONS, append_jsonl, benchmark_dir, done_ids, load_env, load_notes
from typesafe_questions import build_questions, compose, date_candidates, number_candidates

MODEL = "jev-1.13.0"                 # the version used in the study (jev-latest may move)
PRICE_PER_INPUT_TOKEN = 0.042 / 1e6  # USD; output tokens are free (docs.typesafe.ai/models, 2026-09-18)


async def run(benchmark: Path, condition: str, out: Path, model: str, concurrency: int,
              limit: int | None, ids: list[str] | None):
    load_env()
    notes = load_notes(benchmark, condition, limit, ids)
    pred_path = out / "predictions.jsonl"
    skip = done_ids(pred_path)
    todo = [n for n in notes if n["patient_id"] not in skip]
    print(f"{len(todo)} notes to run ({len(skip)} already done) -> {pred_path}")

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    async with AsyncTypeSafeClient(model=model, timeout=120.0) as client:
        async def one(n):
            text = n["note_text"]
            nums, dates = number_candidates(text), date_candidates(text)
            questions = build_questions(nums, dates, max_year=dt.date.today().year)
            async with sem:
                t = time.time()
                try:
                    resp = await client.system_one(state={"clinical_note": text}, questions=questions)
                    latency = time.time() - t
                    pred, trace = compose(resp.answers, nums)
                    rec = {"patient_id": n["patient_id"], **pred.model_dump(), "model": resp.model,
                           "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                           "latency_s": round(latency, 3), "n_questions": len(questions), "answers": trace}
                except Exception as e:  # record the failure and keep going
                    rec = {"patient_id": n["patient_id"], "error": f"{type(e).__name__}: {e}",
                           "latency_s": round(time.time() - t, 3)}
            append_jsonl(pred_path, rec)
            return rec

        recs = await asyncio.gather(*(one(n) for n in todo))

    wall = time.time() - t0
    ok = [r for r in recs if "error" not in r]
    tokens_in = sum(r["input_tokens"] or 0 for r in ok)
    n_q = f"~{round(sum(r['n_questions'] for r in ok) / len(ok))}" if ok else "?"
    meta = {"system": "typesafe", "model_requested": model, "models_served": sorted({r["model"] for r in ok}),
            "settings": f"1 request/note, {n_q} Choice questions; arithmetic and dates in code",
            "sdk": f"typesafe-sdk {SDK_VERSION}", "condition": condition,
            "started_utc": started, "wall_clock_s": round(wall, 2), "concurrency": concurrency,
            "n_requested": len(todo), "n_ok": len(ok), "n_errors": len(recs) - len(ok),
            "input_tokens": tokens_in, "output_tokens": sum(r["output_tokens"] or 0 for r in ok),
            "cost_usd": round(tokens_in * PRICE_PER_INPUT_TOKEN, 6),
            "pricing": "$0.042 per 1M input tokens, output free (docs.typesafe.ai/models, retrieved 2026-09-18)",
            "median_latency_s": round(sorted(r["latency_s"] for r in ok)[len(ok) // 2], 3) if ok else None}
    if ok:
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="smokingbenchmark", help="path to a clone of vclic/smokingbenchmark")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--out", default=None, help="default: results/<condition>/typesafe")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, help="run only the first N notes (for a quick check)")
    ap.add_argument("--ids", nargs="*", help="run only these patient IDs")
    a = ap.parse_args()
    bench = benchmark_dir(a.benchmark)
    out = Path(a.out) if a.out else Path("results") / a.condition / "typesafe"
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(bench, a.condition, out, a.model, a.concurrency, a.limit, a.ids))


if __name__ == "__main__":
    main()
