#!/usr/bin/env python3
"""Collect every run under results/ into one table.

    python summarize.py                 # reads results/, writes results/summary.md
    python summarize.py --csv           # also writes results/summary.csv

Each row is one system on one condition: decision accuracy under both guidelines, how often all
three fields were right, and what the run cost and how long it took.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lcs
from common import CONDITIONS, benchmark_dir, load_answers, load_predictions
from score import score_row


def summarize_run(gold: list[dict], run_dir: Path) -> dict:
    preds = load_predictions(run_dir / "predictions.jsonl")
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    n = len(gold)
    rows = [score_row(g, preds.get(g["patient_id"])) for g in gold]
    row = {
        "system": run_dir.name,
        "model": ", ".join(meta.get("models_served", [])) or "?",
        "uspstf2021": round(lcs.score(gold, preds, "uspstf2021")["accuracy"] * n),
        "acs2023": round(lcs.score(gold, preds, "acs2023")["accuracy"] * n),
        "all_three_fields": sum(r["status"] and r["py"] and r["quit"] for r in rows),
        "n": n,
        "cost_usd": meta.get("cost_usd"),
        "wall_clock_s": meta.get("wall_clock_s"),
        "median_latency_s": meta.get("median_latency_s"),
        "errors": meta.get("n_errors"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results", type=Path)
    ap.add_argument("--benchmark", default="smokingbenchmark")
    ap.add_argument("--csv", action="store_true", help="also write summary.csv")
    a = ap.parse_args()

    bench = benchmark_dir(a.benchmark)
    rows = []
    for condition in CONDITIONS:
        cond_dir = a.results / condition
        if not cond_dir.is_dir():
            continue
        gold = load_answers(bench, condition)
        for run_dir in sorted(p for p in cond_dir.iterdir() if (p / "predictions.jsonl").exists()):
            rows.append({"condition": condition, **summarize_run(gold, run_dir)})

    if not rows:
        raise SystemExit(f"No runs found under {a.results}/")

    cols = ["condition", "system", "model", "uspstf2021", "acs2023", "all_three_fields", "n",
            "cost_usd", "wall_clock_s", "median_latency_s", "errors"]
    head = {"uspstf2021": "USPSTF", "acs2023": "ACS", "all_three_fields": "All 3 fields",
            "cost_usd": "Cost (US$)", "wall_clock_s": "Wall (s)", "median_latency_s": "Median latency (s)"}
    lines = ["| " + " | ".join(head.get(c, c.replace("_", " ").capitalize()) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    table = "\n".join(lines)
    print(table)

    (a.results / "summary.md").write_text(table + "\n")
    print(f"\nwrote {a.results / 'summary.md'}")
    if a.csv:
        with open(a.results / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.results / 'summary.csv'}")


if __name__ == "__main__":
    main()
