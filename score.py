#!/usr/bin/env python3
"""Score one run against the benchmark answer key.

    python score.py results/basic/typesafe --condition basic
    python score.py results/complex/claude-sonnet-5 --condition complex --by pack_years_source

Reports the three extracted fields, the lung cancer screening decision under both guidelines, and
the screening flags a simple CDS rule would produce from the extracted values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import lcs
from common import CONDITIONS, benchmark_dir, load_answers, load_predictions

STATUSES = ["current", "former", "never", "unknown"]
NULLS = {"", "none", "null", "na", "n/a", "nan", "unknown", "not documented", "not_documented"}
PY_ABS_TOL = 0.5  # absorbs rounding (e.g. 8.125 reported as 8.1)


def is_null(v) -> bool:
    return v is None or str(v).strip().lower() in NULLS


def to_float(v):
    if is_null(v):
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    return float(m.group()) if m else None


def to_year(v):
    if is_null(v):
        return None
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group()) if m else None


def score_row(g: dict, p: dict | None) -> dict:
    """Score one note's three fields against the answer key."""
    p = p or {}
    out = {"status": str(p.get("smoking_status", "")).strip().lower() == g["smoking_status"]}

    # pack-years: correct if both are null, or the prediction is inside the accepted range
    gpy, ppy = to_float(g["pack_years"]), to_float(p.get("pack_years"))
    if gpy is None:
        out["py"] = ppy is None
        out["py_invented"] = ppy is not None
    else:
        lo, hi = float(g["pack_years_min"]) - PY_ABS_TOL, float(g["pack_years_max"]) + PY_ABS_TOL
        out["py"] = ppy is not None and lo <= ppy <= hi
        out["py_missed"] = ppy is None

    # quit date: scored on the year, inside the accepted range
    pq = to_year(p.get("quit_date"))
    if not g["quit_date"]:
        out["quit"] = pq is None
        out["quit_invented"] = pq is not None
    else:
        out["quit"] = pq is not None and int(g["quit_year_min"]) <= pq <= int(g["quit_year_max"])
        out["quit_missed"] = pq is None
    return out


def pct(n, d):
    return f"{100 * n / d:5.1f}% ({n}/{d})" if d else "   n/a"


def screening_flags(gold: list[dict], preds: dict) -> dict:
    """A flag is raised when the extracted values make the patient eligible (USPSTF 2021)."""
    tp = fp = fn = tn = 0
    for g in gold:
        p = preds.get(g["patient_id"]) or {}
        pd = lcs.predicted_decision(p.get("smoking_status"), p.get("pack_years"), p.get("quit_date"),
                                    int(g["note_date"][:4]), "uspstf2021")
        pred_flag = pd == "eligible"
        gold_flags = {d == "eligible" for d in lcs.gold_decisions(g, "uspstf2021")}
        if pred_flag in gold_flags:      # a borderline answer key accepts either decision
            tp += pred_flag
            tn += not pred_flag
        elif pred_flag:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="a run folder, or its predictions.jsonl")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--benchmark", default="smokingbenchmark")
    ap.add_argument("--by", action="append", default=[],
                    help="answer-key column to break the fields down by (repeatable); "
                         "'challenge_tags' splits on ';'")
    a = ap.parse_args()

    path = a.run / "predictions.jsonl" if a.run.is_dir() else a.run
    gold = load_answers(benchmark_dir(a.benchmark), a.condition)
    preds = load_predictions(path)
    n = len(gold)

    print(f"{path}  ({a.condition}, {n} notes)")
    missing = [g["patient_id"] for g in gold if g["patient_id"] not in preds]
    errors = [r for r in preds.values() if "error" in r]
    if missing:
        print(f"  WARNING: {len(missing)} notes have no prediction; they are scored as wrong")
    if errors:
        print(f"  WARNING: {len(errors)} notes returned an error; they are scored as wrong")

    results = [(g, score_row(g, preds.get(g["patient_id"]))) for g in gold]

    print("-" * 70)
    print(f"  smoking status  {pct(sum(r['status'] for _, r in results), n)}")
    print(f"  pack-years      {pct(sum(r['py'] for _, r in results), n)}"
          f"   invented: {sum(r.get('py_invented', False) for _, r in results)}"
          f"   missed: {sum(r.get('py_missed', False) for _, r in results)}")
    print(f"  quit date       {pct(sum(r['quit'] for _, r in results), n)}"
          f"   invented: {sum(r.get('quit_invented', False) for _, r in results)}"
          f"   missed: {sum(r.get('quit_missed', False) for _, r in results)}")
    print(f"  all three       {pct(sum(r['status'] and r['py'] and r['quit'] for _, r in results), n)}")

    print("-" * 70)
    for guideline in lcs.GUIDELINES:
        s = lcs.score(gold, preds, guideline)
        print(f"  {guideline:11s} decision {pct(round(s['accuracy'] * n), n)}"
              f"   missed eligible: {s['missed_eligible']}   wrongly eligible: {s['false_eligible']}")

    f = screening_flags(gold, preds)
    print(f"  screening flags (USPSTF 2021): TP {f['tp']}  FP {f['fp']}  FN {f['fn']}  TN {f['tn']}  "
          f"sens {f['sensitivity']:.3f}  spec {f['specificity']:.3f}  PPV {f['ppv']:.3f}")

    for col in a.by:
        groups = defaultdict(list)
        for g, r in results:
            keys = [k for k in g[col].split(";") if k] if col == "challenge_tags" else [g[col]]
            for k in keys or ["(none)"]:
                groups[k].append(r)
        print("-" * 70)
        print(f"  by {col}")
        print(f"  {'group':44s} {'n':>4s} {'status':>7s} {'py':>7s} {'quit':>7s}")
        for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            m = len(rs)
            print(f"  {k[:44]:44s} {m:4d} {100 * sum(r['status'] for r in rs) / m:6.1f}% "
                  f"{100 * sum(r['py'] for r in rs) / m:6.1f}% {100 * sum(r['quit'] for r in rs) / m:6.1f}%")


if __name__ == "__main__":
    main()
