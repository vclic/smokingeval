"""Lung cancer screening (LCS) eligibility from extracted smoking data.

Two guidelines, both applied in code to the same extracted values:
  USPSTF 2021: age 50-80, >= 20 pack-years, currently smokes or quit within the past 15 years.
  ACS 2023:    age 50-80, >= 20 pack-years, current or former smoker (no years-since-quit limit).

Age comes from EHR demographics in a real CDS, and every benchmark patient is 50-80, so age is
not part of the note-derived decision. Years since quitting = note year - quit year.

Decisions: "eligible", "not_eligible", "insufficient" (the note does not settle it -> CDS should
prompt a clinician). The gold decision is a *set*: when the gold tolerance range straddles a
threshold (e.g. "15-25 pack-years"), either side is accepted.
"""
from __future__ import annotations

import re

PY_MIN = 20
YSQ_MAX = 15
GUIDELINES = ("uspstf2021", "acs2023")
DECISIONS = ("eligible", "not_eligible", "insufficient")


def _decide(status: str, py_bucket: str, ysq_bucket: str, guideline: str) -> str:
    """py_bucket: '>=20' | '<20' | 'unknown';  ysq_bucket: '<=15' | '>15' | 'unknown'."""
    if status == "never":
        return "not_eligible"
    if status not in ("current", "former"):
        return "insufficient"
    if py_bucket == "<20":
        return "not_eligible"
    if status == "current" or guideline == "acs2023":
        return "eligible" if py_bucket == ">=20" else "insufficient"
    # USPSTF, former smoker
    if ysq_bucket == ">15":
        return "not_eligible"          # decidable even without pack-years
    if py_bucket == "unknown" or ysq_bucket == "unknown":
        return "insufficient"
    return "eligible"


def _year(v) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(v)) if v not in (None, "") else None
    return int(m.group()) if m else None


def predicted_decision(status, pack_years, quit_date, note_year: int, guideline: str) -> str:
    status = str(status or "").strip().lower()
    try:
        py = float(pack_years) if pack_years not in (None, "") else None
    except (TypeError, ValueError):
        py = None
    py_b = "unknown" if py is None else (">=20" if py >= PY_MIN else "<20")
    qy = _year(quit_date)
    ysq_b = "unknown" if qy is None else ("<=15" if note_year - qy <= YSQ_MAX else ">15")
    return _decide(status, py_b, ysq_b, guideline)


def gold_decisions(g: dict, guideline: str) -> set[str]:
    """All decisions consistent with the gold row's tolerance ranges."""
    note_year = int(g["note_date"][:4])
    if g["pack_years"] == "":
        py_bs = {"unknown"}
    else:
        lo, hi = float(g["pack_years_min"]), float(g["pack_years_max"])
        py_bs = {">=20" if x >= PY_MIN else "<20" for x in (lo, hi)}
    if g["quit_date"] == "":
        ysq_bs = {"unknown"}
    else:
        ysq_bs = {"<=15" if note_year - int(y) <= YSQ_MAX else ">15"
                  for y in (g["quit_year_min"], g["quit_year_max"])}
    return {_decide(g["smoking_status"], p, y, guideline) for p in py_bs for y in ysq_bs}


def score(gold_rows: list[dict], preds: dict, guideline: str) -> dict:
    """Decision-level metrics. preds: patient_id -> dict with smoking_status/pack_years/quit_date."""
    conf = {(a, b): 0 for a in DECISIONS + ("borderline",) for b in DECISIONS}
    correct = 0
    per_note = {}
    for g in gold_rows:
        p = preds.get(g["patient_id"]) or {}
        gs = gold_decisions(g, guideline)
        pd = predicted_decision(p.get("smoking_status"), p.get("pack_years"), p.get("quit_date"),
                                int(g["note_date"][:4]), guideline)
        ok = pd in gs
        correct += ok
        per_note[g["patient_id"]] = ok
        conf[(next(iter(gs)) if len(gs) == 1 else "borderline", pd)] += 1
    n = len(gold_rows)
    gold_elig = sum(v for (a, _), v in conf.items() if a == "eligible")
    gold_not = sum(v for (a, _), v in conf.items() if a == "not_eligible")
    gold_ins = sum(v for (a, _), v in conf.items() if a == "insufficient")
    return {
        "accuracy": correct / n, "n": n, "confusion": conf, "per_note": per_note,
        # the costly error: someone who should be screened is told they are not eligible
        "missed_eligible": conf[("eligible", "not_eligible")], "gold_eligible": gold_elig,
        "false_eligible": conf[("not_eligible", "eligible")] + conf[("insufficient", "eligible")],
        "decided_without_info": conf[("insufficient", "eligible")] + conf[("insufficient", "not_eligible")],
        "gold_insufficient": gold_ins,
        "needless_prompt": conf[("eligible", "insufficient")] + conf[("not_eligible", "insufficient")],
        "gold_not_eligible": gold_not,
        "sensitivity": conf[("eligible", "eligible")] / gold_elig if gold_elig else None,
    }
