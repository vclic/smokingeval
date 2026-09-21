"""Shared pieces of the evaluation: the output model every system must produce, the codebook
given to generative models, and small I/O helpers.

The extraction schema and the codebook below are the ones used in the study, unchanged.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CONDITIONS = ("basic", "messy", "complex")
ENV_FILE = Path.home() / ".config" / "smoking-bench" / ".env"


class SmokingExtraction(BaseModel):
    """The single output model shared by every system under test."""

    smoking_status: Literal["current", "former", "never", "unknown"] = Field(
        description="Tobacco cigarette smoking status as of the note date.")
    pack_years: Optional[float] = Field(
        description="Lifetime pack-years (packs/day x years smoked). 0 for never smokers; null if not determinable.")
    quit_date: Optional[str] = Field(
        description="Former smokers only: quit date as YYYY, YYYY-MM, or YYYY-MM-DD at the precision the note "
                    "supports; null otherwise.")

    @field_validator("quit_date")
    @classmethod
    def _date_shape(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", v):
            raise ValueError(f"quit_date must be YYYY, YYYY-MM or YYYY-MM-DD, got {v!r}")
        return v


CODEBOOK = """You are abstracting tobacco smoking history from one clinical note to support lung cancer
screening decisions. Read the whole note: smoking information may be in the HPI, social history,
templated EHR fields, problem list, assessment/plan, or copied-forward text.

SMOKING STATUS (tobacco cigarettes, as of this note's visit date)
- current: smokes any cigarettes now (daily or some days). Includes patients who are cutting
  down, "trying to quit", have a planned or future quit date, relapsed after an earlier quit, or
  use nicotine replacement while still smoking.
- former: smoked regularly in the past and has quit as of the visit, however recently. Still
  former if they now only vape or use nicotine gum/patches.
- never: never smoked regularly (fewer than 100 lifetime cigarettes, "experimented", "tried a
  few").
- unknown: the note does not establish it (no smoking history, "noncontributory", "unable to
  obtain", declined, "unknown if ever smoked"). Do not infer status from diagnoses (COPD, lung
  nodule), medications, screening orders, or family history.
- Cigars, pipes, cannabis, e-cigarettes, smokeless tobacco, secondhand smoke, and family
  members' smoking do not count as cigarette smoking.
- When statements conflict (e.g. a templated field or text copied from an older visit vs. a
  current statement), use the most recent explicit statement about the patient's own smoking
  as of this visit. Older dated statements describe the past.

PACK-YEARS (lifetime packs per day x years smoked; 1 pack = 20 cigarettes)
- If a pack-year figure is stated for the patient, use it; for a stated range use the midpoint.
- Otherwise compute it only from what the note states: for each smoking period, packs/day x
  years, then add the periods. Years may come from durations, from start/stop ages, or from
  start/stop calendar years (use the patient's age or date of birth and the visit date).
  Exclude stated breaks (years the patient had stopped before restarting). Use the midpoint of a
  stated range of rates or years.
- Never assume a rate or a duration. If only a rate, only a duration, or vague wording ("heavy
  smoker", "long history", "for decades") is given, pack-years is null.
- 0 for never smokers; null when status is unknown.

QUIT DATE (former smokers only; otherwise null)
- The date of the final, lasting quit, at the precision the note supports: YYYY, YYYY-MM, or
  YYYY-MM-DD.
- Resolve relative expressions against this note's visit date (not signature, lab, or older
  copied dates): "10 years ago" -> visit year - 10; "8 months ago" -> YYYY-MM; "at age 55" -> use
  the patient's age or date of birth; "when she retired in 2011" or "after his 2016 MI" -> that
  year, including when the event's year appears elsewhere in the note; a two-digit year like '09
  means 2009; "early/mid/late 2000s" -> approximately 2002 / 2005 / 2008.
- A current smoker's earlier quit attempt or planned future quit date is not a quit date.
- null if no quit timing is given.

Fill the evidence and component fields first, then the final smoking_status, pack_years, and
quit_date fields."""


class Period(BaseModel):
    source: str = Field(description="Verbatim quote(s) describing this smoking period.")
    packs_per_day: Optional[float] = Field(description="Packs per day in this period (cigarettes/20); null if not stated.")
    years: Optional[float] = Field(description="Years this period lasted, stated or computed from ages/years in the "
                                               "note, excluding breaks; null if not determinable.")


class ClaudeExtractionV2(BaseModel):
    """Claude's structured output: evidence and components first, then the shared final fields."""

    visit_date: Optional[str] = Field(description="This note's date of service as YYYY-MM-DD, if stated.")
    patient_age: Optional[int] = Field(description="Patient's age at the visit, if stated or computable from DOB.")
    status_evidence: list[str] = Field(description="Verbatim quotes about the patient's own cigarette smoking.")
    stated_pack_years: Optional[float] = Field(description="A pack-year figure stated in the note (midpoint of a range).")
    smoking_periods: list[Period] = Field(description="Each period with a distinct rate, in order.")
    pack_years_basis: Literal["stated", "computed_from_periods", "not_determinable", "never_smoker",
                              "status_unknown"]
    quit_evidence: Optional[str] = Field(description="Verbatim quote giving the final quit timing, if any.")
    smoking_status: Literal["current", "former", "never", "unknown"]
    pack_years: Optional[float]
    quit_date: Optional[str] = Field(description="YYYY, YYYY-MM, or YYYY-MM-DD; null unless former with quit timing.")

    @field_validator("quit_date")
    @classmethod
    def _date_shape(cls, v):
        return SmokingExtraction._date_shape(v)

    def final(self) -> SmokingExtraction:
        return SmokingExtraction(smoking_status=self.smoking_status, pack_years=self.pack_years,
                                 quit_date=self.quit_date)

# ---------------------------------------------------------------------------- benchmark I/O
def benchmark_dir(path: str | Path) -> Path:
    """Folder holding the benchmark, i.e. a clone of github.com/vclic/smokingbenchmark."""
    p = Path(path).expanduser().resolve()
    if not (p / "basic" / "answers.csv").exists():
        raise SystemExit(f"No benchmark at {p}. Clone https://github.com/vclic/smokingbenchmark "
                         f"into this folder, or pass its path with --benchmark.")
    return p


def load_notes(benchmark: Path, condition: str, limit: int | None = None,
               ids: list[str] | None = None) -> list[dict]:
    files = sorted((benchmark / condition / "notes").glob("*.txt"))
    notes = [{"patient_id": f.stem, "note_text": f.read_text(encoding="utf-8")} for f in files]
    if ids:
        keep = set(ids)
        notes = [n for n in notes if n["patient_id"] in keep]
    return notes[:limit] if limit else notes


def load_answers(benchmark: Path, condition: str) -> list[dict]:
    with open(benchmark / condition / "answers.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions(path: Path) -> dict:
    """patient_id -> prediction, from a run's predictions.jsonl."""
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {str(r["patient_id"]): r for r in recs}


# ---------------------------------------------------------------------------- keys and run files
def load_env():
    """Put API keys in the environment, from ~/.config/smoking-bench/.env if it exists.

    Keys are read from the environment only: TYPESAFE_API_KEY, and ANTHROPIC_API_KEY (or
    SMOKING_BENCH_ANTHROPIC_API_KEY, plus SMOKING_BENCH_ANTHROPIC_WORKSPACE_ID for an
    organization-level key). Nothing is read from, or written to, this repository.
    """
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def append_jsonl(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    """IDs already written to a predictions file, so an interrupted run can be resumed."""
    if not path.exists():
        return set()
    return {json.loads(l)["patient_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
