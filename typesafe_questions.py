"""The TypeSafe question set used in the study, with the candidate finders and the code that
turns Jev's answers into the shared output.

Jev only makes typed judgments: a Choice over a closed set, or over candidate values that a
regular expression found in the note. All arithmetic and date math happen here, in code, as
TypeSafe recommends for this class of model.
"""
from __future__ import annotations

import datetime as dt
import re

from typesafe_sdk import Choice

from common import SmokingExtraction

NONE = "none"

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
PACKS = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75, "1": 1.0, "1 1/2": 1.5, "2": 2.0, "2 1/2": 2.5, "3": 3.0, "4": 4.0}
WORD_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
WORD_NUM.update({w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())})


# ---------------------------------------------------------------------------- candidate finders (code)
def number_candidates(text: str, limit: int = 240) -> dict[str, float]:
    """Numbers (digits or words) in the note that could be a pack-year value, nearest smoking words first."""
    smoke_pos = [m.start() for m in re.finditer(r"smok|tob|cig|pack|pk|PY|ppd|nicotine", text, re.I)] or [0]
    found: dict[str, tuple[float, int]] = {}
    for m in re.finditer(r"(?<![\d./-])\d{1,3}(?:\.\d+)?(?![\d/])", text):
        v = float(m.group())
        if 0 < v <= 300:
            dist = min(abs(m.start() - s) for s in smoke_pos)
            if m.group() not in found or dist < found[m.group()][1]:
                found[m.group()] = (v, dist)
    word_re = r"\b(" + "|".join(sorted(WORD_NUM, key=len, reverse=True)) + r")(?:[- ](" + "|".join(
        k for k in WORD_NUM if WORD_NUM[k] < 10) + r"))?\b"
    for m in re.finditer(word_re, text, re.I):
        v = WORD_NUM[m.group(1).lower()] + (WORD_NUM[m.group(2).lower()] if m.group(2) else 0)
        if v > 0:
            dist = min(abs(m.start() - s) for s in smoke_pos)
            found.setdefault(m.group(0), (float(v), dist))
    ranked = sorted(found.items(), key=lambda kv: kv[1][1])[:limit]
    return {k: v for k, (v, _) in ranked}


def ints(lo: int, hi: int) -> dict[str, None]:
    return {str(i): None for i in range(lo, hi + 1)}


def with_none(opts: dict, desc: str = "Not stated in the note / does not apply.") -> dict:
    return {**opts, NONE: desc}


MON = "|".join(m[:3] for m in MONTHS)
DATE_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[- ](?:%s)[a-z]*[- ,]+\d{4}\b"
    r"|\b(?:%s)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b" % (MON, MON), re.I)
DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%m.%d.%Y", "%m.%d.%y", "%Y-%m-%d", "%Y/%m/%d",
                "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%B %d %Y", "%b %d %Y")


def parse_date(s: str) -> dt.date | None:
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s.strip()).replace(",", "").replace(".", " " if re.match(r"[A-Za-z]", s) else ".")
    t = re.sub(r"\s+", " ", t).strip()
    for fmt in DATE_FORMATS:
        for cand in (t, t.title()):
            try:
                return dt.datetime.strptime(cand, fmt).date()
            except ValueError:
                pass
    return None


def date_candidates(text: str) -> list[str]:
    out = []
    for m in DATE_RE.finditer(text):
        if m.group() not in out and parse_date(m.group()):
            out.append(m.group())
    return out[:250]


def build_questions(nums: dict[str, float], dates: list[str], max_year: int) -> dict[str, Choice]:
    years = {str(y): None for y in range(1935, max_year + 1)}
    packs = {k: f"{k} pack(s) per day" for k in PACKS}
    num_opts = {k: None for k in nums}
    q: dict[str, Choice] = {
        "status": Choice(
            instructions="What is the patient's own tobacco cigarette smoking status as of this note's visit date? If "
                         "statements conflict (a templated field or text copied from an older visit vs. a current "
                         "statement), use the most recent explicit statement about the patient's own smoking.",
            criteria={
                "current": "Smokes cigarettes now (daily or some days), including patients cutting down, trying to "
                           "quit, with a planned or future quit date, who relapsed after an earlier quit, or who use "
                           "nicotine replacement while still smoking.",
                "former": "Smoked cigarettes regularly in the past and has quit as of the visit (however recently), "
                          "even if now only vaping or using nicotine gum/patches.",
                "never": "Never smoked cigarettes regularly (fewer than 100 lifetime cigarettes, experimented, tried "
                         "a few). Cigars, pipes, cannabis, e-cigarettes, smokeless tobacco, secondhand smoke or "
                         "family members' smoking do not count.",
                "unknown": "The note does not establish it: no smoking history, 'noncontributory', unable to obtain, "
                           "declined, or 'unknown if ever smoked'. Diagnoses such as COPD, medications, screening "
                           "orders or family history alone are not enough.",
            }),
        "py_form": Choice(
            instructions="How does the note describe the patient's cumulative cigarette smoking?",
            criteria={
                "stated": "A single pack-year figure for the patient, e.g. '40 pack-years', '40 PY', 'Pack years: 40'.",
                "stated_range": "A range of pack-years, e.g. '30-40 pack-years'.",
                "rate_and_years": "One rate (packs or cigarettes per day, possibly a range) and how many years they "
                                  "smoked (possibly a range), with no pack-year figure.",
                "rate_from_start": "A current smoker's rate plus when they started (an age like 'since age 17' or a "
                                   "year like 'since 1985'), without a number of years.",
                "rate_start_to_stop": "One rate plus when they started and when they stopped, each given as an age "
                                      "or a calendar year ('from age 20 to 50', 'started at 16, quit in 2005').",
                "multiple_periods": "Two or three periods smoked at different rates, each with its own duration "
                                    "('2 ppd for 10 years, then 1 ppd for 15 years').",
                "other_unclear": "Smoking amount is described some other way that is hard to turn into packs per day "
                                 "and years (e.g. on-and-off smoking without durations).",
                "not_computable": "Only a rate, only a duration, vague wording ('heavy smoker', 'for decades'), a "
                                  "never smoker, or no smoking history.",
            }),
        "py_value": Choice(instructions="Which number in the note is the patient's stated pack-year figure?",
                           criteria=with_none(num_opts, "No single pack-year figure is stated for the patient.")),
        "py_low": Choice(instructions="If the patient's pack-years are given as a range, which number is the lower end?",
                         criteria=with_none(num_opts)),
        "py_high": Choice(instructions="If the patient's pack-years are given as a range, which number is the upper end?",
                          criteria=with_none(num_opts)),
        "rate_unit": Choice(
            instructions="How is the patient's cigarette smoking rate expressed (for a single-rate history)?",
            criteria={"packs_per_day": "As packs per day, e.g. '1 ppd', 'a pack a day', 'half a pack daily'.",
                      "cigarettes_per_day": "As cigarettes per day, e.g. '10 cigs/day', 'about 15 cigarettes a day'.",
                      NONE: "No single smoking rate is given."}),
        "rate_packs": Choice(instructions="For a single-rate history, how many packs per day? If a range, the lower end.",
                             criteria=with_none(packs, "Not given in packs per day.")),
        "rate_packs_high": Choice(instructions="If packs per day is given as a range (e.g. '1-2 ppd'), the upper end.",
                                  criteria=with_none(packs, "Not a range.")),
        "rate_cigs": Choice(instructions="For a single-rate history, how many cigarettes per day?",
                            criteria=with_none(ints(1, 80), "Not given in cigarettes per day.")),
        "years_smoked": Choice(instructions="For a single-rate history, how many years did the patient smoke, as "
                                            "written? If a range, the lower end.",
                               criteria=with_none(ints(1, 80), "No number of years smoked is written.")),
        "years_smoked_high": Choice(instructions="If the number of years smoked is a range (e.g. '30-35 years'), the upper end.",
                                    criteria=with_none(ints(1, 80), "Not a range.")),
        "start_age": Choice(instructions="At what age did the patient start smoking cigarettes, as written?",
                            criteria=with_none(ints(8, 60))),
        "start_year": Choice(instructions="In what calendar year did the patient start smoking, if a year is written "
                                          "(e.g. 'since 1985')?", criteria=with_none(years)),
        "stop_age": Choice(instructions="At what age did the patient last stop smoking cigarettes, as written (e.g. "
                                        "'quit at age 55', 'smoked from 20 until age 50')?",
                           criteria=with_none(ints(15, 95))),
        "break_years": Choice(instructions="If the patient stopped smoking for a while and later restarted, how many "
                                           "years was the break?",
                              criteria=with_none(ints(1, 40), "No break is described.")),
        "quit_form": Choice(
            instructions="How does the note say when the patient finally quit smoking cigarettes?",
            criteria={
                "calendar_date": "A calendar year, with or without month/day: 'quit 2011', 'quit 3/2015', 'Quit date: "
                                 "6/12/2011', 'smoke-free since 2012', \"quit '09\".",
                "years_ago": "A number of years before this visit: 'quit 10 years ago', 'smoke-free x 5 yrs'.",
                "months_ago": "A number of months before this visit: 'quit 8 months ago'.",
                "at_age": "The patient's age when they quit: 'quit at age 55'.",
                "event": "At the time of an event whose calendar year is written in the note: 'quit after his MI' "
                         "with 'NSTEMI (2016)' in the history.",
                "decade": "Part of a decade: 'early 2000s', 'mid-90s'.",
                "other_unclear": "Quit timing is given some other way that is hard to turn into a year.",
                "not_stated": "No quit timing is given, or the patient is not a former smoker (a current smoker's "
                              "earlier quit or planned future quit date does not count).",
            }),
        "quit_year": Choice(
            instructions="In which calendar year did the patient finally quit smoking cigarettes? Use a year written "
                         "in the note (a two-digit year like '09 means 2009). If they quit at the time of an event "
                         "(heart attack, stroke, surgery, diagnosis, retirement), use that event's year from anywhere "
                         "in the note.",
            criteria=with_none(years, "No quit year is written.")),
        "quit_month": Choice(instructions="In which month did the patient finally quit smoking, if a month is written?",
                             criteria=with_none({m: None for m in MONTHS}, "No quit month is written.")),
        "quit_day": Choice(instructions="On which day of the month did the patient quit, if a full quit date is written?",
                           criteria=with_none(ints(1, 31), "No quit day is written.")),
        "quit_years_ago": Choice(instructions="If the note says the patient quit a number of years before this visit, "
                                              "how many years?", criteria=with_none(ints(1, 70))),
        "quit_months_ago": Choice(instructions="If the note says the patient quit a number of months before this visit, "
                                               "how many months?", criteria=with_none(ints(1, 36))),
        "quit_decade": Choice(instructions="If the note says the patient quit in part of a decade, which decade?",
                              criteria=with_none({f"{d}s": None for d in range(1950, 2030, 10)})),
        "quit_decade_part": Choice(instructions="If the note says the patient quit in part of a decade, which part?",
                                   criteria=with_none({"early": None, "mid": None, "late": None})),
        "age": Choice(instructions="How old is the patient, in years, at this note's visit?",
                      criteria=with_none(ints(18, 105))),
    }
    ordinal = {1: "first", 2: "second", 3: "third"}
    for i in (1, 2, 3):
        q[f"p{i}_packs"] = Choice(
            instructions=f"If the note describes smoking periods at different rates, how many packs per day in the "
                         f"{ordinal[i]} period?", criteria=with_none(packs, f"No {ordinal[i]} period, or its rate is "
                                                                             f"given as cigarettes per day."))
        q[f"p{i}_cigs"] = Choice(
            instructions=f"If the note describes smoking periods at different rates and the {ordinal[i]} period's rate "
                         f"is given in cigarettes per day, how many cigarettes per day?",
            criteria=with_none(ints(1, 80), f"No {ordinal[i]} period, or its rate is given in packs."))
        q[f"p{i}_years"] = Choice(
            instructions=f"If the note describes smoking periods at different rates, how many years did the "
                         f"{ordinal[i]} period last?", criteria=with_none(ints(1, 80), f"No {ordinal[i]} period."))
    if dates:
        q["note_date"] = Choice(instructions="Which of these is the date of service (visit date) of this note?",
                                criteria=with_none({d: None for d in dates}, "None of these is the visit date."))
    return q


def compose(ans: dict, nums: dict[str, float]) -> tuple[SmokingExtraction, dict]:
    c = {k: a.choice for k, a in ans.items()}

    def val(key, table=None):
        v = c.get(key, NONE)
        if v in (None, NONE):
            return None
        if table is not None:
            return table[v]
        return nums[v] if v in nums else float(v)

    status = c["status"]
    note_date = parse_date(c["note_date"]) if c.get("note_date", NONE) != NONE else None
    age = val("age")
    birth_year = note_date.year - age if (note_date and age is not None) else None

    if c["rate_unit"] == "cigarettes_per_day" and val("rate_cigs") is not None:
        rate = val("rate_cigs") / 20
    else:
        rate, hi = val("rate_packs", PACKS), val("rate_packs_high", PACKS)
        if rate is not None and hi is not None and hi > rate:
            rate = (rate + hi) / 2
    years = val("years_smoked")
    y_hi = val("years_smoked_high")
    if years is not None and y_hi is not None and y_hi > years:
        years = (years + y_hi) / 2
    brk = val("break_years") or 0

    def start_year():
        if val("start_year") is not None:
            return val("start_year")
        if val("start_age") is not None and birth_year is not None:
            return birth_year + val("start_age")
        return None

    def stop_year():
        if val("stop_age") is not None and birth_year is not None:
            return birth_year + val("stop_age")
        return val("quit_year")

    py = None
    form = c["py_form"]
    if status == "never":
        py = 0.0
    elif status in ("current", "former"):
        if form == "stated":
            py = val("py_value")
        elif form == "stated_range" and None not in (val("py_low"), val("py_high")):
            py = (val("py_low") + val("py_high")) / 2
        elif form == "rate_and_years" and None not in (rate, years):
            py = rate * years
        elif form == "rate_from_start" and rate is not None and note_date and start_year() is not None:
            py = rate * (note_date.year - start_year() - brk)
        elif form == "rate_start_to_stop" and rate is not None and None not in (start_year(), stop_year()):
            py = rate * (stop_year() - start_year() - brk)
        elif form == "multiple_periods":
            total, complete = 0.0, 0
            for i in (1, 2, 3):
                r = val(f"p{i}_packs", PACKS)
                if r is None and val(f"p{i}_cigs") is not None:
                    r = val(f"p{i}_cigs") / 20
                y = val(f"p{i}_years")
                if r is not None and y is not None:
                    total, complete = total + r * y, complete + 1
            py = total if complete >= 2 else None
        py = round(py, 2) if py is not None and py >= 0 else None

    quit = None
    qf = c["quit_form"]
    if status == "former":
        year = val("quit_year")
        if qf in ("calendar_date", "event") and year is not None:
            quit = str(int(year))
            if qf == "calendar_date" and c["quit_month"] != NONE:
                m = MONTHS.index(c["quit_month"]) + 1
                quit += f"-{m:02d}"
                if c["quit_day"] != NONE:
                    try:
                        quit = dt.date(int(year), m, int(c["quit_day"])).isoformat()
                    except ValueError:
                        pass
        elif qf == "years_ago" and note_date and val("quit_years_ago") is not None:
            quit = str(note_date.year - int(val("quit_years_ago")))
        elif qf == "months_ago" and note_date and val("quit_months_ago") is not None:
            mi = note_date.year * 12 + note_date.month - 1 - int(val("quit_months_ago"))
            quit = f"{mi // 12}-{mi % 12 + 1:02d}"
        elif qf == "at_age" and birth_year is not None and val("stop_age") is not None:
            quit = str(int(birth_year + val("stop_age")))
        elif qf == "decade" and c["quit_decade"] != NONE and c["quit_decade_part"] != NONE:
            quit = str(int(c["quit_decade"][:4]) + {"early": 2, "mid": 5, "late": 8}[c["quit_decade_part"]])
        elif qf not in ("not_stated", "other_unclear") and year is not None:
            quit = str(int(year))

    trace = {k: {"choice": a.choice, "confidence": round(a.confidence, 4)} for k, a in ans.items()}
    return SmokingExtraction(smoking_status=status, pack_years=py, quit_date=quit), trace
