# smokingeval

Code to run language models over the [smoking benchmark](https://github.com/vclic/smokingbenchmark) and score what they extract.

The benchmark is 3,000 synthetic outpatient notes with a known answer for each one: smoking status, pack-years, and quit date. This repository does two things with it. First, it sends each note to a model and writes the extracted values to a file. Second, it compares those values with the answer key and reports how often each field was right, what lung cancer screening (LCS) decision would follow from them, what the run cost, and how long it took.

Three systems were compared in our study: TypeSafe Jev 1.13, Claude Haiku 4.5, and Claude Sonnet 5. All three are given the same note text and must return the same three fields, so the comparison is of the systems and not of the output formats. Adding a fourth system means writing one more runner that produces `predictions.jsonl` in the same shape; everything downstream works unchanged.

## Setup

```bash
git clone https://github.com/vclic/smokingeval.git
cd smokingeval
git clone https://github.com/vclic/smokingbenchmark.git   # the notes and the answer key
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The scripts look for the benchmark in `./smokingbenchmark`. If you put it somewhere else, pass `--benchmark /path/to/smokingbenchmark` to every command below.

API keys are read from the environment and nothing else:

```bash
export TYPESAFE_API_KEY=...     # for Jev
export ANTHROPIC_API_KEY=...    # for Claude
```

If you keep keys in `~/.config/smoking-bench/.env`, that file is read as well. An organization-level Anthropic key also needs `SMOKING_BENCH_ANTHROPIC_WORKSPACE_ID`, and will be picked up from `SMOKING_BENCH_ANTHROPIC_API_KEY` if you would rather not set `ANTHROPIC_API_KEY`.

## Running a system

Each run covers one of the three conditions (`basic`, `messy`, `complex`):

```bash
python run_typesafe.py --condition basic
python run_claude.py   --condition basic --model claude-sonnet-5 --effort low
python run_claude.py   --condition basic --model claude-haiku-4-5
```

Each writes two files to `results/<condition>/<system>/`: `predictions.jsonl`, with one record per note, and `run_meta.json`, with the model version the API served, tokens, cost, wall-clock time, and latencies. Both runners send 16 requests at a time (`--concurrency`), skip notes that are already in `predictions.jsonl`, so an interrupted run can be restarted by repeating the command, and record failures rather than stopping. Use `--limit 10` first to check that the keys work.

## Scoring

```bash
python score.py results/basic/typesafe --condition basic
```

```
results/basic/typesafe/predictions.jsonl  (basic, 1000 notes)
----------------------------------------------------------------------
  smoking status   99.7% (997/1000)
  pack-years       98.8% (988/1000)   invented: 8   missed: 3
  quit date        99.7% (997/1000)   invented: 0   missed: 2
  all three        98.6% (986/1000)
----------------------------------------------------------------------
  uspstf2021  decision  99.1% (991/1000)   missed eligible: 0   wrongly eligible: 5
  acs2023     decision  98.9% (989/1000)   missed eligible: 0   wrongly eligible: 6
  screening flags (USPSTF 2021): TP 250  FP 5  FN 2  TN 743  sens 0.992  spec 0.993  PPV 0.980
```

Status is scored exactly. Pack-years are correct when the value falls inside the accepted range in the answer key, with a tolerance of 0.5 for rounding, or when both the prediction and the answer key are empty. A quit date is correct when its year falls inside the accepted range. Eligibility is computed in code from the extracted values, by the same function that computes it from the answer key, so a system is never rewarded for guessing the decision without the data behind it. `--by pack_years_source`, `--by quit_date_source`, or `--by challenge_tags` breaks the fields down by how the note documents them, which is how we found where each system failed.

`python summarize.py` collects every run under `results/` into one table and writes `results/summary.md`.

## Reproducing the study

Nine runs, three systems on three conditions:

```bash
for c in basic messy complex; do
  python run_typesafe.py --condition $c
  python run_claude.py   --condition $c --model claude-sonnet-5 --effort low
  python run_claude.py   --condition $c --model claude-haiku-4-5
done
python summarize.py
```

That is 9,000 notes and cost us $44.62 in total, most of it Sonnet. It takes about 20 minutes of wall-clock time at the default concurrency, plus whatever the rate limits add. Our results were:

| Condition | System | Model | USPSTF | ACS | All 3 fields | N | Cost (US$) | Wall (s) | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| basic | typesafe | jev-1.13.0 | 991 | 989 | 986 | 1000 | 0.61 | 85 | 1.09 |
| basic | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 978 | 981 | 960 | 1000 | 3.37 | 119 | 1.71 |
| basic | claude-sonnet-5 | claude-sonnet-5 | 1000 | 999 | 999 | 1000 | 8.54 | 156 | 2.15 |
| messy | typesafe | jev-1.13.0 | 999 | 1000 | 997 | 1000 | 0.64 | 92 | 1.21 |
| messy | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 984 | 990 | 947 | 1000 | 4.21 | 148 | 2.16 |
| messy | claude-sonnet-5 | claude-sonnet-5 | 997 | 997 | 994 | 1000 | 10.75 | 183 | 2.64 |
| complex | typesafe | jev-1.13.0 | 944 | 963 | 810 | 1000 | 0.64 | 39 | 0.45 |
| complex | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 981 | 977 | 803 | 1000 | 4.43 | 160 | 2.41 |
| complex | claude-sonnet-5 | claude-sonnet-5 | 997 | 997 | 993 | 1000 | 11.42 | 212 | 3.20 |

USPSTF, ACS, and "all 3 fields" are counts of notes out of 1,000. Your numbers will not match ours exactly. Vendors update models behind the same name, prices change, and latency depends on load and on how far you are from the service. The prompt, the question set, and the scoring code here are the ones we ran, so differences should come from the models rather than from the harness.

## How the two systems are set up

**Claude.** One request per note. The system prompt is the codebook in `common.py`: the definitions of the three variables and the rules for resolving relative dates, conflicting statements, and histories that do not support a value. Structured outputs constrain the reply to a schema that asks for the evidence first (visit date, patient age, verbatim quotes, each smoking period with its rate and duration) and then for the three fields, which are the only ones scored. We ran Sonnet 5 with its effort setting at low and Haiku 4.5 with its defaults.

**Jev.** Jev makes typed judgments and does not generate text, so the work is split. Code finds the candidate numbers and dates in the note with regular expressions, Jev answers about 34 multiple-choice questions about the note in a single request (what the status is, how the note documents the exposure, which number is the stated pack-year figure, when the patient quit, and so on), and code does all of the arithmetic and date math from its answers. The questions are in `typesafe_questions.py`, which is also where you would start if you wanted to improve them; the vendor's guidance on what this class of model is good at is worth reading first.

## Files

| File | What it is |
|---|---|
| `common.py` | The shared output model, the codebook given to generative models, and benchmark I/O |
| `run_claude.py` | Runs a Claude model over one condition |
| `run_typesafe.py` | Runs Jev over one condition |
| `typesafe_questions.py` | The question set, the candidate finders, and the code that turns Jev's answers into the three fields |
| `lcs.py` | USPSTF 2021 and ACS 2023 eligibility, computed from the three fields |
| `score.py` | Scores one run against the answer key |
| `summarize.py` | Collects all runs into one table |
| `FROZEN.md` | The hashes recorded when the two pipelines were frozen |

## Citation

Wright A. Extracting smoking history from clinical notes for lung cancer screening decision support: a synthetic benchmark comparing a structured-judgment model with large language models. Preprint, 2026.

## License

MIT, in [LICENSE](LICENSE). The benchmark itself is released separately under CC BY 4.0.
