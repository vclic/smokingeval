# smokingeval

Code to run language models over the [smoking benchmark](https://github.com/vclic/smokingbenchmark) and score what they extract.

The benchmark is 3,000 synthetic outpatient notes with a known answer for each one: smoking status, pack-years, and quit date. This repository does two things with it. First, it sends each note to a model and writes the extracted values to a file. Second, it compares those values with the answer key and reports how often each field was right, what lung cancer screening (LCS) decision would follow from them, what the run cost, and how long it took.

Five systems were compared in our study: TypeSafe Jev 1.13, Claude Haiku 4.5, Claude Sonnet 5, GPT-6 Luna, and GPT-6 Sol. All of them are given the same note text, the same prompt where a prompt applies, and the same output schema, so the comparison is of the systems and not of the output formats. Adding another system means writing one more runner that produces `predictions.jsonl` in the same shape; everything downstream works unchanged.

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
export OPENAI_API_KEY=...       # for GPT-6
```

If you keep keys in `~/.config/smoking-bench/.env`, that file is read as well. An organization-level Anthropic key also needs `SMOKING_BENCH_ANTHROPIC_WORKSPACE_ID`, and the `SMOKING_BENCH_` prefixed names are used first if you would rather not set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

If your network inspects TLS traffic, point `SSL_CERT_FILE` at a bundle that includes your organization's certificate authority; `run_openai.py` uses it and allows that CA to act as a trust anchor.

## Running a system

Each run covers one of the three conditions (`basic`, `messy`, `complex`):

```bash
python run_typesafe.py --condition basic
python run_claude.py   --condition basic --model claude-sonnet-5 --effort low --cache
python run_claude.py   --condition basic --model claude-haiku-4-5
python run_openai.py   --condition basic --model gpt-6-sol  --effort low
python run_openai.py   --condition basic --model gpt-6-luna --effort none
```

`--cache` marks the system prompt and schema as cacheable, which bills the repeated prefix at a tenth of the input price. It cut Sonnet's cost by 47%. It does nothing for Haiku 4.5, whose minimum cacheable prefix (4,096 tokens) is longer than our prompt and schema. The OpenAI models cache repeated prefixes automatically.

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
----------------------------------------------------------------------
  screening flags (a flag is raised when the extracted values make the patient eligible)
  uspstf2021  TP  250  FP   5  FN   2  TN  743  sens 0.992  spec 0.993  PPV 0.980  NPV 0.997
  acs2023     TP  315  FP   6  FN   2  TN  677  sens 0.994  spec 0.991  PPV 0.981  NPV 0.997
```

Status is scored exactly. Pack-years are correct when the value falls inside the accepted range in the answer key, with a tolerance of 0.5 for rounding, or when both the prediction and the answer key are empty. A quit date is correct when its year falls inside the accepted range. Eligibility is computed in code from the extracted values, by the same function that computes it from the answer key, so a system is never rewarded for guessing the decision without the data behind it. `--by pack_years_source`, `--by quit_date_source`, or `--by challenge_tags` breaks the fields down by how the note documents them, which is how we found where each system failed.

`python summarize.py` collects every run under `results/` into one table and writes `results/summary.md`.

## Reproducing the study

Fifteen runs, five systems on three conditions:

```bash
for c in basic messy complex; do
  python run_typesafe.py --condition $c
  python run_claude.py   --condition $c --model claude-sonnet-5 --effort low --cache
  python run_claude.py   --condition $c --model claude-haiku-4-5
  python run_openai.py   --condition $c --model gpt-6-sol  --effort low
  python run_openai.py   --condition $c --model gpt-6-luna --effort none
done
python summarize.py
```

That is 15,000 notes and cost us $41.21 in total, most of it Sonnet and Haiku. It takes about an hour of wall-clock time at the default concurrency, plus whatever the rate limits add. Our results were:

| Condition | System | Model | USPSTF | ACS | All 3 fields | Cost (US$) | Wall (s) | Median latency (s) |
|---|---|---|---|---|---|---|---|---|
| basic | typesafe | jev-1.13.0 | 991 | 989 | 986 | 0.61 | 85 | 1.09 |
| basic | gpt-6-luna | gpt-6-luna | 997 | 997 | 994 | 0.12 | 174 | 2.30 |
| basic | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 978 | 981 | 960 | 3.37 | 119 | 1.71 |
| basic | gpt-6-sol | gpt-6-sol | 1000 | 1000 | 1000 | 2.44 | 186 | 2.51 |
| basic | claude-sonnet-5 | claude-sonnet-5 | 1000 | 998 | 998 | 3.76 | 170 | 2.46 |
| messy | typesafe | jev-1.13.0 | 999 | 1000 | 997 | 0.64 | 92 | 1.21 |
| messy | gpt-6-luna | gpt-6-luna | 988 | 990 | 984 | 0.19 | 206 | 3.08 |
| messy | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 984 | 990 | 947 | 4.21 | 148 | 2.16 |
| messy | gpt-6-sol | gpt-6-sol | 998 | 998 | 998 | 3.80 | 202 | 2.94 |
| messy | claude-sonnet-5 | claude-sonnet-5 | 996 | 996 | 994 | 5.95 | 194 | 2.90 |
| complex | typesafe | jev-1.13.0 | 944 | 963 | 810 | 0.64 | 39 | 0.45 |
| complex | gpt-6-luna | gpt-6-luna | 985 | 983 | 928 | 0.21 | 217 | 3.40 |
| complex | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 981 | 977 | 803 | 4.43 | 160 | 2.41 |
| complex | gpt-6-sol | gpt-6-sol | 996 | 997 | 996 | 4.22 | 225 | 3.39 |
| complex | claude-sonnet-5 | claude-sonnet-5 | 998 | 999 | 995 | 6.61 | 224 | 3.50 |

USPSTF, ACS, and "all 3 fields" are counts of notes out of 1,000. The Sonnet rows use prompt caching; the same runs without it are in `study_results` as `claude-sonnet-5-nocache` and cost $8.54, $10.75 and $11.42 per 1,000 notes.

The runs behind that table are in `study_results/`, one folder per system and condition, holding the predictions and the run metadata exactly as they came out of the three services. `python summarize.py study_results` rebuilds the table from them, and `python score.py study_results/complex/typesafe --condition complex --by pack_years_source` is how we looked at where a system went wrong.

Your numbers will not match ours exactly. Vendors update models behind the same name, prices change, and latency depends on load and on how far you are from the service. The prompt, the question set, and the scoring code here are the ones we ran, so differences should come from the models rather than from the harness.

## How the two systems are set up

**Claude.** One request per note. The system prompt is the codebook in `common.py`: the definitions of the three variables and the rules for resolving relative dates, conflicting statements, and histories that do not support a value. Structured outputs constrain the reply to a schema that asks for the evidence first (visit date, patient age, verbatim quotes, each smoking period with its rate and duration) and then for the three fields, which are the only ones scored. We ran Sonnet 5 with its effort setting at low and Haiku 4.5 with its defaults.

**GPT-6 Sol and GPT-6 Luna.** Same prompt and same schema as Claude, through the OpenAI SDK's structured outputs. Both are reasoning models whose effort can be set from none to max; we ran Sol at low, matching Sonnet 5, and Luna with reasoning off, matching Haiku 4.5.

**Jev.** Jev makes typed judgments and does not generate text, so the work is split. Code finds the candidate numbers and dates in the note with regular expressions, Jev answers about 34 multiple-choice questions about the note in a single request (what the status is, how the note documents the exposure, which number is the stated pack-year figure, when the patient quit, and so on), and code does all of the arithmetic and date math from its answers. The questions are in `typesafe_questions.py`, which is also where you would start if you wanted to improve them; the vendor's guidance on what this class of model is good at is worth reading first.

## Files

| File | What it is |
|---|---|
| `common.py` | The shared output model, the codebook given to generative models, and benchmark I/O |
| `run_claude.py` | Runs a Claude model over one condition |
| `run_typesafe.py` | Runs Jev over one condition |
| `run_openai.py` | Runs a GPT-6 model over one condition |
| `typesafe_questions.py` | The question set, the candidate finders, and the code that turns Jev's answers into the three fields |
| `lcs.py` | USPSTF 2021 and ACS 2023 eligibility, computed from the three fields |
| `score.py` | Scores one run against the answer key |
| `summarize.py` | Collects all runs into one table |
| `FROZEN.md` | The hashes recorded when the two pipelines were frozen |
| `study_results/` | The runs from the study: predictions and run metadata, including the uncached Sonnet runs |

## Citation

Wright A, Liu S, Wright A. Extracting smoking history from clinical notes for lung cancer screening decision support: comparing a structured-judgment model with general-purpose large language models. Preprint, 2026.

## License

MIT, in [LICENSE](LICENSE). The benchmark itself is released separately under CC BY 4.0.
