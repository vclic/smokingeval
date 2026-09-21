# Frozen pipelines

Both extraction pipelines were frozen on 2026-09-19 at 02:14 UTC, before the messy and
messy-with-complex-arithmetic conditions were designed or generated. Only a 40-note development
set, generated from a different random seed, was used while writing them. Neither system could
therefore be tuned to the harder notes.

Settings as run:

- Claude: `claude-sonnet-5` with effort `low`, and `claude-haiku-4-5` with its defaults; the
  codebook and schema now in `common.py`.
- Jev: `jev-1.13.0`, with the question set and composition code now in `typesafe_questions.py`.

SHA-256 of the files at the moment they were frozen, under the paths they had in the working
repository:

```
d9333455f5cab343691760b617f04eca0b7ceca99b09e4823114c6af0ce7fb35  bench/common.py
b3913162e8c33f90a8c867e060bdb0645cc97e74c1869beff25016046c032848  bench/run_claude.py
ad3ba1a71a286da547f15f67d42b7da2addea1781321f75390a041bbe4cd8e38  bench/run_typesafe.py
57dbfbd3a166ebc856080df3446612a377290088b9a92b95f8349c1e021e6390  bench/typesafe_v2.py
5cddab2d61be7ae0a7cd1cd8e447b0e3534f97ffd01b3099f174e018f532f1bc  bench/lcs.py
```

The files in this repository are the same prompt, schema, question set, composition code, and
eligibility logic, rearranged for release: the paths changed, an earlier pipeline version that the
study did not use was dropped, and the runners now read the benchmark from its own repository. The
hashes above will therefore not match the files here. They are recorded so that the frozen
versions can be identified.
