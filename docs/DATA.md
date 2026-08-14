# Dataset of record

Every number in `docs/RESULTS.md` was produced from this file, which is checked
into the repository:

```
data/adult/adult_full.csv
```

| | |
|---|---|
| Rows | 48,842 (plus one header row) |
| Columns | the 15 standard Adult columns; target named `Probability` |
| Size | 5,717,105 bytes |
| MD5 | `d934809c8d6118c5e30e55e17eea9da7` |
| SHA-256 | `11e7d89da4dacadf7bde576fdc216b3e8e8948efc704fcec58706f18a87fa569` |

It is the UCI Adult train and test splits concatenated in canonical order: rows
1–32,561 are `adult.data`, rows 32,562–48,842 are `adult.test`. The split boundary
is therefore **positional**, not a column, and `fbu.data.adult` restores it by row
index while asserting the total row count — a truncated or re-sorted file cannot
pass silently as a valid split (`tests/test_data.py::test_truncated_csv_is_rejected`).

`load_raw()` and `load_adult()` read this file by default, so the study runs with
no network access. Pass `source="uci"` to read `adult.data` / `adult.test` instead,
or `source="csv"` to require the bundled file.

## Equivalence to the raw UCI files

The bundled CSV and the files served by
`archive.ics.uci.edu/ml/machine-learning-databases/adult/` are the same data. All
48,842 rows × 15 columns were compared cell for cell after whitespace
normalisation, with zero differences and identical row order; the target
distribution is 37,155 `<=50K` to 11,687 `>50K` in both. Two tests keep this
honest whenever both sources are present, and skip otherwise:

- `test_bundled_csv_and_uci_pair_are_the_same_data` — the raw frames match.
- `test_both_sources_produce_identical_design_matrices` — equivalence survives
  preprocessing, so `X`, `y` and `s` are identical either way.

The differences are packaging only, and the loader normalises all of them:

| | bundled CSV | raw UCI pair |
|---|---|---|
| Files | one combined file | `adult.data` + `adult.test` |
| Header row | yes | no |
| Target column | `Probability` | unnamed (14th index), read as `income` |
| Test-split labels | periods already stripped | trailing `.` on every label |
| Split boundary | row order | separate files |

## Preprocessing applied (spec §3.1)

Unchanged by the source: `income` `">50K" → 1`; `sex` `Male → 1`; `?` kept as its
own category rather than dropping rows (`[D6]`); `fnlwgt` and `education` dropped;
remaining categoricals one-hot encoded drop-first; continuous features
standardised on train statistics only; `sex` retained in the feature matrix; all
FBU inputs computed on test predictions.

## Caveat

Adult is a 1994 census extract with documented measurement problems. Ding et al.
(2021) argue for retiring it in favour of `folktables`. It is adequate as a
development target for FBU and weak as a research claim about fairness in
income prediction.
