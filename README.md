# FBU — Fairness Bonded Utility

An implementation of the FBU metric from Wang et al., *Preventing Discriminatory
Decision-making in Evolving Data Streams*, ACM FAccT 2023 (§5.1, Figures 3 and 6),
exercised on UCI Adult with a linear or logistic base model.

FBU is a **post-hoc comparison metric**. Given the prediction arrays of several
competing bias-mitigation techniques, it reports for each one a distribution over
five effectiveness regions plus a trade-off area score. It depends on no
particular model, and nothing here touches streaming infrastructure.

- `docs/FBU_IMPLEMENTATION.md` — the authoritative specification; it resolves
  seven places where the paper is under-specified.
- `docs/DEVIATIONS.md` — every choice made beyond what the paper states (`[D1]`–`[D9]`).
- `docs/RESULTS.md` — measured Adult numbers and the §6.1 validation checks.
- `docs/DATA.md` — the dataset of record, its checksums, and its equivalence to
  the raw UCI files.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No download step: the dataset used for every reported number is checked in at
`data/adult/adult_full.csv` (48,842 rows, train and test concatenated in canonical
order — see `docs/DATA.md`). To read the raw UCI pair instead, fetch it with
`python -c "from fbu.data.adult import download; download()"` and pass
`load_adult(source="uci")`; the two are verified to be the same data.

Dependencies are numpy, pandas, scikit-learn and matplotlib, plus pytest and
hypothesis for tests. `aif360` and `fairlearn` are deliberately absent: their
metric sign conventions differ and would silently invert results.

## Use

```python
from fbu import evaluate, plot_tradeoff
from fbu.data.adult import load_adult
from fbu.techniques import ALL_TECHNIQUES, fit_original

data = load_adult()
original = fit_original(data)                                    # logistic regression
techniques = [technique(data) for technique in ALL_TECHNIQUES.values()]

report = evaluate(original, techniques, n_runs=5, n_seeds=20)
print(report.summary())          # region percentages + area statistics per technique
print(report.to_dataframe())     # one row per case of the grid
axes = plot_tradeoff(report, "balanced_accuracy", "spd")   # Figure 3(b)
```

The two scripts in `examples/` reproduce everything in `docs/RESULTS.md`,
writing figures and CSV/JSON artefacts to `outputs/<scorer>/`.

## How it works

The fairness axis is always `fairness_score = 1 − |metric|`, so higher is fairer
(`[D1]`; SPD and EOD are lower-is-better and cannot go on that axis raw). The
performance axis is balanced accuracy or recall — never raw accuracy, F1 or
precision (`[D8]`).

The **baseline curve** is built by replacing a random `p%` of the original
model's predictions with its majority label, for `p = 10…100`, averaged over 20
seeds per ratio (`[D3]`). `M_100` is the constant classifier: perfectly fair,
balanced accuracy 0.5.

Regions 1, 3 and 5 are **quadrants around the original model point**. Regions 2
and 4 are the trade-off quadrant split by the **baseline curve**. These are two
different reference objects and collapsing them is a bug. For region-2 points the
trade-off area is the closed integral between the point and the curve, normalised
by the largest attainable area (`[D2]`).

## Honest framing — read before quoting any number

- **The baseline is randomised.** Two FBU runs on identical predictions can put a
  point near the curve in different regions. Always report percentages with a
  seed band; `fbu.seed_sensitivity` produces one. On Adult the four real
  techniques are stable across 5, 20 and 100 seeds, but the `M_30` control — which
  sits *on* the curve — splits 62/33/5 across regions 2/4/5.
- **Three of the five regions are just quadrants.** Only the region-2/4 split uses
  the pseudo-model machinery at all, so the curve carries less of FBU's
  information than the paper's §5.1 implies.
- **Areas are not comparable across settings** without the `A_norm`
  normalisation, because `A_max` depends on curve shape, which depends on base
  rate and class balance. They are also not comparable *across fairness metrics*:
  aggregating them mixes objectives (see `docs/RESULTS.md` §4).
- **Figure 6 will not replicate numerically.** Ordering and shape do.
- **Adult is a 1994 census extract with documented measurement problems**;
  Ding et al. (2021) argue for retiring it in favour of `folktables`. Adequate as
  a development target, weak as a research claim.
- **The base model is logistic regression** (spec §6.4, resolved). The linear
  probability model — OLS on {0,1} labels, thresholded — stays available behind
  `LPMScorer` / `--scorer lpm` and is reported alongside in `docs/RESULTS.md` §6.
  Every qualitative conclusion agrees between the two; the numbers do not.

## Out of scope this phase

Concept and fairness drift detection (ADWIN), sliding windows, prequential
evaluation, `minSize`, FS² itself, and the four streaming baselines. Adult is
static and unordered. The cumulative fairness metrics of paper Equations 2–3 are
implemented and tested for that future phase but refused by the metric registry
here (`[D7]`), and the `Predictions` contract is shaped so a prequential
evaluation drops into the same code path.
