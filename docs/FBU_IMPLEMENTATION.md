# FBU — Fairness Bonded Utility
## Implementation Specification and Build Plan

**Version:** 2.0 — supersedes `FBU_IMPLEMENTATION_PLAN.md` and `FBU_ADULT_ADDENDUM.md`. Delete those; this is the single authoritative document.

**Source:** Wang et al., *Preventing Discriminatory Decision-making in Evolving Data Streams*, ACM FAccT 2023, §5.1, Figure 3, Figure 6.

**Target for this phase:** UCI Adult, linear (or logistic) base model, offline. Streaming is out of scope — see §1.2.

---

# Part I — Context

## 1.1 What is being built

FBU is a **post-hoc evaluation metric**. It takes prediction arrays from a set of competing bias-mitigation techniques and returns, per technique, a distribution over five "effectiveness" regions plus a trade-off area score. It has no dependency on any particular model, on FS², or on streaming infrastructure.

Two facts that govern the whole build:

1. **FBU is a comparison metric.** A base model alone yields the origin point and the baseline curve, with nothing plotted against it. The deliverable is empty without a set of competing techniques. §4 specifies the minimum set.
2. **The paper under-specifies FBU in seven places.** This document resolves each one. Resolutions that go beyond what the paper states are tagged `[D#]` and registered in §9. An exact numerical replication of Figure 6 (79% / 32% / 25% / 58% / 31%) is **not** an achievable acceptance criterion; ordering and shape are.

## 1.2 Scope

**In scope:** the FBU metric, its baseline construction, region classification, area quantification, plots, and enough mitigation techniques to exercise it on Adult.

**Out of scope this phase:** concept drift and fairness drift detection (ADWIN), sliding windows, prequential evaluation, `minSize`, FS² itself, and the four streaming baselines (FAHT, OSBoost, FABBOO, C-SMOTE). Adult is static and unordered, so none of it applies. Tables 1–2 and Figure 7 of the paper are not replicable here and are not targets.

The `Predictions` contract in §2.1 is deliberately shaped so that a streaming prequential evaluation drops into the same code path later, with no change to the FBU core.

## 1.3 Gaps in the paper and how they are resolved

| # | Gap | Resolution | Tag |
|---|-----|------------|-----|
| G1 | Prose describing Fig 3(b) axes actually describes Fig 3(a). | Fig 3(b) is **x = fairness, y = performance**, per its own axis labels. Fig 3(a) is a separate diagnostic (x = replacement ratio). | — |
| G2 | SPD/CEOD are lower-is-better, but the fairness axis must be higher-is-better (Fig 3(a) shows fairness rising with replacement ratio). | `fairness_score = 1 − |metric|`, range [0,1]. | `[D1]` |
| G3 | Regions 1/3/5 defined against the original model, 2/4 against the baseline curve — reads as an inconsistency. | It is not one. See §2.4. Do not collapse the two reference objects. | — |
| G4 | "Area enclosed by the fairness-performance points and the baseline" — the enclosing boundary is never given. | Closed-region integral, §2.5. | `[D2]` |
| G5 | Pseudo-model construction is randomised; no seed or repetition policy. | Average over `n_seeds` (default 20) per ratio. | `[D3]` |
| G6 | "Number of cases **per region** is n_r × n_t × n_f × p_m" — that product is the total. | Treat as total; per-region values are tallies. Drop `n_t`. | `[D4]` |
| G7 | Equation 2 conditions both terms on `S = s`; one must be `S = s̄`. | Typo. Implement as privileged-minus-unprivileged. | — |

---

# Part II — Formal specification

Implement exactly this. Where the code makes a choice not stated in the paper, carry a `# [D#]` comment.

## 2.1 Data contract

```python
@dataclass(frozen=True)
class Predictions:
    y_true: np.ndarray   # (n,), {0,1}; 1 = favorable outcome
    y_pred: np.ndarray   # (n,), {0,1}
    s:      np.ndarray   # (n,), {0,1}; 1 = privileged
    score:  np.ndarray | None  # (n,), continuous; required by threshold-based techniques
    name:   str
```

All techniques must return predictions over the **same test indices in the same order**. Enforce this with a shape-and-index assertion in the orchestrator; a silent misalignment produces plausible-looking garbage.

## 2.2 Metrics

```python
PerformanceMetric = Callable[[np.ndarray, np.ndarray], float]              # higher better
FairnessMetric    = Callable[[np.ndarray, np.ndarray, np.ndarray], float]  # signed, 0 = fair
```

**Performance metrics (use exactly these two):**

```
balanced_accuracy = (TPR + TNR) / 2
recall            = TPR = TP / (TP + FN)
```

Do **not** include raw accuracy. Adult is ~76% negative, so the all-negative constant classifier scores 0.76 — which would place `M_100` *above* the real model, invert the baseline curve, and break region logic outright.

Do **not** include F1 or precision. Both divide by predicted positives, which is exactly zero at `M_100` (§2.3). Recall is safe: it divides by true positives, which is never zero here.

**Fairness metrics (static forms — use these):**

```
SPD = P(ŷ=1 | S=1) − P(ŷ=1 | S=0)
EOD = TPR(S=1) − TPR(S=0)
```

**Cumulative forms (implement, test, but do not use this phase).** Equations 2–3 are decay-weighted over stream order. Adult has no order; imposing a shuffle and applying λ yields a seed-dependent number with no meaning. Implement anyway for the future streaming phase:

```
p_t = (# ŷ=1 & S=1, up to t) / (# S=1, up to t)
u_t = (# ŷ=1 & S=0, up to t) / (# S=0, up to t)
CSPD_t = (1 − λ)(p_t − u_t) + λ · CSPD_{t−1},     CSPD_0 = 0
```

`cum_eod` is identical with every count conditioned on `y_true == 1`. If a denominator is 0, carry the previous value forward. Default λ = 0.5 (paper §6.3.2 finds ≥0.5 stable).

**Fairness score (the plotted quantity):**

```
fairness_score = 1 − abs(metric)      # [D1]
```

Any code path that puts a raw signed SPD/EOD on the fairness axis is a bug.

## 2.3 Pseudo-models and the baseline curve

Let `L*` = the majority label among the **original model's** predictions.

> On Adult, `L* = 0` (`≤50K`) for any reasonable base model, since the base rate is ~24% and a 0.5-thresholded linear or logistic model predicts positive less often still. Assert this in a test rather than assuming it.

For `p ∈ {10, 20, …, 100}%`, pseudo-model `M_p` replaces a uniformly random `p%` of the original predictions with `L*`. `M_100` is the constant classifier `L*`. Repeat for `n_seeds` seeds; the plotted point for `M_p` is the **mean** `(fairness_score, performance)` across seeds `[D3]`.

Anchors that must hold at `M_100` on Adult:

| Quantity | Value |
|---|---|
| SPD, EOD | 0 → fairness_score = 1.0 |
| balanced accuracy | 0.5 |
| recall | 0.0 |

**Baseline curve** `g`: piecewise-linear interpolation through the 11 points `M_ori, M_10, …, M_100`, sorted ascending by `fairness_score`. `g` is defined on `[f_ori, 1.0]` and should be non-increasing. If it is not, raise `n_seeds`; an optional upper monotone envelope exists behind a flag, default **off** — report raw `[D5]`.

## 2.4 Region classification

Given technique point `P = (f_p, y_p)`, original model `M_ori = (f_o, y_o)`, curve `g`, tolerance `ε = 1e-9`:

```python
if   f_p >= f_o - eps and y_p >= y_o - eps:   region = 1  # Jointly advantageous
elif f_p <  f_o        and y_p >= y_o - eps:  region = 3  # Reversed
elif f_p <  f_o        and y_p <  y_o:        region = 5  # Jointly disadvantageous
else:                                          # fairer, less accurate — the trade-off quadrant
    region = 2 if y_p >= g(f_p) - eps else 4   # Impressive / Deficient
```

Regions 1, 3 and 5 are quadrants around `M_ori`. **Only regions 2 and 4 use the curve** — the curve exists solely to split the trade-off quadrant. This is the only reading consistent with all of: the prose ("region 5 … decreases both … compared to the original model"), the numeral positions in Fig 3(b) (3 top-left, 1 top-centre, 2 right, 5 mid-left, 4 bottom-left), and the caption ("effective if it lies above the line"). Do not simplify it to a single reference object.

`M_ori` itself classifies as region 1 (it ties both thresholds). Document this; it is a harmless degenerate case.

## 2.5 Trade-off area (region 2 only)

For `P = (f_p, y_p)` in region 2, the enclosed region is bounded above by `y = y_p`, right by `x = f_p`, and below-left by the curve `[D2]`:

```
f_a = the unique f in [f_o, f_p] with g(f) = y_p      # g is non-increasing, so unique
A   = ∫_{f_a}^{f_p} ( y_p − g(f) ) df
```

Compute analytically over the piecewise-linear segments — no numerical quadrature. Normalise for cross-setting comparability:

```
A_max  = ∫_{f_o}^{1.0} ( y_o − g(f) ) df
A_norm = A / A_max
```

Larger area = better trade-off, per the paper. Expose the area function behind a `TradeoffArea` protocol so an alternative (e.g. perpendicular distance to the curve) can be swapped in for a sensitivity check.

## 2.6 Output

Per technique, tally region assignments across the full case grid:

```
cases = n_runs × n_fairness_metrics × n_performance_metrics × n_datasets     # [D4]
```

`FBUResult` carries: five percentages summing to 100, the case count, and mean/median/std of `A_norm` over that technique's region-2 cases. `n_t` from the paper's formula is dropped — a technique's own case count cannot depend on how many competitors it faces; including it rescales every technique identically.

---

# Part III — Data and models

## 3.1 Adult dataset

Source: UCI Adult, `adult.data` (32,561) + `adult.test` (16,281) = 48,842 rows. [certain]

```
Target      : income  ">50K" -> 1 (favorable),  "<=50K" -> 0
Sensitive   : sex     Male -> 1 (privileged),   Female -> 0
Base rate   : ~24% positive overall                          [likely]
              ~31% men, ~11% women                           [likely]
```

Fix preprocessing once in `src/fbu/data/adult.py` and never change it mid-study:

1. `adult.test` has a stray header line and a trailing period on every label — strip both.
2. Missing values appear as `" ?"` in `workclass`, `occupation`, `native-country`. **Encode `?` as its own category rather than dropping rows** `[D6]` — dropping ~7% of rows is itself an intervention with a differential rate across groups, which would contaminate the fairness measurement.
3. Drop `fnlwgt` (a survey sampling weight, not a feature). `education` and `education-num` are redundant — keep `education-num`.
4. One-hot encode remaining categoricals (drop-first); standardise continuous features.
5. Keep `sex` in the feature matrix — the fairness-unaware technique in §4 needs the contrast.
6. Use the canonical train/test split. All FBU inputs are computed on **test** predictions only.

**README caveat, one line:** Adult is a 1994 census extract with documented measurement problems; Ding et al. (2021) argue for retiring it in favour of `folktables`. Adequate as a development target, weak as a research claim. [likely]

## 3.2 Base model

The instruction was "linear regression." On a binary target that means a **Linear Probability Model** — OLS on {0,1} labels, thresholded. This is legitimate but unusual; "logistic regression" is the conventional choice and may be what was meant. **Confirm before committing to results.** Support both behind a flag so the answer costs nothing:

```python
class BinaryScorer(Protocol):
    def fit(self, X, y, sample_weight=None) -> "BinaryScorer": ...
    def score(self, X) -> np.ndarray: ...                       # continuous, any range
    def predict(self, X, threshold: float = 0.5) -> np.ndarray: # {0,1}
```

- `LPMScorer` — wraps `sklearn.linear_model.LinearRegression`. Scores are unbounded; values outside [0,1] are expected and fine. Threshold 0.5. `lstsq` handles one-hot rank deficiency.
- `LogitScorer` — wraps `LogisticRegression(max_iter=1000)`; `score` = `predict_proba[:, 1]`.

FBU consumes only the thresholded `{0,1}` array, so **the choice does not affect the FBU spec at all.** Both must accept `sample_weight` for reweighing.

## 3.3 Expected values — all [guessing], verify rather than trust

LPM at threshold 0.5 on Adult test: accuracy ~0.83, balanced accuracy ~0.72–0.75, recall ~0.45–0.55, SPD (sex) ~0.15–0.19, EOD ~0.08–0.12. Logistic lands nearby with slightly better recall.

If SPD comes out near 0 or above 0.35, the group coding or threshold is wrong — stop and check before proceeding.

Baseline curve should run from roughly `(0.82, 0.74)` at `M_ori` to exactly `(1.00, 0.50)` at `M_100`, monotone decreasing in performance.

---

# Part IV — Comparison set

Four techniques, all pre- or post-processing, so they compose with any base model. Expected placements are hypotheses to test, not assertions.

| Technique | Mechanism | Expected region | Why included |
|---|---|---|---|
| **Fairness-unaware** | Drop `sex` from the feature matrix; refit. | 3 or 5 | The classic negative result — SPD barely moves because of proxies (`relationship`, `marital-status`, `hours-per-week`). |
| **Reweighing** (Kamiran & Calders 2012, ref [26]) | `w(s,y) = P(S=s)·P(Y=y) / P(S=s, Y=y)`; pass as `sample_weight`. | 1 or 2 | ~30 LOC, strong baseline, works with both scorers. |
| **Massaging** (Kamiran & Calders 2009, ref [25]) | Rank by score; promote unprivileged negatives / demote privileged positives nearest the boundary until SPD ≈ 0; refit. | 2 | Uses the continuous score the base model gives free. |
| **Group-wise thresholds** | Per-group thresholds chosen to equalise TPR on train. | 2, largest area | Directly optimises EOD; should be the area leader. |

**Plus one control:** feed `M_30` back in as a fake technique. It must land in region 2 with `A ≈ 0` (within seed noise). This single check catches sign errors, axis-direction errors, and interpolation errors simultaneously. Keep it in the permanent test suite.

---

# Part V — Repository and build

## 5.1 Layout

```
fbu/
  pyproject.toml
  README.md
  docs/
    FBU_IMPLEMENTATION.md      # this document — authoritative
    DEVIATIONS.md              # generated from §9
  src/fbu/
    __init__.py
    types.py                   # Predictions, FBUResult, protocols
    metrics/
      performance.py           # balanced_accuracy, recall
      fairness.py              # spd, eod, cum_spd, cum_eod, fairness_score
    data/
      adult.py                 # load + preprocess, cached
    models/
      scorers.py               # LPMScorer, LogitScorer
    techniques/
      unaware.py  reweighing.py  massaging.py  group_threshold.py
    pseudo.py                  # pseudo-model generation
    baseline.py                # curve construction, interpolation, envelope
    regions.py                 # 5-region classifier
    area.py                    # trade-off area + normalisation
    fbu.py                     # orchestration -> FBUResult
    plotting.py                # Fig 3(a), Fig 3(b), Fig 6
    io.py
  tests/
    test_metrics.py  test_pseudo.py  test_baseline.py  test_regions.py
    test_area.py  test_data.py  test_techniques.py  test_fbu_end_to_end.py
    fixtures/golden_cases.py
  examples/
    01_adult_baseline_curve.py
    02_adult_full_comparison.py
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pytest`, `hypothesis`. Nothing else. Do **not** add `aif360` or `fairlearn` — their metric sign conventions differ and will silently invert results.

## 5.2 Milestones

One agent session each. Do not start N+1 until N's tests pass.

**M0 — Scaffold.** Package skeleton, `pyproject.toml`, `types.py`, pytest wired, `make test` green on placeholders.

**M1 — Metrics.** All of §2.2.
*Accept (hand-computable golden values, shown in each test docstring):*
- Perfect classifier → balanced accuracy 1.0, recall 1.0.
- Constant-positive → SPD 0, EOD 0, balanced accuracy 0.5, recall 1.0.
- Constant-negative → SPD 0, EOD 0, balanced accuracy 0.5, recall 0.0.
- Fully segregated (privileged all 1, unprivileged all 0) → SPD = 1.0.
- `cum_spd` with λ=0 reduces to the instantaneous rate difference; λ→1 drives the final value toward 0 (zero-initialised prefix dominates). Assert both.

**M2 — Data and base model.**
*Accept:* 48,842 rows loaded; positive rate within 0.005 of 0.24; male/female positive-rate gap ~0.20; test accuracy ≥ 0.80 and balanced accuracy ≥ 0.70 for both scorers; `sample_weight` demonstrably changes the fit.

**M3 — Pseudo-models and baseline curve.**
*Accept:* `M_100` reproduces the constant-`L*` classifier exactly for every seed; `L* == 0` on Adult (assert, don't assume); fairness_score monotone non-decreasing in `p` for SPD (tolerate one inversion ≤ 0.01 — more means a bug or too few seeds); `g(f_i) == y_i` at every knot; `M_100` hits the three anchors in §2.3 exactly.

**M4 — Region classifier.**
*Accept:* fixture with one point in each of the five regions plus four boundary points (on `f_o`, on `y_o`, on the curve, and at `M_ori`). `M_ori` → region 1.

**M5 — Trade-off area.**
*Accept:* point on the curve → `A = 0`; point at `(1.0, y_o)` → `A_norm = 1.0`; analytic check against a straight-line baseline where the area is a hand-computable trapezoid; `hypothesis` property — `A` is monotone increasing in both `y_p` and `f_p`.

**M6 — Techniques.**
*Accept:* every technique returns `Predictions` over identical test indices; reweighing reduces `|SPD|` versus base; group-wise thresholds drive `|EOD|` below 0.02; fairness-unaware moves `|SPD|` by less than 0.05 (the expected negative result — if it goes to zero, `sex` was not actually the driver and something is wrong).

**M7 — Orchestration.** `fbu.evaluate(original, techniques, perf_metrics, fair_metrics, n_runs, n_seeds) -> FBUResult`, plus `to_dataframe()`.
*Accept:* the `M_30` control lands in region 2 with `A_norm < 0.02`; a synthetically constructed Pareto-dominating technique lands in region 1 for every metric pair; a synthetically constructed accuracy-up/fairness-down technique lands in region 3.

**M8 — Plots.** Fig 3(a) (fairness and performance vs replacement ratio), Fig 3(b) (curve, technique points, shaded region-2 areas), Fig 6 (stacked bar per technique).
*Accept:* visual review; in 3(a) fairness rises and performance falls monotonically.

## 5.3 Agent configuration

`.cursorrules` / `AGENTS.md` at repo root:

```
This repo implements FBU (Fairness Bonded Utility), Wang et al., FAccT 2023.

- docs/FBU_IMPLEMENTATION.md is authoritative. It resolves ambiguities in the
  paper. Do NOT re-derive definitions from the PDF.
- The fairness axis is ALWAYS higher-is-better: fairness_score = 1 - abs(metric).
  Any code putting a raw signed SPD/EOD on that axis is a bug.
- Regions 1/3/5 are relative to the ORIGINAL MODEL point.
  Regions 2/4 are relative to the BASELINE CURVE.
  Do not collapse these into one reference.
- Never use accuracy, F1, or precision as a performance metric. See spec 2.2.
- No new dependencies without asking.
  numpy / pandas / scikit-learn / matplotlib / pytest / hypothesis only.
- Any modelling choice not stated in the paper carries a `# [D#]` comment
  referencing docs/DEVIATIONS.md.
- Tests first. Golden values must be hand-computable and derived in the docstring.
- Type hints on all public APIs. No bare `Any`.
```

Per-milestone prompt:

```
Implement milestone M<N> from docs/FBU_IMPLEMENTATION.md §5.2.

Before writing any code:
1. Read the spec sections it references.
2. List the acceptance tests you will write, the exact expected values, and how
   you derived each by hand. Stop and wait for my confirmation of those values.

Then:
3. Write the test file first. Run it; confirm it fails for the right reason.
4. Implement the module.
5. Run the full suite. Report every test you modified after writing it, and why.

Do not touch modules outside M<N>. Flag any refactor of earlier milestones
separately rather than doing it inline.
```

Step 2 is the load-bearing one. The dominant failure mode on metric code is an agent writing tests that assert whatever the implementation happened to produce.

---

# Part VI — Validation and honesty

## 6.1 Interpretation checks (unit tests will not catch these)

1. **Degenerate anchor.** The constant classifier must show fairness_score 1.0 and balanced accuracy 0.5. Disagreement means the fairness sign or normalisation is wrong.
2. **Baseline membership.** `M_30` fed back as a technique → region 2, `A ≈ 0`. Strongest single end-to-end check.
3. **Pareto dominance.** A synthetic technique beating the original on both axes → region 1 for every metric pair.
4. **Seed sensitivity.** Run `n_seeds ∈ {5, 20, 100}`; report the spread of region percentages. If assignments flip between budgets, the default is too low — raise and document.
5. **Ordinal plausibility.** Group-wise thresholds should out-area massaging, which should out-area fairness-unaware. Ordering, not exact percentages, is the pass condition.

## 6.2 Known weaknesses — report these, don't bury them

- **The baseline is randomised.** Two FBU runs on identical predictions can assign different regions to a point near the curve. Always report region percentages with a seed-variance band.
- **Three of five regions are just quadrants.** Only the region-2/4 split uses the pseudo-model machinery. The curve carries less of FBU's information than §5.1 of the paper implies. State this in the README; it is the honest framing.
- **Area is not comparable across settings** without the `A_norm` normalisation, because `A_max` depends on curve shape, which depends on base rate and class balance.
- **Figure 6 will not replicate numerically**, for the reasons in §1.3. Report ordering, and list the deviations.

## 6.3 Deviations register (`docs/DEVIATIONS.md`)

| Tag | Decision | Rationale |
|---|---|---|
| D1 | `fairness_score = 1 − |metric|` | Baseline curve requires a higher-is-better fairness axis; paper's metrics are lower-is-better. |
| D2 | Area = `∫ (y_p − g(f)) df` over `[f_a, f_p]`, normalised by `A_max` | Paper names the area but never its boundary. Pluggable. |
| D3 | `n_seeds = 20`, mean across seeds | Pseudo-model construction is randomised; paper gives no repetition policy. |
| D4 | Drop `n_t` from the case-count formula | A technique's case count cannot depend on its competitors. |
| D5 | Monotone envelope available, default off | Paper assumes a smooth curve; real curves are noisy. Report raw. |
| D6 | Encode `?` as a category rather than dropping rows | Dropping ~7% with differential group rates contaminates the fairness measurement. |
| D7 | Static SPD/EOD used instead of cumulative Eq. 2–3 | Adult has no temporal order; decay over an arbitrary shuffle is meaningless. |
| D8 | F1/precision and raw accuracy excluded from performance metrics | Undefined at `M_100` and majority-class-dominated respectively. |

## 6.4 Open question blocking final results

**Linear regression or logistic regression?** §3.2 supports both; the FBU spec is unaffected either way, but the reported numbers are not. Resolve before anything is written up.

> **Resolved: logistic regression.** It is the default for every scorer factory,
> technique and example script; the linear probability model remains available
> behind `--scorer lpm` and is reported alongside in `docs/RESULTS.md` §6. Every
> qualitative conclusion agrees between the two; the numbers do not.
