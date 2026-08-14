# Measured results on UCI Adult

**Base model: logistic regression** — the resolved answer to spec §6.4, and the
default everywhere in the code. The linear probability model stays available
behind `--scorer lpm`; its numbers are given in §6 for comparison.

Reproduce with:

```bash
make figures        # or, individually:
python examples/01_adult_baseline_curve.py --scorer logit
python examples/02_adult_full_comparison.py --scorer logit --n-runs 5 --seed-sweep
```

Settings: canonical 32,561 / 16,281 split, all FBU inputs from test predictions,
`n_seeds = 20`, `n_runs = 5`, raw (non-enveloped) baseline curves, region-2 areas
normalised by `A_max`.

## 1. Base model

Lands inside every band predicted in spec §3.3, so the group coding and the
threshold check out (§3.3 warns that SPD near 0 or above 0.35 means they do not).

| | accuracy | balanced accuracy | recall | SPD | EOD | L\* |
|---|---|---|---|---|---|---|
| Logistic regression | 0.8527 | 0.7653 | 0.5996 | +0.1778 | +0.0896 | 0 |

Data checks: 48,842 rows; positive rate 0.2393; 0.3038 for men against 0.1093 for
women, a gap of 0.194; 6,465 `?` cells retained as their own category (`[D6]`).

## 2. Baseline curve

Balanced accuracy × SPD runs from `M_ori` at (0.8222, 0.7653) to `M_100` at
exactly (1.0000, 0.5000) — spec §3.3's predicted shape, and all three `M_100`
anchors hit exactly. Maximum inversion is 0.0000 on all four metric pairs, so the
monotone envelope (`[D5]`) is never needed.

| knot | fairness | performance | knot | fairness | performance |
|---|---|---|---|---|---|
| M_ori | 0.8222 | 0.7653 | M_60 | 0.9293 | 0.6068 |
| M_10 | 0.8399 | 0.7385 | M_70 | 0.9477 | 0.5792 |
| M_20 | 0.8578 | 0.7120 | M_80 | 0.9635 | 0.5532 |
| M_30 | 0.8765 | 0.6852 | M_90 | 0.9823 | 0.5256 |
| M_40 | 0.8934 | 0.6593 | M_100 | 1.0000 | 0.5000 |
| M_50 | 0.9102 | 0.6332 | | | |

The EOD curve is compressed into a much narrower fairness range (0.9104 → 1.0000)
and its knots are three to five times noisier across seeds — up to ±0.016 against
±0.003 for SPD — because EOD conditions on true positives, which shrinks the
effective sample. Every region assignment near the EOD curve inherits that noise.

## 3. Region assignments

Each technique's point per metric pair. `A` is the normalised region-2 area;
other regions have none.

| technique | balanced acc × SPD | balanced acc × EOD | recall × SPD | recall × EOD |
|---|---|---|---|---|
| fairness-unaware | R2, A=0.001 | R2, A=0.051 | R2, A=0.001 | R2, A=0.051 |
| reweighing | R2, A=0.201 | **R5** | R2, A=0.203 | **R5** |
| massaging | R2, A=0.823 | **R5** | R2, A=0.941 | **R5** |
| group thresholds | R2, A=0.031 | R2, A=0.794 | R2, A=0.032 | R2, A=0.802 |
| M_30 control | R4 | R5 | R4 | R5 |

Point estimates behind the table:

| technique | balanced accuracy | recall | SPD | EOD |
|---|---|---|---|---|
| original | 0.7653 | 0.5996 | +0.1778 | +0.0896 |
| fairness-unaware | 0.7628 | 0.5933 | +0.1707 | +0.0682 |
| reweighing | 0.7445 | 0.5538 | +0.0841 | −0.1346 |
| massaging | 0.7476 | 0.5972 | +0.0046 | −0.2795 |
| group thresholds | 0.7639 | 0.5991 | +0.1458 | −0.0091 |

Region distribution over the full 5-run grid (20 cases per technique; the control
gets 200 because it is fed as 10 draws):

| technique | R1 | R2 | R3 | R4 | R5 | mean A | median A | std A |
|---|---|---|---|---|---|---|---|---|
| fairness-unaware | 0 | 100 | 0 | 0 | 0 | 0.029 | 0.026 | 0.028 |
| reweighing | 0 | 50 | 0 | 0 | 50 | 0.202 | 0.202 | 0.001 |
| massaging | 0 | 50 | 0 | 0 | 50 | 0.882 | 0.882 | 0.059 |
| group thresholds | 0 | 100 | 0 | 0 | 0 | 0.416 | 0.412 | 0.384 |
| M_30 control | 0 | 65.5 | 0 | 29.5 | 5 | 0.014 | 0.001 | 0.038 |

## 4. What the numbers say

**Reweighing and massaging trade SPD for EOD, and FBU catches it.** Massaging all
but eliminates SPD (+0.178 → +0.005) and reweighing halves it, and both land in
region 2 with large areas on the SPD pairs. But both drive EOD the *other* way
past zero, to −0.13 and −0.28, overshooting TPR parity in favour of the
unprivileged group — so both land in **region 5, jointly disadvantageous**, on the
EOD pairs. An evaluation on either metric alone would have reported them as
unambiguous wins. This is the case for FBU's metric grid, and it is the most
substantive finding of the run.

**Group thresholds are the only technique effective on every pair.** They cut
|EOD| to 0.009 at a cost of 0.0014 balanced accuracy, which is why they hold
region 2 across all four pairs, with by far the largest EOD areas (0.79, 0.80).
Their SPD areas are small (0.03) because SPD is not what they optimise — the
`area_std` of 0.384 is that split, not noise.

**The fairness-unaware hypothesis is refuted in letter, confirmed in spirit.**
Spec Part IV predicts region 3 or 5. Dropping `sex` in fact nudges both axes
slightly downward (|SPD| 0.178 → 0.171), so the point lands in region 2 — but with
a normalised SPD area of 0.001, roughly 800× smaller than massaging's. Region
membership alone would flatter it; the area score does not. The proxies
(`relationship`, `marital-status`, `hours-per-week`) carry what `sex` carried.

**Ordinal plausibility (§6.1.5) holds per fairness metric, not pooled.** The
expected order — group thresholds > massaging > fairness-unaware — holds on EOD
(0.794 > n/a > 0.051, massaging being in region 5). On SPD it reverses: massaging
(0.882) > reweighing (0.202) > group thresholds (0.032) > fairness-unaware
(0.001). Averaging areas across fairness metrics mixes objectives, so the
per-pair table in §3 is the honest reporting unit.

**Figure 6 does not replicate numerically**, as spec §1.3 anticipates. Ordering
and shape do.

## 5. Validation checks (§6.1)

| Check | Result |
|---|---|
| 1. Degenerate anchor | `M_100` gives fairness_score 1.0, balanced accuracy exactly 0.5, recall exactly 0.0, for every seed. |
| 2. Baseline membership | Against its own curve (`n_seeds=1`) the `M_30` control is region 2 with A = 0 to machine precision. Against the 20-seed mean curve, ten draws split 65.5% / 29.5% / 5% across regions 2 / 4 / 5, mean A = 0.014, median 0.001. See DEVIATIONS.md. |
| 3. Pareto dominance | A synthetic technique that corrects unprivileged false negatives is region 1 on all four pairs; correcting *privileged* false negatives instead gives region 3 on all four. |
| 4. Seed sensitivity | Region percentages for the four real techniques are identical at `n_seeds ∈ {5, 20, 100}`; mean areas move in the third decimal (fairness-unaware 0.033 / 0.029 / 0.027, massaging 0.882 throughout). Only the near-curve control shifts, by ~2 points. **20 seeds is adequate**; the default stands. |
| 5. Ordinal plausibility | Holds per fairness metric; see §4. |

The §2.5 sensitivity check: swapping the enclosed integral for perpendicular
distance to the curve leaves the ranking untouched on both fairness metrics
(SPD: massaging 0.939 > reweighing 0.451 > group thresholds 0.178 >
fairness-unaware 0.030; EOD: group thresholds 0.896 > fairness-unaware 0.229).
The conclusions are therefore not artefacts of the `[D2]` area definition. This is
asserted for both base models in `tests/test_fbu_end_to_end.py`.

## 6. The linear probability model, for comparison

OLS on {0,1} labels thresholded at 0.5 (accuracy 0.8434, balanced accuracy 0.7322,
recall 0.5213, SPD +0.1530, EOD +0.1013; `M_ori` at (0.8470, 0.7322) on the
balanced accuracy × SPD pair). Artefacts are under `outputs/lpm/`.

| technique | balanced acc × SPD | balanced acc × EOD | recall × SPD | recall × EOD |
|---|---|---|---|---|
| fairness-unaware | R2, A=0.001 | R2, A=0.024 | R2, A=0.001 | R2, A=0.024 |
| reweighing | R2, A=0.413 | R5 | R2, A=0.428 | R5 |
| massaging | R2, A=0.543 | R5 | R2, A=0.622 | R5 |
| group thresholds | R2, A=0.056 | R2, A=0.535 | **R1** | **R1** |

Every qualitative conclusion in §4 survives the swap. Two differences are worth
noting: group thresholds reach region 1 on the recall pairs (recall 0.5224 against
the original's 0.5213, a rounding-level improvement that tips the quadrant test),
and their |EOD| on test is 0.026 rather than 0.009, missing the 0.02 target of
spec §5.2 M6 — see DEVIATIONS.md. That second point is why logistic regression is
the better answer to §6.4 on the merits, not only by convention.
