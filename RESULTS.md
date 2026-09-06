# Local experiments

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation. Every number below is a development metric; none establishes clinical utility, safety, or ER performance.

## Early-warning evaluation and visual replay

The existing frozen boosting scores now have patient-level timing evaluation. No model was retrained and no threshold was changed. The onset proxy is the first observed 0-to-1 label transition plus six hours, following the [dataset label definition](https://physionet.org/content/challenge-2019/1.0.0/). It is derived from the existing target, not an independent clinical event.

The primary timing denominator requires the complete inclusive `[onset-12, onset-6]` window and the onset proxy inside the recorded stay. These are new descriptive analysis conventions applied to already-inspected test cohorts, not a prospectively registered evaluation or clinical timing requirement. An ongoing alert counts when it overlaps that window.

| Timing result | Random split | Site-held-out |
|---|---:|---:|
| Patients with a positive label | 587 | 1,142 |
| Timing-eligible patients | 421 | 802 |
| Warned in the 12-to-6-hour window | 107 / 421 (25.4%) | 155 / 802 (19.3%) |
| 95% patient-bootstrap interval | 21.3–29.8% | 16.7–22.0% |
| Missed that early window | 314 | 647 |
| Any alert in `[onset-12, onset)` | 34.4% [29.8–38.9%] | 25.8% [22.6–29.1%] |
| No alert in that before-onset interval | 276 | 595 |
| Nonsepsis patients receiving any alert | 1.80% [1.51–2.10%] | 1.17% [1.01–1.32%] |
| Alert episodes per 100 recorded hours | 0.414 [0.345–0.486] | 0.246 [0.217–0.273] |
| Repeated episodes per 100 recorded hours | 0.304 [0.240–0.371] | 0.178 [0.152–0.202] |

An episode is a consecutive run of threshold-positive hours, separated from another episode by at least one threshold-negative hour. Repeated episodes are episodes after the first in a stay. Episode rates use all recorded hours; timing detection rates use eligible positive-label patients. Intervals use 1,000 whole-patient bootstrap draws with seed 42.

Timing exclusions were 90 positive-at-start records, 70 incomplete windows, and 6 onset proxies beyond the record on the random split; corresponding site-held-out counts were 223, 104, and 13. No nonpersistent label sequences occurred in these test cohorts. Their alert workload remains counted even when timing is excluded. These exclusions change the population being described; the eligible rates must not be presented as sensitivity for all sepsis patients.

The main finding is limited early-window coverage at the existing placeholder alert budget: most eligible positive-label patients had no alert in the chosen window. That is a retrospective window miss, not a clinical missed diagnosis. The two test cohorts overlap, use different training populations, and must not be pooled as independent replications.

The new local dashboard provides play/pause, stepping, scrubbing, patient selection, vital-sign gaps, score/threshold traces, and separate review and alert markers. Future scores and labels are hidden in ordinary playback; a retrospective toggle reveals the full trace and onset-proxy window. It uses the same review scheduler as command-line replay. The matching cohort evaluation appears beside the patient view. Run instructions are in README.md, and `dashboard-demo.ps1` launches the viewer.

Local outputs: `artifacts/early-warning-full-random-42` and `artifacts/early-warning-full-site-42`. Each contains a Markdown summary, JSON with definitions and provenance hashes, and per-patient warning/episode counts. All 28 Python tests pass, including timing-window boundaries, censoring/exclusions, undefined denominators, patient-history verification and read-only HTTP routes.

## Review workload audit added 2026-09-05

The adaptive review policy has now been evaluated across both existing full-cohort test sets, using their frozen selected-model scores and original hourly observations. This audit fits nothing and does not choose a policy. Each fixed schedule starts at the first recorded hour of each stay; consequently, fixed two- and four-hour schedules have slightly more than 50 and 25 reviews per 100 hours on finite stays.

| Schedule | Random split: reviews / 100 hours [95%] | Site-held-out: reviews / 100 hours [95%] |
|---|---:|---:|
| Adaptive | 55.66 [55.05, 56.29] | 44.86 [44.63, 45.11] |
| Fixed every hour | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| Fixed every 2 hours | 50.64 [50.62, 50.65] | 50.64 [50.63, 50.65] |
| Fixed every 4 hours | 25.97 [25.95, 25.99] | 25.97 [25.95, 25.98] |

There are 172,306 adaptive review events across 309,558 hours for 8,068 random-split patients, and 341,860 across 761,995 hours for 20,000 site-held-out patients. The two test sets overlap and must not be pooled as independent cohorts. Intervals use 1,000 whole-patient resamples with seed 42; JSON outputs also contain paired adaptive-minus-fixed differences.

Missing or stale vitals force review in **36.8% of random-split hours and 22.4% of site-held-out hours**. Adaptive review therefore creates more events than a fixed two-hour schedule in the random split, but fewer on the held-out site. This demonstrates that the placeholder 2% validation **alert-hour** budget is not a **review-event** budget. Neither review counts nor their differences establish staffing requirements, clinical benefit, or a preferable schedule. All measurements still arrive hourly, and fixed schedules do not respond to missingness or scores.

Reproduce with the `workload` commands in README.md. Local outputs are in `artifacts/workload-full-random-42` and `artifacts/workload-full-site-42`, each containing a Markdown report, JSON with source/code hashes, and per-patient counts. Patient hashes, development/test separation, prediction counts and ordered labels were verified. These historical prediction files lack explicit hour keys, so the audit assumes their original within-patient row order is intact. New training outputs include hour keys for an additional alignment check.

**Budget-rule correction:** the old rule returned the highest observed score when no threshold satisfied the budget. Tied scores could therefore exceed capacity; in particular the constant prevalence baseline alerted on 100% of hours. New training chooses no alerts in that case and preserves the decision through recalibration. The historical training tables below retain the old results, including that prevalence-baseline bug. No full model retraining was performed for this audit; the two workload reports use the existing frozen boosting thresholds.

Verification for this addition: 19 tests pass, including budget ties and endpoint scores, causal scheduling and early interruption, paired workload resampling, artifact-integrity failures, and training-to-audit-to-replay consistency. The full random-split replay still runs, and `pip check` reports no broken requirements.

## What the full cohort changed

Earlier releases trained on a seeded 2,000-patient subsample. Training on all 40,336 patients **retracts two of the previous headline findings and sharpens a third.** Both are recorded here rather than quietly replaced.

- **Retracted: the subgroup disparities.** The previous release reported that the selected model ranked one 445-patient subgroup below chance on an unseen site (AUROC 0.427), alerted on no positive hour at all in three subgroups, and returned worse-than-silence utility in fourteen subgroup levels. **None of that survives.** At full cohort there is no subgroup with below-chance discrimination, none with zero recall, and none with negative utility, on either split. Subgroup AUROC now spans 0.805–0.850 on the random split and 0.746–0.791 across sites. Those were small-sample artifacts, and the earlier writeup gave them more weight than the evidence supported.
- **Retracted: the "badly overextended scores" finding.** Held-out calibration slopes of 0.11–0.49 were largely an overfitting artifact of 47,000 training hours. With 930,647, the selected model's slope is 0.903 within source and the logistic baseline's is 1.002. Recalibration is now nearly a no-op and on one model makes calibration slightly worse.
- **Sharpened: the model comparison was underpowered, not tied.** The previous release found that no validation selection margin excluded zero across ten runs, and concluded the baselines were indistinguishable. At full cohort a *smaller* observed margin is decisively non-zero: +0.0235 [+0.0123, +0.0372] on the random split. The correct reading of the earlier result is that 254–400 validation patients could not resolve the comparison, not that the models were equivalent.

**Cross-site degradation is the finding that survived every configuration and every cohort size**, and it is now measured precisely.

## Cohort

All 40,336 patients from the two official PhysioNet 2019 training sets: 20,336 in set A and 20,000 in set B. Every file was schema-checked on download and all 40,336 SHA-256 digests are distinct.

| Run | Split | Train | Validation | Test | Test hours | Positive test hours |
|---|---|---:|---:|---:|---:|---:|
| `artifacts/full-random-42` | random patients | 24,201 | 8,067 | 8,068 | 309,558 | 5,556 (1.79%) |
| `artifacts/full-site-42` | all of set B held out | 15,252 | 5,084 | 20,000 | 761,995 | 10,780 (1.41%) |

Both use the default configuration: an alert budget of 2.0 alert hours per 100 for the threshold, and stable selection for the model. The budget is an arbitrary placeholder; no clinician set it. Prior 2,000-patient runs are retained for comparison in [Cohort size](#cohort-size).

## Held-out results

95% percentile intervals from 1,000 bootstrap resamples of whole test patients. Normalized utility is the official PhysioNet/CinC 2019 score, where 1.0 is the best attainable alert timing and 0.0 is never alerting.

Random-patient split, selected model **gradient boosting**:

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall | Alerts per 100 h |
|---|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0179 [0.017, 0.019] | −0.470 [−0.596, −0.352] | 0.018 | 1.000 | 100.00 |
| Logistic regression | 0.776 [0.758, 0.793] | 0.0833 [0.071, 0.100] | 0.133 [0.111, 0.157] | 0.154 [0.127, 0.181] | 0.141 [0.120, 0.166] | 1.65 [1.35, 2.00] |
| Gradient boosting | 0.827 [0.812, 0.841] | 0.0952 [0.084, 0.108] | 0.161 [0.137, 0.186] | 0.150 [0.126, 0.177] | 0.166 [0.142, 0.190] | 1.98 [1.60, 2.35] |

Site-held-out split, selected model **gradient boosting**, on the 20,000 set B patients no model saw:

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall | Alerts per 100 h |
|---|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0141 [0.013, 0.015] | −0.916 [−1.043, −0.799] | 0.014 | 1.000 | 100.00 |
| Logistic regression | 0.703 [0.688, 0.719] | 0.0508 [0.045, 0.059] | 0.092 [0.077, 0.108] | 0.109 [0.092, 0.130] | 0.102 [0.088, 0.119] | 1.33 [1.09, 1.59] |
| Gradient boosting | 0.765 [0.752, 0.778] | 0.0660 [0.059, 0.075] | 0.121 [0.104, 0.139] | 0.126 [0.108, 0.144] | 0.128 [0.112, 0.145] | 1.44 [1.24, 1.66] |

At the 2.0-alert-hour budget the selected model reaches recall 0.166 at precision 0.150 within source, and recall 0.128 at precision 0.126 across sites. It alerts at some point during the stay of 1.80% of nonsepsis patients on the random split and 1.17% on the unseen site. Both utility intervals now exclude zero by a wide margin, so unlike every previous release **the selected model is distinguishable from never alerting, including on an unseen source.** That is a statement about a retrospective ICU cohort and a label the dataset already shifted; it is not evidence of clinical benefit.

**Cross-site degradation is real and precisely bounded.** Selected-model AUROC falls from 0.827 [0.812, 0.841] within source to 0.765 [0.752, 0.778] on the unseen site — intervals that do not overlap. Average precision falls by a third, from 0.0952 to 0.0660. Training on twenty times more data raised both numbers but did not close the gap between them.

## Model selection

`stable` takes the validation average-precision leader, then runs a paired patient bootstrap on validation predictions — the same resampled patients scored under both models — and steps back to the simpler model when the interval for the difference includes zero.

| Split | Validation patients | Validation AP, logistic vs boosting | Paired margin [95%] | Leader | Reported |
|---|---:|---|---|---|---|
| Random | 8,067 | 0.0792 vs 0.1027 | +0.0235 [+0.0123, +0.0372] | boosting | boosting |
| Site | 5,084 | 0.0912 vs 0.1191 | +0.0279 [+0.0141, +0.0438] | boosting | boosting |

Both margins exclude zero, so the rule declines to step back and reports boosting in both splits. Set against the ten 2,000-patient runs, where boosting once led by +0.0561 with an interval of [−0.0318, +0.1716], this is the clearest available demonstration that the earlier indistinguishability was a power problem: **a margin half the size is now decisive because the validation set is twenty times larger.** The rule behaved correctly in both regimes — it stepped back when the evidence was absent and refuses to step back when it is present.

## Calibration and recalibration

A Platt map is fitted on validation predictions only, frozen, then applied to held-out hours. It is strictly monotone, so ranking, discrimination and every alert decision are unchanged once the threshold passes through the same map; each run records `alerts_identical: true` as a check.

| Run | Model | ECE before → after | Slope before → after | Intercept before → after | Brier before → after |
|---|---|---|---|---|---|
| Random | Logistic | 0.0027 → 0.0035 | 1.002 → 1.057 | 0.009 → 0.241 | 0.01722 → 0.01719 |
| Random | Boosting | 0.0028 → 0.0021 | 0.903 → 0.987 | −0.374 → −0.023 | 0.01713 → 0.01698 |
| Site-held-out | Logistic | 0.0047 → 0.0046 | 0.809 → 0.827 | −0.954 → −0.888 | 0.01399 → 0.01395 |
| Site-held-out | Boosting | 0.0035 → 0.0032 | 0.788 → 0.802 | −0.921 → −0.864 | 0.01391 → 0.01386 |

**Within source, the models are already close to calibrated and recalibration has almost nothing left to do.** The logistic baseline's raw slope is 1.002 with intercept 0.009, and applying the map actually makes it slightly worse — slope 1.057, intercept 0.241, ECE up from 0.0027 to 0.0035. Fitting a correction on a validation set that no longer needs one adds noise. The previous release's slopes of 0.11 to 0.49 were reporting overfitting on 47,000 training hours, not a property of the method.

**Across sites the calibration gap is real but much smaller than previously reported.** The selected model's slope on the unseen site is 0.788 with intercept −0.921: scores carried over from set A are systematically too high on set B, and slightly overextended. A map fitted on set A validation barely moves them, to 0.802 and −0.864, which is the same qualitative conclusion as before — same-source recalibration does not transfer — at a fraction of the previously reported magnitude. Brier scores now beat the prevalence baseline on both splits before any recalibration (0.01713 against 0.01763, and 0.01391 against 0.01400), which no earlier run achieved.

Expected calibration error is small throughout partly because 98% of hours are negative. Ten equal-count reliability bins per model, before and after, are in each run's `metrics.json`.

## Subgroup description

Held-out hours split by the recorded administrative fields Age, Gender, Unit1 and Unit2, assigned from each patient's first hour before scoring. No intervals and no multiplicity control; these describe one cohort. Gender is the dataset's 0/1 code, kept unlabelled because it is an administrative record. `unit_unrecorded` is a data-availability level: set B carries no unit indicator.

Selected model (boosting), random split:

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 3,543 | 1.58% | 0.823 | 0.0767 | 0.141 | 0.133 | 1.99 |
| gender_1 | 4,525 | 1.97% | 0.829 | 0.1145 | 0.181 | 0.179 | 1.97 |
| age_lt_50 | 1,848 | 1.98% | 0.850 | 0.1241 | 0.151 | 0.154 | 1.44 |
| age_50_64 | 2,474 | 1.60% | 0.822 | 0.0751 | 0.160 | 0.147 | 2.36 |
| age_65_79 | 2,698 | 1.83% | 0.818 | 0.1021 | 0.182 | 0.177 | 2.14 |
| age_80_plus | 1,048 | 1.82% | 0.819 | 0.1010 | 0.159 | 0.160 | 1.57 |
| unit1 | 2,510 | 1.86% | 0.811 | 0.0827 | 0.136 | 0.133 | 1.89 |
| unit2 | 2,441 | 1.34% | 0.805 | 0.0824 | 0.145 | 0.144 | 1.39 |
| unit_unrecorded | 3,117 | 2.09% | 0.840 | 0.1124 | 0.196 | 0.189 | 2.50 |
| site_A | 4,061 | 2.11% | 0.814 | 0.1035 | 0.181 | 0.178 | 2.46 |
| site_B | 4,007 | 1.47% | 0.838 | 0.0843 | 0.143 | 0.136 | 1.48 |

Selected model (boosting), site-held-out split, on the 20,000 unseen set B patients:

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 9,268 | 1.33% | 0.772 | 0.0751 | 0.143 | 0.140 | 1.34 |
| gender_1 | 10,732 | 1.49% | 0.759 | 0.0612 | 0.117 | 0.107 | 1.54 |
| age_lt_50 | 4,818 | 1.46% | 0.746 | 0.0572 | 0.124 | 0.116 | 1.68 |
| age_50_64 | 6,278 | 1.40% | 0.770 | 0.0645 | 0.108 | 0.101 | 1.25 |
| age_65_79 | 6,582 | 1.41% | 0.773 | 0.0730 | 0.142 | 0.134 | 1.59 |
| age_80_plus | 2,322 | 1.37% | 0.771 | 0.0804 | 0.148 | 0.146 | 1.06 |
| unit1 | 6,923 | 1.39% | 0.754 | 0.0608 | 0.119 | 0.109 | 1.28 |
| unit2 | 6,982 | 1.48% | 0.754 | 0.0691 | 0.171 | 0.161 | 2.10 |
| unit_unrecorded | 6,095 | 1.36% | 0.791 | 0.0737 | 0.083 | 0.082 | 0.84 |

**Every subgroup alarm from the previous release was noise.** The clearest case: gender_0 on the unseen site was reported at AUROC 0.427 on 445 patients, below chance, with negative utility and more than double the alert load of gender_1. On 9,268 patients it is 0.772 — *higher* than gender_1's 0.759 — with positive utility and a slightly lower alert load. Across both splits' twenty subgroup levels there is now no below-chance level, no zero-recall level and no negative-utility level; the smallest level here holds 1,048 patients rather than 56.

What remains is modest and unremarkable variation: 0.045 AUROC between the best and worst random-split level, 0.045 across sites. Alert load still varies about twofold between levels (0.84 to 2.50 per 100 hours), which matters operationally if a budget is ever set for real. These are still descriptions of one retrospective cohort without intervals or multiplicity control, and subgroup prevalence still differs between levels, so they confound model behavior with case mix. They no longer show a system that fails particular groups.

## Cohort size

Selected-model held-out results at each cohort size, default configuration, seed 42.

| Split | Cohort | Test patients | Selected | AUROC | Average precision | Utility |
|---|---|---:|---|---|---|---|
| Random | 2,000 | 400 | logistic | 0.674 [0.575, 0.775] | 0.0472 [0.022, 0.104] | 0.103 [0.005, 0.213] |
| Random | 40,336 | 8,068 | boosting | 0.827 [0.812, 0.841] | 0.0952 [0.084, 0.108] | 0.161 [0.137, 0.186] |
| Site | 2,000 | 987 | logistic | 0.588 [0.508, 0.669] | 0.0271 [0.016, 0.055] | 0.063 [−0.003, 0.131] |
| Site | 40,336 | 20,000 | boosting | 0.765 [0.752, 0.778] | 0.0660 [0.059, 0.075] | 0.121 [0.104, 0.139] |

Interval widths fall by roughly an order of magnitude — random-split AUROC from ±0.100 to ±0.015 — and the point estimates move outside the old intervals in three of four cases. **A twentyfold larger cohort changed the reported conclusion, not just its precision.** The earlier five-seed sweeps measured how much a 2,000-patient subsample varies, which is a real quantity, but it was never a bound on how far those runs sat from the full-cohort answer.

The five-seed sweeps of the 2,000-patient cohort remain in the git history for the previous release. They have not been repeated at full cohort: with all 40,336 patients the seed no longer selects a subsample, so it would vary only the split assignment, and a single full-cohort run takes about nine minutes for the random split and fourteen for the site split.

## Verification

- Fifteen tests passed, covering causal feature prefixes, disjoint reproducible patient splits, whole-site holdout, invalid-input rejection, the official utility ramps and normalization, calibration estimates, recalibration monotonicity and alert preservation, each threshold rule against its own objective, the selection tie-break in both directions including the margins it records, subgroup partitioning against directly computed metrics, patient-descriptor bracketing, bootstrap determinism, illustrative policy behavior, and end-to-end batch/replay equivalence.
- `pip check` found no broken requirements.
- All 40,336 records downloaded and schema-checked, with 40,336 distinct SHA-256 digests, so no patient file is a duplicate of another. Source URLs and digests are in `data/physionet2019/provenance.json`.
- The full-cohort runs were repeated after a change to what the selection rule records. Held-out predictions and metrics were unaffected; only the recorded selection margin was added.
- `--threshold f1 --select ap` still reproduces the pre-rule-change configuration bit for bit at seed 42 on the 2,000-patient cohort.
- Historical replay completed for held-out record `site_A/p000002.psv` under the full random-split run: 23 hourly rows, scores 0.004–0.010, review forced to hourly wherever observations are missing or stale and otherwise driven by low scores. Recalibration changed no review event.

Use the commands in README.md to replay locally or reproduce a run. Review timing remains an illustrative software policy; it has not been optimized or clinically validated, and the 2.0 alert-hour budget is a placeholder for a capacity nobody has stated.
