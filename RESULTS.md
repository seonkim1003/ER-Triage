# Local experiments

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation. Every number below is a development metric; none establishes clinical utility, safety, or ER performance.

## Cohort

Uniform seeded download of 2,000 patients from the 40,336 records listed in the two official PhysioNet 2019 training sets: 1,013 from set A and 987 from set B. No full-cohort training has been run.

| Run | Split | Train | Validation | Test | Test hours | Positive test hours |
|---|---|---:|---:|---:|---:|---:|
| `artifacts/default-random-42` | random patients | 1,200 | 400 | 400 | 15,789 | 240 (1.52%) |
| `artifacts/default-site-42` | all of set B held out | 759 | 254 | 987 | 37,592 | 507 (1.35%) |

Fifty runs in total: five seeds per split for the default configuration, for the previous configuration, and for each rule varied one factor at a time. See [Operating point and model selection](#operating-point-and-model-selection) and [Seed sensitivity](#seed-sensitivity).

The **default configuration** is an alert budget of 2.0 alert hours per 100 for the threshold, and stable selection for the model. Both are defined in the next section. The budget is an arbitrary placeholder: no clinician set it, and it is a stand-in for a review capacity someone would have to state.

## Held-out results, default configuration

95% percentile intervals come from 1,000 bootstrap resamples of whole test patients. Normalized utility is the official PhysioNet/CinC 2019 score, where 1.0 is the best attainable alert timing and 0.0 is never alerting.

Random-patient split, selected model **logistic regression**:

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall | Alerts per 100 h |
|---|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0152 [0.010, 0.021] | −0.851 [−1.979, −0.277] | 0.015 | 1.000 | 100.00 |
| Logistic regression | 0.674 [0.575, 0.775] | 0.0472 [0.022, 0.104] | 0.103 [0.005, 0.213] | 0.116 [0.032, 0.240] | 0.125 [0.033, 0.237] | 1.63 [0.86, 2.52] |
| Gradient boosting | 0.723 [0.643, 0.811] | 0.0890 [0.033, 0.199] | 0.143 [0.027, 0.276] | 0.125 [0.040, 0.250] | 0.154 [0.044, 0.289] | 1.88 [0.80, 3.21] |

Site-held-out split, selected model **logistic regression**, evaluated on the 987 set B patients no model saw:

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall | Alerts per 100 h |
|---|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0135 [0.010, 0.017] | −1.070 [−1.878, −0.612] | 0.013 | 1.000 | 100.00 |
| Logistic regression | 0.588 [0.508, 0.669] | 0.0271 [0.016, 0.055] | 0.063 [−0.003, 0.131] | 0.063 [0.024, 0.140] | 0.097 [0.040, 0.166] | 2.06 [1.05, 3.27] |
| Gradient boosting | 0.690 [0.616, 0.754] | 0.0403 [0.023, 0.080] | 0.027 [−0.006, 0.070] | 0.104 [0.011, 0.376] | 0.036 [0.003, 0.080] | 0.46 [0.11, 0.85] |

The prevalence baseline is far worse than silence because alerting every hour is penalized. On the random split the selected model's utility interval clears zero by a hair, [0.005, 0.213]; on the unseen site it does not, [−0.003, 0.131]. **Across the site, the selected model is still not distinguishable from never alerting at all.** Held-out discrimination degrades on the unseen site exactly as before: AUROC 0.674 to 0.588, lower bound 0.508.

Boosting scores higher on the random split's held-out hours than the model that was selected. That was not used to revise the selection; the reason the selection rule reports logistic anyway is the subject of the next section.

## Operating point and model selection

The previous release's seed sweep showed that both choices frozen on validation — which model to report, and where to put the threshold — were unstable enough to be near arbitrary. Two replacement rules are now implemented and compared, one factor at a time, against the previous configuration.

### Model selection

`ap` reports whichever baseline has the highest validation average precision. `stable` finds that leader, then runs a **paired patient bootstrap on validation predictions**: the same resampled patients are scored under both models, giving an interval for the average-precision difference. If that interval includes zero, the leader is not distinguishable from the simpler model, and the simpler model is reported. The order is prespecified as logistic before boosting; prevalence remains a reference baseline and is never selected.

| Split | Seed | AP leader | Validation AP, logistic vs boosting | Paired margin [95%] | Reported |
|---|---:|---|---|---|---|
| Random | 42 | logistic | 0.0942 vs 0.0824 | — | logistic |
| Random | 1 | logistic | 0.0585 vs 0.0583 | — | logistic |
| Random | 7 | boosting | 0.0295 vs 0.0312 | +0.0018 [−0.0193, +0.0212] | logistic |
| Random | 13 | boosting | 0.0661 vs 0.0721 | +0.0060 [−0.0353, +0.1121] | logistic |
| Random | 2024 | boosting | 0.0680 vs 0.0868 | +0.0188 [−0.0250, +0.1204] | logistic |
| Site | 42 | boosting | 0.0435 vs 0.0497 | +0.0062 [−0.0217, +0.0493] | logistic |
| Site | 1 | boosting | 0.1091 vs 0.1594 | +0.0503 [−0.0725, +0.2018] | logistic |
| Site | 7 | logistic | 0.1075 vs 0.0645 | — | logistic |
| Site | 13 | boosting | 0.0764 vs 0.1324 | +0.0561 [−0.0318, +0.1716] | logistic |
| Site | 2024 | logistic | 0.0967 vs 0.0778 | — | logistic |

**Not one validation margin excludes zero.** Even the largest, boosting ahead by 0.056 average precision at site seed 13, has an interval from −0.032 to +0.172. The `ap` rule flipped between models five times across these ten runs; `stable` reports logistic in all ten, stepping back from a boosting leader six times. That is a reproducibility gain and nothing more — it is a tie-breaking convention, not evidence that logistic generalizes better. What the table actually shows is that 254 to 400 validation patients cannot tell these two baselines apart.

### Threshold rule

Three validation-only rules, each frozen before test hours are scored:

- **`f1`** (previous default) maximizes hourly F1 on validation.
- **`utility`** maximizes the official patient-weighted normalized utility on validation.
- **`budget`** takes the lowest threshold whose validation alert load stays within a stated cap, here 2.0 alert hours per 100.

Alert load across the five seeds per split, holding selection at `ap` so only the threshold rule varies:

| Split | Rule | Validation alerts per 100 h | Held-out alerts per 100 h | Held-out utility, median [range] |
|---|---|---|---|---|
| Random | f1 | 1.67 – 16.37 | 1.38 – 16.08 | 0.174 [0.079, 0.394] |
| Random | utility | 7.50 – 26.94 | 5.74 – 28.61 | 0.325 [0.157, 0.395] |
| Random | budget | 1.98 – 2.00 | 1.19 – 3.28 | 0.111 [0.064, 0.122] |
| Site | f1 | 0.70 – 5.60 | 0.53 – 4.33 | 0.029 [0.007, 0.151] |
| Site | utility | 9.51 – 25.51 | 13.01 – 29.29 | 0.070 [−0.056, 0.121] |
| Site | budget | 1.99 – 2.00 | 0.46 – 4.48 | 0.032 [0.024, 0.149] |

Three things follow, and the first is a caveat about the comparison itself:

- **Validation stability cannot arbitrate between these rules.** The budget rule pins the validation alert load to the budget by construction, so its perfect validation spread is a tautology, not a finding. What it does is move the operating point out of the data and into a stated decision. The held-out columns are reported as consequences, not as the basis for choosing.
- **Maximizing the official utility buys score by alerting constantly.** It reaches the best random-split utility of the three rules, median 0.325, by alerting on 6% to 29% of all hours at a precision of 0.04 to 0.06. A false alert hour costs 0.05 while a well-timed alert earns up to 1.0, so the score rewards blanketing the ward. On the unseen site that strategy collapses: median utility 0.070, negative at one seed, every interval spanning zero, while still alerting on 13% to 29% of hours. This is a property of the scoring function, not a defensible operating point.
- **A stated budget transfers within a source and not across sites.** A threshold set to 2.0 alert hours per 100 on validation lands between 1.19 and 3.28 on held-out random-split patients, but between 0.46 and 4.48 on the unseen site — a tenfold spread for the same nominal capacity. Even fixing the alert budget does not fix what the alert load will actually be at another hospital.

The default is now `budget` with a 2.0 cap and `stable` selection, and `--threshold f1 --select ap` reproduces the previous configuration exactly. The budget rule was made the default because it makes the operating point an explicit, reproducible choice rather than the argmax of a noisy curve, and because its utility spread across seeds is the narrowest of the three (0.058 wide on the random split against F1's 0.315). **The 2.0 figure itself is arbitrary and is the single most important unvalidated number in this project.** Some of that narrowness is mechanical: a fixed alert budget also caps how much utility a run can earn or lose.

## Calibration and recalibration

Raw scores are not calibrated probabilities: logit recalibration on held-out hours gives slopes far below 1 and strongly negative intercepts, meaning the score spread is wider than the evidence supports. A Platt map is fitted **on validation predictions only**, frozen, and then applied to held-out hours. It is strictly monotone, so it changes reported probabilities but leaves ranking, AUROC, average precision and every alert decision unchanged once the threshold is passed through the same map. Each run records `alerts_identical: true` as a check.

Seed 42, default configuration; the selected model is logistic in both splits.

| Run | Model | ECE before → after | Slope before → after | Intercept before → after | Brier before → after | Threshold before → after |
|---|---|---|---|---|---|---|
| Random | Logistic | 0.0114 → 0.0035 | 0.491 → 0.728 | −2.077 → −1.056 | 0.01614 → 0.01484 | 0.203170 → 0.088814 |
| Random | Boosting | 0.0136 → 0.0026 | 0.479 → 1.143 | −1.734 → 0.488 | 0.01614 → 0.01455 | 0.209705 → 0.075805 |
| Site-held-out | Logistic | 0.0218 → 0.0083 | 0.110 → 0.346 | −3.786 → −2.906 | 0.01795 → 0.01335 | 0.233254 → 0.051392 |
| Site-held-out | Boosting | 0.0108 → 0.0058 | 0.430 → 1.257 | −2.159 → 0.589 | 0.01446 → 0.01320 | 0.364766 → 0.085259 |

Recalibration removes most of the overextension within the same source: the selected logistic model's held-out Brier improves from 0.01614 to 0.01484, the first time a learned baseline scores better than the prevalence predictor's 0.01497 on this cohort. **It does not fix the cross-site case.** The selected model's residual slope on the unseen site is 0.346, still far from 1, so a map fitted on set A does not transfer to set B; across the five site-split seeds the recalibrated Brier beats the prevalence baseline's 0.01334 in only one run. The fitted map is itself seed-dependent — on the random split it lands between 0.67 and 2.09, overshooting past a slope of 1 on two of five seeds. Recalibration cannot change discrimination or which hours alert, so none of this is an operational gain.

Expected calibration error looks small throughout only because almost every hour is negative. Ten equal-count reliability bins per model, before and after, are recorded in each run's `metrics.json`.

## Subgroup description

Held-out hours split by the recorded administrative fields Age, Gender, Unit1 and Unit2, assigned from each patient's first hour before scoring. These are descriptions of one cohort with no intervals and no multiplicity control. Some levels hold only a few dozen positive hours, so the differences below are unstable and are **not** evidence about fairness in care. Gender is the dataset's 0/1 code, kept unlabelled here because it is an administrative record. `unit_unrecorded` is a data-availability level: set B carries no unit indicator at all.

Selected model, random split (logistic, default configuration):

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 182 | 1.21% | 0.623 | 0.0247 | 0.011 | −0.005 | 0.75 |
| gender_1 | 218 | 1.79% | 0.691 | 0.0645 | 0.191 | 0.162 | 2.39 |
| age_lt_50 | 111 | 1.34% | 0.719 | 0.0486 | 0.263 | 0.214 | 3.35 |
| age_50_64 | 104 | 1.70% | 0.498 | 0.0181 | 0.000 | −0.020 | 1.17 |
| age_65_79 | 129 | 1.66% | 0.776 | 0.1028 | 0.095 | 0.097 | 0.67 |
| age_80_plus | 56 | 1.21% | 0.694 | 0.0989 | 0.250 | 0.203 | 1.43 |
| unit1 | 113 | 0.68% | 0.675 | 0.0127 | 0.000 | −0.147 | 3.18 |
| unit2 | 125 | 1.72% | 0.580 | 0.0291 | 0.024 | 0.026 | 0.53 |
| unit_unrecorded | 162 | 2.00% | 0.761 | 0.1554 | 0.226 | 0.213 | 1.32 |
| site_A | 199 | 1.90% | 0.712 | 0.0633 | 0.113 | 0.096 | 1.08 |
| site_B | 201 | 1.18% | 0.633 | 0.0479 | 0.141 | 0.112 | 2.12 |

Selected model, site-held-out split (logistic, default configuration), on the 987 unseen set B patients:

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 445 | 0.74% | 0.427 | 0.0073 | 0.062 | −0.047 | 3.12 |
| gender_1 | 542 | 1.87% | 0.654 | 0.0697 | 0.108 | 0.101 | 1.17 |
| age_lt_50 | 258 | 1.40% | 0.620 | 0.0270 | 0.111 | 0.034 | 4.38 |
| age_50_64 | 298 | 1.39% | 0.514 | 0.0375 | 0.142 | 0.141 | 1.59 |
| age_65_79 | 301 | 1.17% | 0.656 | 0.0517 | 0.081 | 0.058 | 0.76 |
| age_80_plus | 130 | 1.56% | 0.554 | 0.0183 | 0.012 | −0.020 | 1.70 |
| unit1 | 356 | 1.23% | 0.577 | 0.0324 | 0.155 | 0.118 | 2.94 |
| unit2 | 330 | 1.49% | 0.620 | 0.0263 | 0.011 | −0.011 | 1.07 |
| unit_unrecorded | 301 | 1.34% | 0.625 | 0.0320 | 0.140 | 0.097 | 2.12 |

**The aggregate numbers hide subgroups the selected model never helps, and one where it is worse than a coin flip.** On the unseen site its AUROC for gender_0 is 0.427 across 445 patients — below chance — while it spends 3.12 alert hours per 100 there, more than double the 1.17 it spends on gender_1, and returns negative utility. On the random split it alerts on no positive hour at all for age_50_64 (104 patients) and unit1 (113 patients), while still raising 3.18 false alert hours per 100 for unit1. Negative subgroup utility, worse than staying silent, appears in three of eleven random-split levels and three of nine site-split levels here, and gender_0 was negative in all five site seeds of the previous configuration as well. Subgroup prevalence differs between levels, so these gaps confound model behavior with case mix and cannot be read as a disparity estimate. They are enough to say the aggregate metrics are not describing a uniform system.

## Seed sensitivity

Five seeds per split per configuration. The seed drives both the 2,000-patient download subset and the split, so this varies cohort and split together; it is not a decomposition of the two.

Default configuration (budget threshold, stable selection); every run reports logistic:

| Split | Seed | AUROC | Average precision | Utility [95% interval] | Alerts per 100 h |
|---|---:|---:|---:|---|---:|
| Random | 42 | 0.674 | 0.0472 | 0.103 [0.005, 0.213] | 1.63 |
| Random | 1 | 0.762 | 0.0559 | 0.064 [−0.004, 0.147] | 1.99 |
| Random | 7 | 0.679 | 0.0680 | 0.100 [0.015, 0.201] | 0.81 |
| Random | 13 | 0.763 | 0.0935 | 0.217 [0.086, 0.360] | 2.37 |
| Random | 2024 | 0.743 | 0.0458 | 0.097 [−0.021, 0.221] | 3.52 |
| Site | 42 | 0.588 | 0.0271 | 0.063 [−0.003, 0.131] | 2.06 |
| Site | 1 | 0.563 | 0.0232 | 0.057 [−0.019, 0.127] | 2.91 |
| Site | 7 | 0.613 | 0.0257 | 0.024 [−0.020, 0.072] | 1.64 |
| Site | 13 | 0.613 | 0.0302 | 0.111 [0.019, 0.209] | 4.38 |
| Site | 2024 | 0.639 | 0.0422 | 0.149 [0.056, 0.237] | 4.48 |

Previous configuration (F1 threshold, average-precision selection), for comparison:

| Split | Seed | Selected | AUROC | Utility [95% interval] | Alerts per 100 h |
|---|---:|---|---:|---|---:|
| Random | 42 | logistic | 0.674 | 0.079 [−0.001, 0.176] | 1.38 |
| Random | 1 | logistic | 0.762 | 0.097 [0.012, 0.196] | 2.94 |
| Random | 7 | boosting | 0.781 | 0.394 [0.182, 0.557] | 16.08 |
| Random | 13 | boosting | 0.818 | 0.305 [0.126, 0.470] | 6.21 |
| Random | 2024 | boosting | 0.730 | 0.174 [0.002, 0.333] | 7.37 |
| Site | 42 | boosting | 0.690 | 0.073 [−0.016, 0.166] | 3.11 |
| Site | 1 | boosting | 0.650 | 0.007 [−0.026, 0.044] | 0.53 |
| Site | 7 | logistic | 0.613 | 0.027 [−0.018, 0.080] | 1.75 |
| Site | 13 | boosting | 0.639 | 0.029 [−0.006, 0.078] | 0.93 |
| Site | 2024 | logistic | 0.639 | 0.151 [0.060, 0.240] | 4.33 |

- **Fixing the rules narrows the spread but does not make one run informative.** Held-out utility across the five random-split seeds spans 0.064 to 0.217 under the default configuration against 0.079 to 0.394 before, and alert load 0.81 to 3.52 against 1.38 to 16.08. Reported AUROC now spans 0.674 to 0.763 because the same model family is reported every time; the previous spread of 0.674 to 0.818 partly measured which model happened to win.
- **Seed 42 remains near the pessimistic end of its own range**, on utility and on discrimination, under both configurations. No seed was chosen after seeing test results; 42 was fixed from the first experiment.
- **Cross-site degradation is the one pattern that survives every configuration.** Every site-split seed scores below every random-split seed on AUROC, under both rule sets. Under the default configuration 3 of 5 site-split utility intervals include zero, against 1 of 5 on the random split.

## Verification

- Fifteen tests passed, covering causal feature prefixes, disjoint reproducible patient splits, whole-site holdout, invalid-input rejection, the official utility ramps and normalization, calibration estimates, recalibration monotonicity and alert preservation, each threshold rule against its own objective, the selection tie-break in both directions, subgroup partitioning against directly computed metrics, patient-descriptor bracketing, bootstrap determinism, illustrative policy behavior, and end-to-end batch/replay equivalence.
- `pip check` found no broken requirements.
- **The previous configuration is reproducible bit for bit.** `--threshold f1 --select ap` at seed 42 reproduces the pre-change run exactly: identical held-out predictions and identical test metrics for all three baselines. The refactor that added the new rules did not disturb the old path.
- Patient manifests and file hashes are identical across configurations at the same seed, and so are each model's held-out scores: neither rule can alter a fitted model. The random-split prediction file is byte-identical to the previous configuration's because the same model is reported; the site-split one differs only because a different model is now reported.
- Historical replay completed for held-out record `site_A/p000119.psv` (17 hourly rows, every one forced to hourly review by missing or stale observations, scores 0.002–0.012) and for `site_B/p100001.psv` under the site-held-out run (24 rows, 11 reviews, hourly review forced for the first six hours and the cadence otherwise driven by low scores). Recalibration raises the displayed probabilities on the site run without changing a single review event.

Use the commands in README.md to replay locally or start a larger experiment. Review timing remains an illustrative software policy; it has not been optimized or clinically validated, and the 2.0 alert-hour budget is a placeholder for a capacity nobody has stated.
