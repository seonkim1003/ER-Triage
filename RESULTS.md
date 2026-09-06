# Local experiments

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation. Every number below is a development metric; none establishes clinical utility, safety, or ER performance.

## Cohort

Uniform seeded download of 2,000 patients from the 40,336 records listed in the two official PhysioNet 2019 training sets: 1,013 from set A and 987 from set B. No full-cohort training has been run.

| Run | Split | Train | Validation | Test | Test hours | Positive test hours |
|---|---|---:|---:|---:|---:|---:|
| `artifacts/random-seed-42` | random patients | 1,200 | 400 | 400 | 15,789 | 240 (1.52%) |
| `artifacts/site-seed-42` | all of set B held out | 759 | 254 | 987 | 37,592 | 507 (1.35%) |

Eight further runs use seeds 1, 7, 13 and 2024 for each split; see [Seed sensitivity](#seed-sensitivity).

## Random-patient split

Held-out results with 95% percentile intervals from 1,000 bootstrap resamples of whole test patients.

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall |
|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0152 [0.010, 0.021] | −0.851 [−1.979, −0.277] | 0.015 | 1.000 |
| Logistic regression | 0.674 [0.575, 0.775] | 0.0472 [0.022, 0.104] | 0.079 [−0.001, 0.176] | 0.110 [0.025, 0.238] | 0.100 [0.022, 0.198] |
| Gradient boosting | 0.723 [0.643, 0.811] | 0.0890 [0.033, 0.199] | 0.221 [0.047, 0.391] | 0.088 [0.037, 0.144] | 0.254 [0.093, 0.419] |

Normalized utility is the official PhysioNet/CinC 2019 score, where 1.0 is the best attainable alert timing and 0.0 is never alerting. The prevalence baseline is far worse than silence because alerting every hour is penalized. **The selected logistic model's utility interval includes zero: on this cohort it is not distinguishable from never alerting at all.**

**Logistic regression was selected by validation average precision**, 0.0942 versus boosting's 0.0824. Boosting's better test result was not used to change this selection. The logistic threshold is 0.224281, giving 1.38 alert hours per 100 test hours [0.69, 2.26], 24 true and 194 false alert hours, and alerts in 5.35% of nonsepsis test patients.

## Site-held-out split

Training and validation patients come from set A only; every set B patient is held out. This is a different-source evaluation, still ICU and still retrospective; it is not external hospital validation of an ER system.

| Baseline | AUROC | Average precision | Normalized utility | Precision | Recall |
|---|---|---|---|---|---|
| Prevalence | 0.500 | 0.0135 [0.010, 0.017] | −1.070 [−1.878, −0.612] | 0.013 | 1.000 |
| Logistic regression | 0.588 [0.508, 0.669] | 0.0271 [0.016, 0.055] | 0.076 [−0.005, 0.157] | 0.057 [0.024, 0.113] | 0.122 [0.053, 0.204] |
| Gradient boosting | 0.690 [0.616, 0.754] | 0.0403 [0.023, 0.080] | 0.073 [−0.016, 0.166] | 0.057 [0.022, 0.116] | 0.130 [0.058, 0.218] |

Validation average precision selected gradient boosting here (0.0497 versus 0.0435), a different model than the random split selected. Discrimination degrades on the held-out site for both learned baselines: logistic AUROC falls from 0.674 to 0.588, with a lower interval bound of 0.508, barely above chance. Both selected-model utility intervals include zero.

## Calibration and recalibration

Raw scores are not calibrated probabilities: logit recalibration on held-out hours gives slopes far below 1 and strongly negative intercepts, meaning the score spread is wider than the evidence supports. A Platt map is now fitted **on validation predictions only**, frozen, and then applied to held-out hours. It is strictly monotone, so it changes reported probabilities but leaves ranking, AUROC, average precision and every alert decision unchanged once the threshold is passed through the same map. Each run records `alerts_identical: true` as a check of that.

| Run | Model | ECE before → after | Slope before → after | Intercept before → after | Brier before → after | Threshold before → after |
|---|---|---|---|---|---|---|
| Random | Logistic | 0.0114 → 0.0035 | 0.491 → 0.728 | −2.077 → −1.056 | 0.01614 → 0.01484 | 0.224281 → 0.095909 |
| Random | Boosting | 0.0136 → 0.0026 | 0.479 → 1.143 | −1.734 → 0.488 | 0.01614 → 0.01455 | 0.060638 → 0.043365 |
| Site-held-out | Logistic | 0.0218 → 0.0083 | 0.110 → 0.346 | −3.786 → −2.906 | 0.01795 → 0.01335 | 0.175774 → 0.046163 |
| Site-held-out | Boosting | 0.0108 → 0.0058 | 0.430 → 1.257 | −2.159 → 0.589 | 0.01446 → 0.01320 | 0.105035 → 0.051258 |

Recalibration removes most of the overextension within the same source: on the random split the selected logistic model's held-out Brier improves from 0.01614 to 0.01484, the first time a learned baseline scores better than the prevalence predictor's 0.01497 on this cohort. **It does not fix the cross-site case.** The logistic model's residual slope on the unseen site is 0.346, still far from 1, so a map fitted on set A does not transfer to set B; across the five site-split seeds the recalibrated Brier beats the prevalence baseline's 0.01334 in only one run. The fitted map is itself seed-dependent — on the random split it lands between 0.67 and 2.09, overshooting past a slope of 1 on two of five seeds. Recalibration cannot change discrimination or which hours alert, so none of these improvements is an operational gain.

Expected calibration error looks small throughout only because almost every hour is negative. Ten equal-count reliability bins per model, before and after, are recorded in each run's `metrics.json`.

## Subgroup description

Held-out hours split by the recorded administrative fields Age, Gender, Unit1 and Unit2, assigned from each patient's first hour before scoring. These are descriptions of one cohort with no intervals and no multiplicity control. Some levels hold only a few dozen positive hours, so the differences below are unstable and are **not** evidence about fairness in care. Gender is the dataset's 0/1 code, kept unlabelled here because it is an administrative record. `unit_unrecorded` is a data-availability level: set B carries no unit indicator at all.

Selected model, random split (logistic):

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 182 | 1.21% | 0.623 | 0.0247 | 0.000 | −0.014 | 0.56 |
| gender_1 | 218 | 1.79% | 0.691 | 0.0645 | 0.158 | 0.130 | 2.08 |
| age_lt_50 | 111 | 1.34% | 0.719 | 0.0486 | 0.246 | 0.192 | 3.02 |
| age_50_64 | 104 | 1.70% | 0.498 | 0.0181 | 0.000 | −0.017 | 0.98 |
| age_65_79 | 129 | 1.66% | 0.776 | 0.1028 | 0.060 | 0.059 | 0.49 |
| age_80_plus | 56 | 1.21% | 0.694 | 0.0989 | 0.179 | 0.155 | 1.04 |
| unit1 | 113 | 0.68% | 0.675 | 0.0127 | 0.000 | −0.131 | 2.80 |
| unit2 | 125 | 1.72% | 0.580 | 0.0291 | 0.012 | 0.010 | 0.43 |
| unit_unrecorded | 162 | 2.00% | 0.761 | 0.1554 | 0.185 | 0.175 | 1.05 |

Selected model, site-held-out split (boosting), on the 987 unseen set B patients:

| Subgroup | Patients | Positive hours | AUROC | Average precision | Recall | Utility | Alerts per 100 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| gender_0 | 445 | 0.74% | 0.688 | 0.0157 | 0.102 | −0.054 | 4.06 |
| gender_1 | 542 | 1.87% | 0.698 | 0.0764 | 0.140 | 0.116 | 2.30 |
| age_lt_50 | 258 | 1.40% | 0.683 | 0.0357 | 0.081 | −0.028 | 6.11 |
| age_50_64 | 298 | 1.39% | 0.690 | 0.0770 | 0.194 | 0.170 | 2.44 |
| age_65_79 | 301 | 1.17% | 0.772 | 0.0749 | 0.185 | 0.125 | 1.94 |
| age_80_plus | 130 | 1.56% | 0.605 | 0.0239 | 0.000 | −0.027 | 1.58 |
| unit1 | 356 | 1.23% | 0.739 | 0.0718 | 0.161 | 0.113 | 3.04 |
| unit2 | 330 | 1.49% | 0.719 | 0.0391 | 0.101 | 0.030 | 4.17 |
| unit_unrecorded | 301 | 1.34% | 0.590 | 0.0417 | 0.133 | 0.081 | 1.99 |

**The aggregate numbers hide subgroups the selected model never helps.** On the random split it alerts on no positive hour at all for gender_0 (182 patients), for age_50_64 (104 patients) and for unit1 (113 patients), while still raising 2.80 false alert hours per 100 for unit1. Across all ten runs, 12 subgroup levels have zero recall, five of them in one site-split seed alone. Negative subgroup utility — worse than staying silent — appears in 8 of 45 random-split levels and 14 of 45 site-split levels, most often for gender_0 (negative in all five site seeds) and age_80_plus (four of five). Subgroup prevalence differs as well, so these gaps confound model behavior with case mix and cannot be read as a disparity estimate.

## Seed sensitivity

Ten runs, five seeds per split. The seed drives both the 2,000-patient download subset and the split, so this varies cohort and split together; it is not a decomposition of the two.

| Split | Seed | Selected | AUROC | Average precision | Utility [95% interval] | Alerts per 100 h |
|---|---:|---|---:|---:|---|---:|
| Random | 42 | logistic | 0.674 | 0.0472 | 0.079 [−0.001, 0.176] | 1.38 |
| Random | 1 | logistic | 0.762 | 0.0559 | 0.097 [0.012, 0.196] | 2.94 |
| Random | 7 | boosting | 0.781 | 0.0947 | 0.394 [0.182, 0.557] | 16.08 |
| Random | 13 | boosting | 0.818 | 0.1229 | 0.305 [0.126, 0.470] | 6.21 |
| Random | 2024 | boosting | 0.730 | 0.0616 | 0.174 [0.002, 0.333] | 7.37 |
| Site | 42 | boosting | 0.690 | 0.0403 | 0.073 [−0.016, 0.166] | 3.11 |
| Site | 1 | boosting | 0.650 | 0.0326 | 0.007 [−0.026, 0.044] | 0.53 |
| Site | 7 | logistic | 0.613 | 0.0257 | 0.027 [−0.018, 0.080] | 1.75 |
| Site | 13 | boosting | 0.639 | 0.0309 | 0.029 [−0.006, 0.078] | 0.93 |
| Site | 2024 | logistic | 0.639 | 0.0422 | 0.151 [0.060, 0.240] | 4.33 |

Three findings, all of which weaken the single-seed report above:

- **The seed-42 random split sits near the pessimistic end of its own range.** Selected-model AUROC spans 0.674–0.818 and utility spans 0.079–0.394 across five seeds. A single run's headline number is not a stable estimate of anything. No seed was chosen after seeing test results; 42 was fixed from the first experiment.
- **Model selection is effectively arbitrary.** The validation average-precision margin between logistic and boosting is 0.0002 at random seed 1 (0.0585 versus 0.0583) and 0.0017 at random seed 7 (0.0295 versus 0.0312). The rule picks logistic twice and boosting three times per split, and its choice is not consistently the model with the better held-out AUROC. The F1-maximizing threshold is just as unstable: alert load ranges from 0.53 to 16.08 alert hours per 100 hours, a thirtyfold spread produced entirely by the seed.
- **Cross-site degradation is the one consistent pattern.** Every site-split seed scores below every random-split seed on AUROC (0.613–0.690 versus 0.674–0.818), and 4 of 5 site-split utility intervals include zero versus 1 of 5 random-split intervals. This survives the seed variation; nothing else here does.

## Verification

- Thirteen tests passed, covering causal feature prefixes, disjoint reproducible patient splits, whole-site holdout, invalid-input rejection, the official utility ramps and normalization, calibration estimates, recalibration monotonicity and alert preservation, subgroup partitioning against directly computed metrics, patient-descriptor bracketing, bootstrap determinism, illustrative policy behavior, and end-to-end batch/replay equivalence.
- `pip check` found no broken requirements.
- **This release changes no model behavior.** For both splits at seed 42, the new runs reproduce the previously reported experiment exactly: identical patient manifests and file hashes, identical held-out predictions to the last bit, and identical replay output apart from the added `calibrated_score` column. The manifests gain four descriptor columns (`site`, `age_band`, `gender`, `unit`) and `metrics.json` gains recalibration and subgroup blocks; every previously reported metric is unchanged.
- A second independent random-split invocation produced byte-identical patient manifests, `metrics.json` including all bootstrap intervals, and selected-model test predictions.
- Historical replay completed for held-out record `site_A/p000119.psv` (17 hourly rows, every one forced to hourly review by missing or stale observations, scores 0.002–0.012) and for `site_B/p100001.psv` under the site-held-out run (24 rows, 11 reviews, hourly review forced for the first six hours and the cadence otherwise driven by low scores). Recalibration raises the displayed probabilities on the site run from 0.002–0.009 to 0.014–0.022 without changing a single review event.

Use the commands in README.md to replay locally or start a larger experiment. Review timing remains an illustrative software policy; it has not been optimized or clinically validated.
