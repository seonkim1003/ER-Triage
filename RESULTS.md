# Local experiments

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation. Every number below is a development metric; none establishes clinical utility, safety, or ER performance.

## Cohort

Uniform seeded download of 2,000 patients from the 40,336 records listed in the two official PhysioNet 2019 training sets: 1,013 from set A and 987 from set B. No full-cohort training has been run.

| Run | Split | Train | Validation | Test | Test hours | Positive test hours |
|---|---|---:|---:|---:|---:|---:|
| `artifacts/baseline` | random patients | 1,200 | 400 | 400 | 15,789 | 240 (1.52%) |
| `artifacts/site-holdout` | all of set B held out | 759 | 254 | 987 | 37,592 | 507 (1.35%) |

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

## Calibration

Scores are not calibrated probabilities. Logit recalibration on held-out hours gives slopes far below 1 and strongly negative intercepts, meaning the score spread is wider than the evidence supports.

| Run | Model | Expected calibration error | Recalibration slope | Recalibration intercept | Brier |
|---|---|---:|---:|---:|---:|
| Random | Logistic | 0.0114 | 0.491 | −2.077 | 0.01614 |
| Random | Boosting | 0.0136 | 0.479 | −1.734 | 0.01614 |
| Site-held-out | Logistic | 0.0218 | 0.110 | −3.786 | 0.01795 |
| Site-held-out | Boosting | 0.0108 | 0.430 | −2.159 | 0.01446 |

The logistic model's cross-site slope of 0.110 indicates its score ordering carries little usable magnitude information on the unseen site. Expected calibration error looks small only because almost every hour is negative; both learned baselines still have a worse Brier score than the prevalence predictor on the random split. Ten equal-count reliability bins per model are recorded in each run's `metrics.json`.

## Verification

- Ten tests passed, covering causal feature prefixes, disjoint reproducible patient splits, whole-site holdout, invalid-input rejection, the official utility ramps and normalization, calibration estimates, bootstrap determinism, illustrative policy behavior, and end-to-end batch/replay equivalence.
- `pip check` found no broken requirements.
- A second independent random-split invocation produced byte-identical patient manifests, `metrics.json` including all bootstrap intervals, and selected-model test predictions. Runs are in `artifacts/baseline` and `artifacts/reproducibility`.
- The random-split run reproduces the earlier experiment exactly: patient manifests and test predictions are byte-identical to the previous baseline, and every previously reported metric is unchanged. The new fields are additions, not revisions.
- Historical replay completed for held-out record `site_A/p000119.psv` (17 hourly rows, `artifacts/baseline/replay_p000119.csv`) and for `site_B/p100001.psv` under the site-held-out run. Missing or stale observations force hourly review throughout both examples, despite low model scores.

Use the commands in README.md to replay locally or start a larger experiment. Review timing remains an illustrative software policy; it has not been optimized or clinically validated.
