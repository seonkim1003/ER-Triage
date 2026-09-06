# Initial local experiment

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation.

## Cohort

Uniform seeded download of 2,000 patients from the 40,336 records listed in the two official PhysioNet 2019 training sets. Patient split: 1,200 train / 400 validation / 400 test. Training contains 45,480 hours; validation 15,193 hours; test 15,789 hours. Positive test hours: 240 (1.52%). No full-cohort training has been run.

## Held-out results

| Baseline | AUROC | Average precision | Brier score | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Prevalence | 0.500 | 0.0152 | 0.01497 | 0.015 | 1.000 |
| Logistic regression | 0.674 | 0.0472 | 0.01614 | 0.110 | 0.100 |
| Gradient boosting | 0.723 | 0.0890 | 0.01614 | 0.088 | 0.254 |

Thresholds were chosen by validation hourly F1, separately for each model. The prevalence baseline's selected threshold flags every hour; it is a reference, not a useful alert policy.

**Logistic regression was selected by validation average precision**, 0.0942 versus boosting's 0.0824. Boosting's better test result was not used to change this selection. The logistic threshold is 0.224281, with 1.38 alert hours per 100 test hours and alerts in 5.35% of nonsepsis test patients. Its low recall and worse Brier score than the prevalence predictor are significant limitations. No calibrated clinical probability or useful clinical performance is established.

## Verification

- Six tests passed, including causal feature prefixes, disjoint reproducible patient splits, invalid-input rejection, illustrative policy behavior, and end-to-end batch/replay equivalence.
- `pip check` found no broken requirements.
- A second independent training invocation produced byte-identical train/validation/test manifests, metrics JSON, and selected-model test predictions. Runs are in `artifacts/baseline` and `artifacts/reproducibility`.
- Historical replay completed for held-out record `site_A/p000119.psv`. Its 17 hourly rows are in `artifacts/baseline/replay_p000119.csv`. Missing/stale observations cause hourly review throughout this example, despite low model scores.

Use the commands in README.md to replay locally or start a larger experiment. Review timing remains an illustrative software policy; it has not been optimized or clinically validated.
