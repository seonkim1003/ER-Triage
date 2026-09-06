# ER-Triage-AI

Local research prototype for adaptive monitoring and reassessment support. **PhysioNet 2019 is ICU data, not ER validation. This prototype is not for patient care.** No paid services, API keys, or cloud training are used.

## Work from another computer

Install Git and Python 3.12 on the laptop, then clone this repository into a local development folder and open that folder in Codex:

```powershell
git clone https://github.com/seonkim1003/ER-Triage.git
cd ER-Triage
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ertriage download --limit 0
.\.venv\Scripts\python.exe -m ertriage train --limit 0 --out artifacts/full-random-42
.\.venv\Scripts\python.exe -m ertriage replay --run artifacts/full-random-42
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` for the remaining Python commands. The dataset, environment and generated models are excluded from Git and recreated on each computer. The committed RESULTS.md records the initial experiment; its local model artifacts are not included.

Before switching computers, commit and push your code changes. On the other computer, run `git pull --ff-only` before starting work. Commit or stash any unfinished local edits before pulling. If Git reports divergent changes, reconcile them before continuing; do not force-push over the other computer's work. Git synchronizes committed project files, not Codex conversation history.

## Run on Windows

An isolated Python 3.12 environment is available in `.venv`. Activation is optional:

```powershell
.\.venv\Scripts\python.exe -m ertriage download
.\.venv\Scripts\python.exe -m ertriage train --limit 0 --out artifacts/full-random-42
.\.venv\Scripts\python.exe -m ertriage train --limit 0 --split site --out artifacts/full-site-42
.\.venv\Scripts\python.exe -m ertriage replay --run artifacts/full-random-42
.\.venv\Scripts\python.exe -m pytest -q
```

`--split site` holds out every patient from one source set instead of random patients, so training never sees the evaluated site. `--draws` sets the number of patient bootstrap resamples used for held-out intervals (default 1000); lower it for a faster run.

`--threshold` picks the validation rule that fixes the operating point and `--select` picks the model. The defaults are `--threshold budget --alert-budget 2.0 --select stable`; `--threshold f1 --select ap` reproduces the configuration used before those rules existed, bit for bit. **The 2.0 alert-hour budget is an arbitrary placeholder, not a clinical capacity** — set `--alert-budget` to a review capacity someone has actually stated before reading anything into the alert load.

`--limit 0` uses all 40,336 patients and is what RESULTS.md now reports; `--limit N` takes a seeded subsample of N. **Subsampled runs are not a cheap preview of the full-cohort answer.** At 2,000 patients the reported model, the subgroup findings and the calibration findings all differed from the full cohort, and in three of four cases the full-cohort point estimate fell outside the subsample's bootstrap interval. Use a subsample to exercise the pipeline, not to draw conclusions.

To vary the split assignment, repeat with several seeds, each in its own output folder:

```powershell
foreach ($s in 42,1,7,13,2024) { .\.venv\Scripts\python.exe -m ertriage train --limit 0 --seed $s --out artifacts/full-random-$s }
```

Below `--limit 0` the seed selects the patient subset as well, so subsampled seeds vary cohort and split at once.

The default replay is the first test patient in sorted manifest order, selected without consulting outcomes. Pick a specific test patient using `--patient site_A/p000001.psv` (use an actual identifier from `test_patients.csv`). Replay prints hourly scores and illustrative review events, and saves CSV output locally.

After the first run, `powershell -ExecutionPolicy Bypass -File .\replay-demo.ps1` launches the demo from any current directory when given the script's full path. This flag applies only to that PowerShell process.

For a new machine, use Python 3.12: `python -m venv .venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`. The lock file records all installed dependencies. Linux/macOS commands use `.venv/bin/python`.

The full dataset is 40,336 records and about 332 MB on disk. Downloading it takes roughly half an hour on a home connection; training then takes about nine minutes for the random split and fourteen for the site-held-out split, and peaks near 4 GB of RAM because features are held in memory. CPU threads are capped at two. Use a new output folder for each experiment; existing runs are never overwritten by training. Download and package installation require internet; training, evaluation, and replay run offline.

## Data and provenance

- Source: [PhysioNet Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/), public training sets A and B, 40,336 patient records in total: 20,336 in set A and 20,000 in set B. All of them are downloaded and used by the reported runs.
- Hourly pipe-delimited records with 40 predictors and `SepsisLabel`. The target is used **as provided**, already shifted six hours before sepsis onset. It remains positive afterwards; it is not an isolated future-event label and scores are not calibrated ER probabilities.
- The downloader reads both official file listings and fetches every record; `--limit N` uniformly selects N of them with the given seed. It downloads individual files because the old ZIP endpoints return 404. Each record is schema-checked before an atomic local rename. Downloads resume by validating existing files; there are six concurrent downloads and bounded retries. Source URLs and locally computed SHA-256 hashes are saved in `data/physionet2019/provenance.json`. These hashes record local provenance, not independent publisher authentication.
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Reyna et al., *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, PhysioNet (2019), [doi:10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857), and the [associated paper](https://doi.org/10.1097/CCM.0000000000004145).
- Downloaded patient records and model artifacts are excluded from version control. No reidentification, uploads, or live patient inputs are implemented.

## Method

The reported experiment uses all 40,336 patient files; `--limit N` instead samples N of them uniformly before looking at labels, using the given seed. Patients are split 60/20/20 into train, validation and test, stratified by whether any row has a positive label. Source-prefixed patient identifiers prevent filename collisions; duplicate selected file contents are rejected. This is patient separation at the dataset's record level, not independent verification of underlying identities.

Features include the 40 current/forward-filled values, current missingness flags, three-hour changes in seven vital signs, and time since each of those vitals was last observed. No backward filling, future aggregates, or labels enter feature construction. Leading missing values remain missing. Logistic-regression medians and scaling are fitted on training patients only. The tree model handles NaNs directly; its internal random-row early stopping is disabled.

Baselines are a prevalence-only predictor, unweighted logistic regression, and histogram gradient boosting. Both remaining choices are made on validation patients and frozen before any test hour is scored. No test tuning or clinical interpretation of the threshold is justified.

The **threshold rule** sets the operating point. `budget` (default) takes the lowest threshold whose validation alert load stays within `--alert-budget` alert hours per 100, making the operating point a stated capacity rather than the argmax of a noisy curve. `f1` maximizes hourly F1 and `utility` maximizes the official patient-weighted utility; RESULTS.md shows the first swings alert load tenfold across seeds and the second buys score by alerting on up to 29% of all hours.

The **selection rule** decides which baseline is reported. `stable` (default) takes the validation average-precision leader, then runs a paired patient bootstrap on validation predictions; when the interval for the average-precision difference includes zero, the leader is not distinguishable from the simpler model and the simpler model is reported, in the prespecified order logistic before boosting. Prevalence stays a reference baseline and is never selected. `ap` reports the bare leader. Across ten runs no validation margin excluded zero, so this is a tie-breaking convention for reproducibility, not evidence that the simpler model generalizes better.

Artifacts include exact patient manifests and file hashes, serialized models, hourly held-out predictions for the selected model, and `metrics.json` with AUROC, average precision, Brier score, confusion counts, precision/recall, alert hours per 100 hours, and the fraction of nonsepsis patients receiving any alert. Manifests also carry each patient's source site and recorded age band, gender code and ICU-unit indicator, which are read from the first hour and used only to describe held-out subgroups. Hourly metrics weight long stays more heavily.

## Evaluation

`ertriage/evaluate.py` describes frozen predictions and selects nothing. The only thing it fits is the recalibration map, from validation predictions alone, before held-out hours are scored.

- **Utility score.** `normalized_utility` implements the official PhysioNet/CinC 2019 scoring: alerting is rewarded on a ramp from twelve to six hours before onset, penalized on a ramp from six hours before to three hours after, staying silent inside that late window is penalized, and each false alert hour costs 0.05. The score is normalized so 1.0 is the best attainable alert timing and 0.0 is never alerting. Because it reuses the provided already-shifted label, it inherits that label's definition and is not an independent outcome.
- **Uncertainty.** `bootstrap` resamples whole test patients with replacement, seeded and reproducible, and reports 95% percentile intervals for AUROC, average precision, utility, precision, recall and alert rate. These intervals describe sampling variation within one cohort. They are not evidence of generalization to another hospital, to an ER, or to future care.
- **Calibration.** `calibration` reports ten equal-count reliability bins, expected calibration error, and the intercept and slope of a logit recalibration fit. A calibrated score would give intercept 0 and slope 1; the observed slopes are well below 1. Model outputs are not calibrated probabilities and must not be read as risk of sepsis.
- **Selection margins.** `paired_difference` gives a percentile interval for the average-precision difference between two frozen score vectors, resampling the same validation patients under both. It answers whether a selection margin survives resampling; it is not a hypothesis test and says nothing about which model generalizes.
- **Recalibration.** `recalibrator` fits a Platt map on validation predictions only; `recalibrate` applies the frozen map to held-out scores and to the threshold. The map is strictly monotone, so it changes reported probabilities, Brier score and reliability but never the ranking, the discrimination metrics, or which hours alert. Each run records `alerts_identical` as a check. Recalibrated output is still an ICU development score, not a probability of sepsis for an ER patient.
- **Subgroup description.** `subgroup_metrics` reports held-out discrimination, utility, recall and alert load for patient groups assigned from recorded administrative fields before scoring. It fits and selects nothing, reports no intervals, and applies no multiplicity control. Levels can be small and prevalence differs between them, so these are exploratory descriptions of one cohort, not subgroup validation and not evidence about fairness in care.
- **Site-held-out evaluation.** `--split site` holds out an entire source set, so the evaluated patients come from a source the models never saw. This is still ICU data and still retrospective; it is a different-source check, not external hospital or ER validation.

RESULTS.md reports the full-cohort runs and records what training on all 40,336 patients retracted. The subgroup disparities and the severe miscalibration reported from the 2,000-patient subsample were small-sample artifacts and do not survive; the apparent tie between the two learned baselines was a power problem, not equivalence. Cross-site degradation survived every cohort size and configuration: held-out AUROC falls from 0.827 within source to 0.765 on an unseen source, on non-overlapping intervals, and scores carried across sites remain systematically too high. Prospective validation and any clinical assessment remain unimplemented.

## Historical replay

Replay accepts held-out patients only and recomputes each prediction from the visible history prefix. Future observations and retrospective labels are excluded from scoring and scheduling. Labels are displayed only for retrospective comparison.

Replay also prints a `calibrated_score` column when the run recorded a recalibration map. Because that map is monotone, it changes no review event. An illustrative policy schedules review after 1, 2, or 4 hours based on score, score increase, and observation age. Any vital absent or at least four hours old forces a one-hour review. Review events can occur early when scores rise. All recorded hourly measurements still arrive in this simulation: the policy changes review events, **not measurement acquisition**, and cannot estimate outcomes under a different monitoring schedule. It does not make treatment, triage, discharge, or real-world timing recommendations. Neither policy thresholds nor review intervals have clinical validation.

## Layout

- `ertriage/data.py`: download, schema checks and causal features.
- `ertriage/model.py`: patient splits, baselines, threshold rules, model selection and evaluation.
- `ertriage/evaluate.py`: official utility scoring, calibration, validation-fitted recalibration, subgroup description, selection margins and patient bootstrap.
- `ertriage/replay.py`: prefix-only replay and illustrative cadence.
- `tests/test_pipeline.py`: leakage, split, utility, calibration, recalibration, subgroup, threshold-rule, selection, bootstrap, validation and policy regression tests.
- `artifacts/`: local run outputs; `data/`: local records and download provenance.

Only load this project's trusted local `.joblib` files: the serialization format can execute code. Site-held-out evaluation, patient-bootstrap uncertainty, calibration assessment, held-out recalibration, subgroup description, multi-seed sensitivity and official utility scoring are now implemented. Remaining work: a review capacity stated by someone qualified to state one, cross-site recalibration that actually transfers, subgroup comparisons stated in advance with intervals and multiplicity control, a target that is an isolated future event rather than the provided persistent label, clinician-designed policies, and genuinely ER-specific retrospective validation.
