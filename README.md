# ER-Triage-AI

Local research prototype for adaptive monitoring and reassessment support. **PhysioNet 2019 is ICU data, not ER validation. This prototype is not for patient care.** No paid services, API keys, or cloud training are used.

## Work from another computer

Install Git and Python 3.12 on the laptop, then clone this repository into a local development folder and open that folder in Codex:

```powershell
git clone https://github.com/seonkim1003/ER-Triage.git
cd ER-Triage
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ertriage download
.\.venv\Scripts\python.exe -m ertriage train --out artifacts/baseline
.\.venv\Scripts\python.exe -m ertriage replay
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` for the remaining Python commands. The dataset, environment and generated models are excluded from Git and recreated on each computer. The committed RESULTS.md records the initial experiment; its local model artifacts are not included.

Before switching computers, commit and push your code changes. On the other computer, run `git pull --ff-only` before starting work. Commit or stash any unfinished local edits before pulling. If Git reports divergent changes, reconcile them before continuing; do not force-push over the other computer's work. Git synchronizes committed project files, not Codex conversation history.

## Run on Windows

An isolated Python 3.12 environment is available in `.venv`. Activation is optional:

```powershell
.\.venv\Scripts\python.exe -m ertriage download
.\.venv\Scripts\python.exe -m ertriage train --limit 2000 --out artifacts/baseline
.\.venv\Scripts\python.exe -m ertriage train --limit 2000 --split site --out artifacts/site-holdout
.\.venv\Scripts\python.exe -m ertriage replay --run artifacts/baseline
.\.venv\Scripts\python.exe -m pytest -q
```

`--split site` holds out every patient from one source set instead of random patients, so training never sees the evaluated site. `--draws` sets the number of patient bootstrap resamples used for held-out intervals (default 1000); lower it for a faster run.

The default replay is the first test patient in sorted manifest order, selected without consulting outcomes. Pick a specific test patient using `--patient site_A/p000001.psv` (use an actual identifier from `test_patients.csv`). Replay prints hourly scores and illustrative review events, and saves CSV output locally.

After the baseline run, `powershell -ExecutionPolicy Bypass -File .\replay-demo.ps1` launches the demo from any current directory when given the script's full path. This flag applies only to that PowerShell process.

For a new machine, use Python 3.12: `python -m venv .venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`. The lock file records all installed dependencies. Linux/macOS commands use `.venv/bin/python`.

For the full dataset, first run `.\.venv\Scripts\python.exe -m ertriage download --limit 0`, then `.\.venv\Scripts\python.exe -m ertriage train --limit 0 --out artifacts/full`. This requires substantially more RAM and time: features are held in memory. CPU threads are capped at two. Use a new output folder for each experiment; existing runs are never overwritten by training. Download and package installation require internet; training, evaluation, and replay run offline.

## Data and provenance

- Source: [PhysioNet Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/), public training sets A and B, 40,336 patient records in total.
- Hourly pipe-delimited records with 40 predictors and `SepsisLabel`. The target is used **as provided**, already shifted six hours before sepsis onset. It remains positive afterwards; it is not an isolated future-event label and scores are not calibrated ER probabilities.
- The downloader reads both official file listings and uniformly selects 2,000 records with seed 42 by default. It downloads individual files because the old ZIP endpoints return 404. Each record is schema-checked before an atomic local rename. Downloads resume by validating existing files; there are six concurrent downloads and bounded retries. Source URLs and locally computed SHA-256 hashes are saved in `data/physionet2019/provenance.json`. These hashes record local provenance, not independent publisher authentication.
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Reyna et al., *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, PhysioNet (2019), [doi:10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857), and the [associated paper](https://doi.org/10.1097/CCM.0000000000004145).
- Downloaded patient records and model artifacts are excluded from version control. No reidentification, uploads, or live patient inputs are implemented.

## Method

The default experiment uniformly samples 2,000 patient files before looking at labels, using seed 42. Patients are split 60/20/20 into train, validation and test, stratified by whether any row has a positive label. Source-prefixed patient identifiers prevent filename collisions; duplicate selected file contents are rejected. This is patient separation at the dataset's record level, not independent verification of underlying identities.

Features include the 40 current/forward-filled values, current missingness flags, three-hour changes in seven vital signs, and time since each of those vitals was last observed. No backward filling, future aggregates, or labels enter feature construction. Leading missing values remain missing. Logistic-regression medians and scaling are fitted on training patients only. The tree model handles NaNs directly; its internal random-row early stopping is disabled.

Baselines are a prevalence-only predictor, unweighted logistic regression, and histogram gradient boosting. Each threshold maximizes hourly F1 on validation patients; the selected model maximizes validation average precision. These choices are frozen before evaluating test patients. No test tuning or clinical interpretation of the threshold is justified.

Artifacts include exact patient manifests and file hashes, serialized models, hourly held-out predictions for the selected model, and `metrics.json` with AUROC, average precision, Brier score, confusion counts, precision/recall, alert hours per 100 hours, and the fraction of nonsepsis patients receiving any alert. Hourly metrics weight long stays more heavily.

## Evaluation

`ertriage/evaluate.py` describes frozen predictions only; it fits nothing and selects nothing.

- **Utility score.** `normalized_utility` implements the official PhysioNet/CinC 2019 scoring: alerting is rewarded on a ramp from twelve to six hours before onset, penalized on a ramp from six hours before to three hours after, staying silent inside that late window is penalized, and each false alert hour costs 0.05. The score is normalized so 1.0 is the best attainable alert timing and 0.0 is never alerting. Because it reuses the provided already-shifted label, it inherits that label's definition and is not an independent outcome.
- **Uncertainty.** `bootstrap` resamples whole test patients with replacement, seeded and reproducible, and reports 95% percentile intervals for AUROC, average precision, utility, precision, recall and alert rate. These intervals describe sampling variation within one cohort. They are not evidence of generalization to another hospital, to an ER, or to future care.
- **Calibration.** `calibration` reports ten equal-count reliability bins, expected calibration error, and the intercept and slope of a logit recalibration fit. A calibrated score would give intercept 0 and slope 1; the observed slopes are well below 1. Model outputs are not calibrated probabilities and must not be read as risk of sepsis.
- **Site-held-out evaluation.** `--split site` holds out an entire source set, so the evaluated patients come from a source the models never saw. This is still ICU data and still retrospective; it is a different-source check, not external hospital or ER validation.

RESULTS.md records both runs. Subgroup validation, prospective validation and any clinical assessment remain unimplemented. Scores and thresholds still need recalibration and assessment before any downstream study.

## Historical replay

Replay accepts held-out patients only and recomputes each prediction from the visible history prefix. Future observations and retrospective labels are excluded from scoring and scheduling. Labels are displayed only for retrospective comparison.

An illustrative policy schedules review after 1, 2, or 4 hours based on score, score increase, and observation age. Any vital absent or at least four hours old forces a one-hour review. Review events can occur early when scores rise. All recorded hourly measurements still arrive in this simulation: the policy changes review events, **not measurement acquisition**, and cannot estimate outcomes under a different monitoring schedule. It does not make treatment, triage, discharge, or real-world timing recommendations. Neither policy thresholds nor review intervals have clinical validation.

## Layout

- `ertriage/data.py`: download, schema checks and causal features.
- `ertriage/model.py`: patient splits, baselines, threshold selection and evaluation.
- `ertriage/evaluate.py`: official utility scoring, calibration and patient bootstrap.
- `ertriage/replay.py`: prefix-only replay and illustrative cadence.
- `tests/test_pipeline.py`: leakage, split, utility, calibration, bootstrap, validation and policy regression tests.
- `artifacts/`: local run outputs; `data/`: local records and download provenance.

Only load this project's trusted local `.joblib` files: the serialization format can execute code. Site-held-out evaluation, patient-bootstrap uncertainty, calibration assessment and official utility scoring are now implemented. Remaining work: recalibration and its held-out assessment, subgroup analysis, full-cohort and multi-seed runs, clinician-designed policies, and genuinely ER-specific retrospective validation.
