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

Audit review workload across every held-out patient using an existing run:

```powershell
.\.venv\Scripts\python.exe -m ertriage workload --run artifacts/full-random-42 --out artifacts/workload-full-random-42
.\.venv\Scripts\python.exe -m ertriage workload --run artifacts/full-site-42 --out artifacts/workload-full-site-42
```

Each audit writes `WORKLOAD.md`, `workload.json`, and `patient_workload.csv` into a new directory. It compares the existing adaptive policy with fixed 1-, 2-, and 4-hour schedules, all starting at the first recorded hour. It reports review counts, missing/stale-observation workload, and paired patient-bootstrap intervals (`--draws`, default 1000; `--seed`, default 42). Patient files must match the training manifest hashes, test patients must be separate from development patients, and prediction counts and ordered labels must match the source. New training runs also save explicit hour keys for alignment checks; older runs rely on their saved within-patient row order. No models or thresholds are fitted and no policy is selected by this audit.

**Review events and alert hours are different quantities.** The alert budget constrains threshold crossings on validation data; it does not constrain the adaptive scheduler, whose missingness rule can force hourly reviews. Fixed schedules ignore score changes and missingness. These comparisons measure simulated review counts, not staff time, outcome effects, or which policy is clinically preferable. All original measurements continue to arrive.

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

The **threshold rule** sets the operating point. `budget` (default) takes the lowest observed threshold whose validation alert load stays within `--alert-budget` alert hours per 100. If even the highest tied score exceeds the budget, it chooses a finite threshold just above 1, meaning no alerts; recalibration preserves that decision. The budget is only a validation constraint and held-out alert rates can exceed it. `f1` maximizes hourly F1 and `utility` maximizes the official patient-weighted utility; RESULTS.md shows the first swings alert load tenfold across seeds and the second buys score by alerting on up to 29% of all hours.

The **selection rule** decides which baseline is reported. `stable` (default) takes the validation average-precision leader, then runs a paired patient bootstrap on validation predictions; when the interval for the average-precision difference includes zero, the leader is not distinguishable from the simpler model and the simpler model is reported, in the prespecified order logistic before boosting. Prevalence stays a reference baseline under `stable`. `ap` reports the bare leader. The ten earlier subsampled runs could not resolve a margin, while both full-cohort runs selected boosting with intervals excluding zero. This is a selection convention, not evidence that an unresolved comparison establishes equivalence.

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

### Visual dashboard and early-warning evaluation

The local dashboard shows recorded vitals with missing-measurement gaps, frozen model scores, threshold alerts, and the existing adaptive review schedule. It supports play/pause, one-hour stepping, a keyboard-accessible time slider, patient search and previous/next navigation, seven vital-sign traces, and an hourly values table. Ordinary replay displays the current history prefix. An explicit retrospective toggle reveals the full record, labels, the onset proxy and warning window, plus first-alert and repeated-episode counts. Loading another patient resets the toggle. The first record is selected by sorted identifier without inspecting outcomes.

```powershell
.\.venv\Scripts\python.exe -m ertriage early-warning --run artifacts/full-random-42 --out artifacts/early-warning-full-random-42
.\.venv\Scripts\python.exe -m ertriage early-warning --run artifacts/full-site-42 --out artifacts/early-warning-full-site-42
.\.venv\Scripts\python.exe -m ertriage dashboard --run artifacts/full-random-42 --evaluation artifacts/early-warning-full-random-42/early_warning.json
```

Open [the local dashboard](http://127.0.0.1:8765) after the server starts. Stop with Ctrl+C; `--port` selects another port. `dashboard-demo.ps1` runs the viewer from any working directory and includes the matching evaluation when available. Without `--evaluation`, patient replay still works. The viewer binds only to `127.0.0.1`, serves a fixed set of assets and read-only APIs, and uses no CDN, paid service, extra package, or live patient input. It reads saved predictions without loading serialized models. This is visual playback of frozen research outputs, not newly computed live inference.

Early-warning evaluation writes `EARLY_WARNING.md`, `early_warning.json`, and `patient_warnings.csv` to a new directory. It does not tune the frozen model or threshold. `--draws` and `--seed` control a whole-patient bootstrap (defaults: 1000 and 42). Definitions:

- **Onset proxy:** the first observed 0-to-1 transition in the persistent dataset label, plus six hours. The label has already been shifted by six hours according to [PhysioNet's definition](https://physionet.org/content/challenge-2019/1.0.0/). This is not an independently observed onset.
- **Timing denominator:** positive-label patients with a recoverable transition, the onset proxy inside the recorded stay, and a fully observed 12-to-6-hour pre-onset window. Positive-at-start records, nonpersistent labels, onset beyond the record, and incomplete windows are counted separately and excluded from timing fractions.
- **Early warning:** any threshold-positive hour in the inclusive `[onset-12, onset-6]` window. An alert episode already in progress counts if it overlaps the window. This analysis window is not a clinician-specified requirement.
- **Before-onset detection:** any threshold-positive hour from `onset-12` up to but excluding onset. Very early, unrelated alerts and alerts at/after onset do not count.
- **Alert episodes:** contiguous threshold-positive hours. At least one threshold-negative hour separates episodes. Repeated episodes are all episodes after the first within each stay; no clinical cooldown is assumed.
- **Misses and false alerts:** missed timing windows among eligible positive-label patients, and any alert among patients whose recorded labels remain zero. These describe recorded labels and must not be read as clinical diagnoses or missed diagnoses.
- **Lead time:** per-patient first-alert timing is retained, and early-window lead-time summaries include detected eligible patients only. They exclude misses and can be affected by alerts already active at the start of the window.

The early-warning and dashboard readers check patient hashes, split separation, score validity, row counts and ordered labels; explicit hour keys are checked where present. Older prediction files still rely on their original within-patient row order. A supplied cohort evaluation must match the loaded run's source hashes.

Replay accepts held-out patients only and recomputes each prediction from the visible history prefix. Future observations and retrospective labels are excluded from scoring and scheduling. Labels are displayed only for retrospective comparison.

Replay also prints a `calibrated_score` column when the run recorded a recalibration map. Because that map is monotone, it changes no review event. An illustrative policy schedules review after 1, 2, or 4 hours based on score, score increase, and observation age. Any vital absent or at least four hours old forces a one-hour review. Review events can occur early when scores rise. All recorded hourly measurements still arrive in this simulation: the policy changes review events, **not measurement acquisition**, and cannot estimate outcomes under a different monitoring schedule. It does not make treatment, triage, discharge, or real-world timing recommendations. Neither policy thresholds nor review intervals have clinical validation.

### Event-anchored target

The dataset label is *persistent*: once it turns 1 it stays 1 until the record ends. Training and
scoring on it mixes "will this patient deteriorate?" with "is this patient already deteriorating?".
It also contradicts the early-warning metric above: in the `[onset-12, onset-6]` window that metric
scores, the persistent label calls almost every hour negative, so the model is fitted to stay silent
exactly where it is then measured.

`--target event` replaces that label. An hour is positive when the onset proxy falls within the next
`--horizon` hours, and every hour at or after the onset proxy is dropped from fitting, thresholding
and reported metrics. Excluded are records whose label is nonpersistent or already positive at the
first recorded hour, because their transition cannot be recovered; records whose onset proxy falls
past the end of the stay are kept, since all of their hours are genuinely pre-onset. Every recorded
hour is still scored and written to `test_predictions.csv`, so the timing, workload, replay and
dashboard tools read a complete record either way.

```powershell
.\.venv\Scripts\python.exe -m ertriage train --out artifacts/event12-random-42 --limit 0 --split random --target event --horizon 12
.\.venv\Scripts\python.exe -m ertriage basis --run artifacts/full-random-42 --out artifacts/basis-full-random-42
.\.venv\Scripts\python.exe -m ertriage contrasts --run artifacts/full-random-42 --out artifacts/contrasts-full-random-42
```

The official PhysioNet utility is defined against the persistent label and its timing, so it is left
undefined for this target rather than recomputed on a different one. `--threshold utility` is refused
for the same reason.

`basis` re-summarizes a frozen run's saved scores twice, over all recorded hours against the
persistent label and over pre-onset hours only against the event label, with a whole-patient
bootstrap on the difference. It refits nothing and loads no `.joblib`. At horizon 6 the two labels
coincide on pre-onset hours by construction, which the report verifies elementwise; the difference
there measures the dropped post-onset hours and nothing else.

### Prespecified subgroup contrasts

The per-level subgroup descriptions in a training report carry no intervals and no multiplicity
control, and are labelled that way. `contrasts` states the comparisons instead: a descriptor family
fixed in code before any held-out hour is read, one level-versus-rest AUROC difference per level with
a whole-patient bootstrap interval, and Holm-Bonferroni across the whole family. A descriptor with
exactly two levels contributes one contrast, not two mirrored ones. It reads a frozen run, so it can
be applied to runs trained before it existed. It remains a post-hoc description of one cohort.

## Layout

- `ertriage/data.py`: download, schema checks and causal features.
- `ertriage/model.py`: patient splits, baselines, threshold rules, model selection and evaluation.
- `ertriage/evaluate.py`: official utility scoring, calibration, validation-fitted recalibration, subgroup description, selection margins and patient bootstrap.
- `ertriage/target.py`: the event-anchored prediction target and its pre-onset mask.
- `ertriage/basis.py`: re-scores a frozen run on the pre-onset basis without refitting.
- `ertriage/contrasts.py`: prespecified subgroup contrasts with intervals and Holm correction.
- `ertriage/replay.py`: prefix-only replay and illustrative cadence.
- `ertriage/workload.py`: held-out review workload, artifact checks, fixed schedule comparisons and paired patient-bootstrap intervals.
- `ertriage/history.py`: verified access to frozen held-out patient histories.
- `ertriage/early_warning.py`: onset-proxy timing, missed windows, false alerts, repeated episodes and patient-bootstrap intervals.
- `ertriage/dashboard.py` and `ertriage/static/`: local browser replay and evaluation viewer.
- `tests/test_early_warning.py`: timing boundaries, exclusions, denominator checks, artifact checks and local HTTP tests.
- `tests/test_target.py`: the horizon-6 label identity, pre-onset masking, and the proof that
  masking late equals truncating first.
- `tests/test_contrasts.py`: Holm correction, planted and null subgroup differences.
- `tests/test_basis.py`: pre-onset re-scoring, the horizon-6 identity, and contrast artifacts.
- `tests/test_workload.py`: budget ties, causal scheduling, workload intervals and artifact-integrity regression tests.
- `tests/test_pipeline.py`: leakage, split, utility, calibration, recalibration, subgroup, threshold-rule, selection, bootstrap, validation and policy regression tests.
- `artifacts/`: local run outputs; `data/`: local records and download provenance.

Only load this project's trusted local `.joblib` files: the serialization format can execute code. Site-held-out evaluation, patient-bootstrap uncertainty, calibration assessment, held-out recalibration, subgroup description, multi-seed sensitivity and official utility scoring are now implemented. Prespecified subgroup contrasts with intervals and Holm correction, and an event-anchored target that is not the provided persistent label, are now implemented. Remaining work: a review capacity stated by someone qualified to state one, cross-site recalibration that actually transfers, clinician-designed policies, and genuinely ER-specific retrospective validation.
