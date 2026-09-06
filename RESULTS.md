# Local experiments

Completed on 2026-09-05 using Python 3.12.14 and the pinned environment. This is retrospective ICU development, not ER validation. Every number below is a development metric; none establishes clinical utility, safety, or ER performance.

## Event-anchored target added 2026-09-06

The dataset label is persistent: once it turns 1 it stays 1 until the record ends. Two consequences were measured, one of which reverses a hypothesis this section was written to test.

**The training target contradicted the evaluation window.** The early-warning metric above scores alerts in the inclusive seven-hour window `[onset-12, onset-6]`. The persistent label turns 1 at `onset-6`, the window's *final* hour. So exactly one of those seven hours is positive during training: 421 of 2,947 window hours on the random split and 802 of 5,614 across sites, 14.3% in both because the ratio is structural rather than empirical. The model was fitted to stay silent in 85.7% of the window it is then measured on. `--target event --horizon 12` removes the contradiction by calling every hour within twelve hours of the onset proxy positive.

**Removing post-onset hours costs little.** A frozen model can score well on the persistent label by recognizing physiology that has already declared itself. Re-summarizing the existing frozen scores over pre-onset hours only, refitting nothing, costs 0.0094 AUROC [0.0058, 0.0128] on the random split and 0.0115 [0.0080, 0.0148] across sites. The concern is real and the interval excludes zero, but it is small, because these records end close to the onset proxy: only 1,830 of 308,791 hours and 3,429 of 760,117 lie at or after it. At horizon 6 the event label restricted to pre-onset hours is provably identical to the persistent label restricted to the same hours, and `basis` verifies this elementwise, so that difference is attributable to the dropped hours alone and not to a change of label definition.

**Retraining on the corrected target did not buy earlier warning per alert hour.** This was the expected result and it did not hold. At the default 2% budget on the random split, early-window coverage went from 25.4% [21.3, 29.8] to 22.8% [19.0, 26.7]. Because the two runs selected different model families, the comparison was repeated with the family held fixed and the splits identical, and the direction survived. Nominal budgets also do not produce equal test alert loads across targets, so thresholds were swept over a prespecified validation budget grid and both curves interpolated to common burdens. No test hour informs any threshold.

| Matched on | Comparisons | Event target better | Mean difference | Range |
|---|---:|---:|---:|---:|
| Alert hours per 100 | 11 | 1 | -1.11pp | -3.3 to +1.9 |
| Nonsepsis patients alerted | 12 | 12 | +4.84pp | +1.5 to +9.8 |

Both splits at seed 42, both model families. `sweep` produces this table; the section below varies the seed and the horizon, because one seed at one horizon established nothing on its own.

**The target changes how a fixed alert budget is spread, not how much it detects.** Matched on alert hours the persistent target is equal or slightly ahead almost everywhere. Matched on how many patients are disturbed, the event target is ahead in every one of twelve comparisons, across both splits and both model families. It concentrates the same alert hours onto fewer distinct patients and fewer episodes: on the random split at a 2% budget, boosting alerts on 0.92% of nonsepsis patients against 1.80%, with 0.276 episodes per 100 hours against 0.414.

Which of those two denominators is the real constraint is a review-capacity question, and it remains the case that nobody qualified has stated one. **Neither target is therefore preferable on this evidence**; they are preferable under different and unstated denominators. The site-split result at the default budget looks like a clear win for the event target, 24.7% [21.8, 27.8] against 19.3% [16.7, 22.0], but that comparison is not like-for-like: it spends 2.51 alert hours per 100 against 1.44. The matched-burden table is the honest version and shows a smaller, denominator-dependent effect.

The official PhysioNet utility is defined against the persistent label and its timing, so it is left undefined for this target rather than recomputed against a different one, and `--threshold utility` is refused. Every recorded hour is still scored and saved, so the timing, workload, replay and dashboard tools read complete records under either target. Reproduce the table with `sweep --run artifacts/full-random-42 --against artifacts/event12-random-42`. Local outputs: `artifacts/event12-*-42`, `artifacts/basis-full-*-42` and `artifacts/sweep-full-vs-event12-*`.

## Seed and horizon sensitivity of the burden finding

This project has retracted findings twice for resting on one sample, and its own seed sweep found model selection flipping on validation margins of 0.0002. The matched-burden result above was one seed at one horizon, so it was not established. Both knobs were varied.

Four independent random splits, event target at horizon 12 against the persistent target. Each seed repartitions train, validation and test, so these are different cohorts and not resamples of one.

| Seed | Hours: better | Mean | Patients: better | Mean | Range |
|---|---:|---:|---:|---:|---:|
| 42 | 1/6 | -0.76pp | 6/6 | +6.40pp | +4.3 to +9.8 |
| 1 | 0/6 | -1.55pp | 6/6 | +7.98pp | +3.5 to +12.9 |
| 7 | 3/6 | -0.51pp | 6/6 | +3.53pp | +1.8 to +5.1 |
| 13 | 0/6 | -1.43pp | 6/6 | +5.31pp | +3.8 to +7.0 |

**The direction is seed-stable.** Every seed puts the event target ahead on all six patient-matched comparisons and behind or level on most hours-matched ones. The size moves, +3.53 to +7.98 points, which is the honest width of the effect.

Horizon, at seed 42 on the random split:

| Horizon | Hours: better | Mean | Patients: better | Mean |
|---|---:|---:|---:|---:|
| 6 | 3/6 | +0.28pp | 6/6 | +4.36pp |
| 8 | 2/6 | -0.14pp | 6/6 | +5.80pp |
| 12 | 1/6 | -0.76pp | 6/6 | +6.40pp |
| 24 | 1/6 | -1.96pp | 6/6 | +9.75pp |

**Both denominators respond monotonically to the horizon**, in opposite directions, across every step tested. That is a dose-response to a parameter set deliberately, which is harder to explain as chance than any single comparison, and it turns the finding into a dial rather than a setting. Horizon 6 is roughly free on alert hours while still gaining on patients; horizon 24 buys the most on patients and costs the most on hours.

Horizon 6 also isolates the mechanism. At that horizon the event label restricted to pre-onset hours is provably identical to the persistent label there, so training differs in exactly one respect: hours at or after the onset proxy are dropped from the training set. Nothing is relabelled. That alone earns +4.36 points on the patient denominator. Excluding post-onset hours from fitting is doing most of the work, and widening the positive window adds the rest.

Across every configuration run here, two splits, four seeds, four horizons and both model families, the event target is ahead on **48 of 48** patient-matched comparisons and on **10 of 47** hours-matched ones. Neither target is thereby preferable: the two denominators disagree, and which one binds is the review capacity nobody qualified has stated. What has changed is that the disagreement is now measured and stable rather than a single observation.

## Review workload barely notices the target

The findings above are about which hours alert. The review policy is a different thing: it schedules review from scores, score changes and observation age, and any vital absent or four hours stale forces review regardless of score. Running the existing audit against the event-target runs asks whether a target that halves alerted patients also lightens review.

| Split | Persistent, reviews/100h [95%] | Event target, reviews/100h [95%] | Difference | Hours forced by stale vitals |
|---|---:|---:|---:|---:|
| random | 55.66 [55.05, 56.29] | 55.48 [54.89, 56.12] | -0.19 | 36.8% |
| site | 44.86 [44.63, 45.11] | 45.56 [45.27, 45.85] | +0.70 | 22.4% |

**It does not.** The adaptive review rate moves by well under a percent in either direction, while the same models differ by roughly a factor of two in how many nonsepsis patients they alert on. The reason is the last column: review is dominated by missing and stale observations, which no model changes. This sharpens the earlier finding that a 2% alert-hour budget is not a review-event budget. Alert burden and review burden are close to independent here, so improving one should not be expected to move the other, and a review capacity stated in reviews would not constrain the alert threshold much at all. Local outputs: `artifacts/workload-event12-random-42` and `artifacts/workload-event12-site-42`.

## Cross-site recalibration measured, not asserted

"Cross-site recalibration that actually transfers" sat on the remaining work list as a goal with no measurement attached. `transfer` fits a Platt map on k patients from the held-out site and scores the held-out patients that sample never touched, twenty draws per k, against two references: no correction at all, and the run's own frozen source-fitted map. Nothing is refitted and no model file is loaded.

Calibration slope, median [interquartile range across draws]. A slope below 1 means scores that are too extreme; 20,000 unseen-site patients and 8,068 within-source patients.

| Patients used to fit | Site: uncorrected | Site: source map | Site: fitted on target | Random: uncorrected | Random: source map | Random: fitted on target |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.788 | 0.802 | 0.607 [0.379, 0.827] | 0.902 | 0.986 | 0.639 |
| 50 | 0.788 | 0.802 | 1.258 [0.849, 1.711] | 0.903 | 0.987 | 0.877 |
| 100 | 0.788 | 0.802 | 0.891 [0.660, 1.120] | 0.904 | 0.988 | 0.955 |
| 250 | 0.788 | 0.803 | 1.073 [0.897, 1.293] | 0.901 | 0.985 | 0.919 |
| 500 | 0.788 | 0.802 | 0.988 [0.833, 1.138] | 0.902 | 0.986 | 0.981 |
| 1,000 | 0.787 | 0.802 | 0.996 [0.948, 1.054] | 0.903 | 0.987 | 0.996 |

**The existing map closes 7% of the cross-site gap.** It moves the slope from 0.787 to 0.802 against a target of 1. That is what "does not transfer" had meant qualitatively, now with a number on it. Fitting on target-site patients closes 98% of it, reaching 0.996 [0.948, 1.054] at 1,000 patients, and Brier falls from 0.01388 to 0.01358.

**Below roughly 250 target patients, recalibrating is worse than leaving the scores alone.** At 25 patients the slope lands at 0.607, further from 1 than the 0.788 it started from, and the interquartile range spans 0.448; at 50 it overshoots to 1.258. Roughly one hour in sixty carries a positive label, so a few dozen patients supply too few positive hours to pin a two-parameter map. The intuition that a little local data must help is wrong here, and the failure is quiet: a map fitted on 25 patients returns confident, worse-calibrated scores rather than an error.

Within source the picture inverts, which is the control this needs. There the source map is fitted on far more patients than any k tested and reaches 0.987, so target fitting only catches up at around 1,000 patients and never beats it by much. Recalibration is worth doing where it was already known not to transfer, and close to pointless where it already worked.

So the item is answerable and the answer has a price: on the order of a thousand labelled patients from the new site. In deployment that means waiting for a thousand patients' outcomes before the calibration can be trusted, which is a real constraint rather than a free fix. This is also an upper bound: fitting and scoring inside one frozen cohort shares that cohort's idiosyncrasies, and it presumes labelled target outcomes already exist. Local outputs: `artifacts/transfer-full-site-42` and `artifacts/transfer-full-random-42`.

## Subgroup contrasts replace the uncontrolled descriptions

Previous releases described each subgroup level on its own with no intervals and no multiplicity control, and said so. That is the combination that produced the disparity findings the full-cohort release had to retract. The new `contrasts` command states the comparisons instead: a descriptor family fixed in code before any held-out hour is read, one level-versus-rest AUROC difference per level with a whole-patient bootstrap interval, and Holm-Bonferroni across the family. A descriptor with exactly two levels contributes one contrast rather than two mirrored ones, which would otherwise inflate the family and make the correction needlessly conservative.

**No contrast survives correction on any run tested**, across nine contrasts per random-split run and eight per site-held-out run. The site descriptor contributes nothing to the latter because that test cohort is a single site.

| Run | Closest contrast | AUROC vs rest | 95% interval | p | p after Holm |
|---|---|---:|---:|---:|---:|
| Random, persistent | `age_band/age_lt_50` | +0.0309 | [+0.002, +0.061] | 0.038 | 0.342 |
| Random, event target | `unit/unit_unrecorded` | +0.0366 | [+0.004, +0.072] | 0.036 | 0.324 |
| Site, persistent | `unit/unit_unrecorded` | +0.0368 | [+0.011, +0.065] | 0.008 | 0.064 |
| Site, event target | `age_band/age_lt_50` | -0.0387 | [-0.071, -0.005] | 0.030 | 0.240 |

All four of these unadjusted intervals exclude zero, and the correction is precisely what stops them being read as findings. `age_band/age_lt_50` illustrates why: +0.0309 on the random split, -0.0250 [-0.056, +0.009] on the site split, and -0.0387 [-0.071, -0.005] on the site split under the event target. The same recorded level, opposite directions, twice with an interval that excludes zero. That is what resampling noise looks like when it is scanned across a family. This corroborates the full-cohort retraction recorded below, where subgroup disparities reported from a 2,000-patient subsample did not survive the full cohort.

Because it reads a frozen run rather than running inside training, this applies to runs trained before it existed, including the previously reported full-cohort runs. It remains a post-hoc description of one cohort and is not subgroup validation.

## Verification for these additions

All 63 tests pass, up from 28 before this branch. New coverage: the horizon-6 label identity and that it holds only at that horizon, pre-onset masking and exclusion of unrecoverable transitions, the proof that masking features after the fact equals truncating the record first (without which post-onset physiology could leak backwards into fitted hours), Holm against its step-down definition, planted and null subgroup differences, binary-descriptor deduplication, burden matching that refuses to extrapolate past a measured budget, recalibration that never scores a patient with a map fitted on itself, and every new command end to end.

The burden comparison was first produced by a throwaway script, which left the central table of this release unreproducible from the repository. It is now the `sweep` command, and the command reproduces the original numbers exactly.

Re-running the persistent target under the new code reproduces both frozen full-cohort runs exactly, across 309,558 and 761,995 predictions, with identical AUROC, average precision, threshold and utility. Every module hash recorded in each new run matches the shipped code, so these numbers are reproducible from a clean checkout.

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
