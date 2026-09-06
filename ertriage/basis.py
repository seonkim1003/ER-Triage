"""Re-score a frozen run on the event-anchored basis, without refitting anything.

This answers one question about results already reported: how much of the
discrimination credited to a model came from hours at or after the onset proxy,
when the patient had already declared themselves?

Nothing is fitted, loaded from a pickle, or re-thresholded. The frozen scores in
`test_predictions.csv` are read back and summarized twice, once over every
recorded hour against the persistent label, and once over pre-onset hours only
against the event-anchored label.

A useful identity makes the comparison sharp. The persistent label turns 1 at
`first`, the onset proxy is `first + 6`, and the pre-onset mask keeps `t < first
+ 6`. So at horizon 6 the event label restricted to pre-onset hours *is* the
persistent label restricted to pre-onset hours, exactly. At that horizon the two
summaries differ in one respect and one only: whether hours at or after onset are
counted. Other horizons genuinely change the question being asked.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluate import patient_groups
from .history import HeldoutRun
from .target import HORIZON, METHOD, UNUSABLE, event_target

NOTE = ("Descriptive re-summary of frozen held-out scores. No model was refitted, no threshold was "
        "re-tuned, and the operating point is the one already frozen on validation against the "
        "persistent label. Comparing bases compares two summaries of one score vector, not two "
        "models. The official PhysioNet utility is defined against the persistent label and its "
        "timing and is deliberately not reported on the event basis.")


def _summary(y, p, threshold):
    """Discrimination and frozen-threshold behavior for one basis."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    alerts = p >= threshold
    hits = float(np.sum(alerts & (y == 1)))
    return dict(hours=int(len(y)), positive_hours=int(y.sum()),
                positive_hour_fraction=float(y.mean()) if len(y) else None,
                auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if y.any() else None,
                alert_hours_per_100=float(100 * alerts.mean()) if len(y) else None,
                precision=float(hits / alerts.sum()) if alerts.any() else None,
                recall=float(hits / y.sum()) if y.any() else None)


def _auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else None


def basis_arrays(cohort, horizon=HORIZON):
    """Per-patient scores, both label definitions, and the pre-onset mask."""
    scores, persistent, event, keep, ids, rows, counts = [], [], [], [], [], [], {}
    for pid in cohort.patients:
        frame, saved = cohort.patient(pid)
        y, mask, status = event_target(frame.SepsisLabel.to_numpy(dtype=int), horizon)
        counts[status] = counts.get(status, 0) + 1
        rows.append(dict(patient=pid, hours=len(frame), status=status,
                         usable=status not in UNUSABLE,
                         preonset_hours=int(mask.sum()) if mask is not None else 0,
                         dropped_hours=int((~mask).sum()) if mask is not None else len(frame),
                         event_positive_hours=int(y.sum()) if y is not None else 0))
        if status in UNUSABLE:
            continue
        scores.append(saved.score.to_numpy(dtype=float))
        persistent.append(frame.SepsisLabel.to_numpy(dtype=int))
        event.append(y)
        keep.append(mask)
        ids.append(np.repeat(pid, len(frame)))
    if not scores:
        raise ValueError("No held-out patient has a recoverable onset proxy")
    return (np.concatenate(scores), np.concatenate(persistent), np.concatenate(event),
            np.concatenate(keep), np.concatenate(ids), pd.DataFrame(rows), counts)


def basis_report(cohort, horizon=HORIZON, seed=42, draws=1000, level=.95):
    p, persistent, event, keep, ids, per_patient, counts = basis_arrays(cohort, horizon)
    threshold = cohort.threshold
    full = _summary(persistent, p, threshold)
    pre = _summary(event[keep], p[keep], threshold)
    # Resample whole patients and recompute both summaries on the same draw.
    groups = patient_groups(ids)
    parts = [np.arange(part.start, part.stop) for part in groups]
    rng = np.random.default_rng(seed)
    differences, full_draws, pre_draws = [], [], []
    for _ in range(draws):
        index = np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))])
        a, b = _auroc(persistent[index], p[index]), None
        sub = index[keep[index]]
        if a is not None and len(sub):
            b = _auroc(event[sub], p[sub])
        if a is not None and b is not None:
            full_draws.append(a)
            pre_draws.append(b)
            differences.append(a - b)
    if len(differences) < draws // 2:
        raise ValueError("Too few usable bootstrap draws to report an interval")
    half = (1 - level) / 2

    def interval(values):
        return [float(np.percentile(values, 100 * half)), float(np.percentile(values, 100 * (1 - half)))]

    identity = None
    if horizon == 6:
        identity = bool(np.array_equal(event[keep], persistent[keep]))
    return dict(
        run=str(cohort.run), horizon=horizon, threshold=threshold, selected_model=cohort.selected,
        method=METHOD, note=NOTE, seed=seed, draws=len(differences), level=level,
        patients_total=len(cohort.patients), patients_used=int(per_patient.usable.sum()),
        status_counts=counts, hours_all=int(len(p)), hours_preonset=int(keep.sum()),
        hours_dropped_at_or_after_onset=int((~keep).sum()),
        horizon6_label_identity=identity,
        all_hours_persistent_label=full, preonset_hours_event_label=pre,
        auroc_all_hours_interval=interval(full_draws),
        auroc_preonset_interval=interval(pre_draws),
        auroc_difference=dict(observed=(None if full["auroc"] is None or pre["auroc"] is None
                                        else float(full["auroc"] - pre["auroc"])),
                              low=interval(differences)[0], high=interval(differences)[1]),
        provenance=cohort.provenance()), per_patient


def _markdown(report):
    full, pre = report["all_hours_persistent_label"], report["preonset_hours_event_label"]
    d = report["auroc_difference"]

    def show(value, digits=3):
        return "undefined" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Event-anchored re-evaluation", "",
        f"Frozen run `{report['run']}`, model `{report['selected_model']}`, horizon "
        f"{report['horizon']}h. Descriptive re-summary of saved scores; nothing was refitted "
        "or re-thresholded.", "",
        f"Both rows cover the same {report['patients_used']:,} patients with a recoverable onset "
        "proxy, so neither reproduces the run's headline metrics over the whole cohort.", "",
        "| Basis | Hours | Positive hours | AUROC | Average precision | Recall at frozen threshold |",
        "|---|---:|---:|---:|---:|---:|",
        f"| All recorded hours, persistent label | {full['hours']:,} | {full['positive_hours']:,} | "
        f"{show(full['auroc'])} | {show(full['average_precision'])} | {show(full['recall'])} |",
        f"| Pre-onset hours only, event label | {pre['hours']:,} | {pre['positive_hours']:,} | "
        f"{show(pre['auroc'])} | {show(pre['average_precision'])} | {show(pre['recall'])} |", "",
        f"AUROC difference {show(d['observed'])} [{show(d['low'])}, {show(d['high'])}], "
        f"{report['draws']} whole-patient draws, seed {report['seed']}.", "",
        f"{report['hours_dropped_at_or_after_onset']:,} of {report['hours_all']:,} hours are at or "
        f"after the onset proxy and are dropped. {report['patients_used']:,} of "
        f"{report['patients_total']:,} held-out patients have a recoverable onset proxy.", "",
    ]
    if report["horizon6_label_identity"]:
        lines += ["At horizon 6 the event label restricted to pre-onset hours is identical to the "
                  "persistent label restricted to the same hours, verified elementwise here. The "
                  "difference above is therefore attributable to the dropped post-onset hours "
                  "alone, not to a change of label definition.", ""]
    lines += ["Patient statuses: " + ", ".join(f"{k} {v:,}" for k, v in sorted(report["status_counts"].items())),
              "", report["note"], "", "## Method", "", report["method"], ""]
    return "\n".join(lines)


def basis(root, run, out, horizon=HORIZON, seed=42, draws=1000):
    out = Path(out)
    if out.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    cohort = HeldoutRun(root, run)
    report, per_patient = basis_report(cohort, horizon, seed, draws)
    out.mkdir(parents=True)
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))}
    (out / "basis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "BASIS.md").write_text(_markdown(report), encoding="utf-8")
    per_patient.to_csv(out / "patient_basis.csv", index=False)
    print(_markdown(report))
    return report
