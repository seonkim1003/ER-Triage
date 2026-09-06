"""Patient-level warning timing against a label-derived onset proxy, never a fitted target."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .history import HeldoutRun

SOURCE = "https://physionet.org/content/challenge-2019/1.0.0/"
METHOD = ("Onset proxy = first observed 0-to-1 label transition hour + 6. Primary timing denominator "
          "requires a persistent label, that onset proxy within the recorded stay, and the complete "
          "[onset-12, onset-6] window. An early warning is any threshold-positive hour in that inclusive "
          "window, including an ongoing alert episode. Before-onset detection uses [onset-12, onset). "
          "Episodes are contiguous threshold-positive hours separated by at least one negative hour. "
          "There is no clinical cooldown, policy selection, model fitting, or threshold tuning.")


def warning_patient(hours, labels, scores, threshold):
    hours, labels, scores = np.asarray(hours), np.asarray(labels), np.asarray(scores, dtype=float)
    if (hours.ndim != 1 or not len(hours) or labels.shape != hours.shape or scores.shape != hours.shape
            or not np.isfinite(hours).all() or not np.all(np.diff(hours) == 1)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or not np.all((scores >= 0) & (scores <= 1)) or not np.isfinite(threshold)):
        raise ValueError("Warning evaluation needs aligned consecutive hours, binary labels and finite scores")
    alerts = scores >= threshold
    starts = np.flatnonzero(alerts & ~np.r_[False, alerts[:-1]])
    positive = np.flatnonzero(labels)
    first_alert = float(hours[np.flatnonzero(alerts)[0]]) if alerts.any() else None
    result = dict(hours=len(hours), positive_patient=bool(len(positive)),
                  alert_hours=int(alerts.sum()), alert_episodes=len(starts),
                  repeated_episodes=max(0, len(starts) - 1), any_alert=bool(alerts.any()),
                  first_alert_hour=first_alert, first_positive_label_hour=None, onset_proxy_hour=None,
                  timing_status="no_positive_label", timing_eligible=False, early_warning=False,
                  detected_before_onset=False, first_alert_lead_hours=None, early_warning_lead_hours=None)
    if not len(positive):
        return result
    first = int(positive[0])
    result["first_positive_label_hour"] = float(hours[first])
    if (np.diff(labels.astype(int)) < 0).any():
        result["timing_status"] = "nonpersistent_label"
        return result
    if first == 0:
        result["timing_status"] = "positive_at_record_start"
        return result  # the actual transition and onset cannot be recovered
    onset = float(hours[first] + 6)
    result["onset_proxy_hour"] = onset
    result["first_alert_lead_hours"] = onset - first_alert if first_alert is not None else None
    if onset > hours[-1]:
        result["timing_status"] = "onset_beyond_record"
        return result
    if onset - 12 < hours[0]:
        result["timing_status"] = "incomplete_warning_window"
        return result
    early = alerts & (hours >= onset - 12) & (hours <= onset - 6)
    before = alerts & (hours >= onset - 12) & (hours < onset)
    result.update(timing_status="eligible", timing_eligible=True, early_warning=bool(early.any()),
                  detected_before_onset=bool(before.any()),
                  early_warning_lead_hours=(onset - float(hours[np.flatnonzero(early)[0]])) if early.any() else None)
    return result


def summarize_warning(patients, seed=42, draws=1000):
    if patients.empty or draws < 1:
        raise ValueError("Early-warning summary needs patients and positive bootstrap draws")
    eligible = patients.timing_eligible.to_numpy(dtype=bool)
    healthy = ~patients.positive_patient.to_numpy(dtype=bool)
    matrix = np.column_stack([eligible, eligible & patients.early_warning,
                              eligible & patients.detected_before_onset, healthy,
                              healthy & patients.any_alert, patients.hours,
                              patients.alert_episodes, patients.repeated_episodes]).astype(float)

    def measures(total):
        e, early, before, neg, fp, hours, episodes, repeated = total
        return dict(early_warning_fraction=early / e if e else None,
                    before_onset_detection_fraction=before / e if e else None,
                    nonsepsis_patients_with_any_alert=fp / neg if neg else None,
                    alert_episodes_per_100_hours=100 * episodes / hours,
                    repeated_episodes_per_100_hours=100 * repeated / hours)

    observed = measures(matrix.sum(axis=0))
    samples = {key: [] for key in observed}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        pick = rng.integers(0, len(matrix), len(matrix))
        for key, value in measures(matrix[pick].sum(axis=0)).items():
            if value is not None:
                samples[key].append(float(value))
    intervals = {key: (dict(low=float(np.percentile(v, 2.5)), high=float(np.percentile(v, 97.5)),
                           valid_draws=len(v)) if len(v) >= max(1, draws / 2) else None)
                 for key, v in samples.items()}
    lead = patients.loc[patients.timing_eligible & patients.early_warning, "early_warning_lead_hours"]
    return dict(patients=len(patients), positive_patients=int(patients.positive_patient.sum()),
                timing_eligible_patients=int(eligible.sum()), early_warned_patients=int(matrix[:, 1].sum()),
                missed_early_window_patients=int(eligible.sum() - matrix[:, 1].sum()),
                missed_before_onset_patients=int(eligible.sum() - matrix[:, 2].sum()),
                nonsepsis_patients=int(healthy.sum()), nonsepsis_patients_alerted=int(matrix[:, 4].sum()),
                timing_status_counts={str(k): int(v) for k, v in patients.timing_status.value_counts().items()},
                measures=observed, bootstrap=dict(seed=seed, draws=draws, unit="patient", level=.95, intervals=intervals),
                early_warning_lead_hours_among_detected=(dict(count=len(lead), median=float(lead.median()),
                    q25=float(lead.quantile(.25)), q75=float(lead.quantile(.75))) if len(lead) else None))


def early_warning(root, run, output, seed=42, draws=1000):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory to preserve prior early-warning reports")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    cohort = HeldoutRun(root, run)
    rows = []
    for i, pid in enumerate(cohort.patients, start=1):
        frame, saved = cohort.patient(pid)
        rows.append(dict(patient=pid, **warning_patient(frame.ICULOS, frame.SepsisLabel, saved.score, cohort.threshold)))
        if i % 2000 == 0:
            print(f"Evaluated timing for {i}/{len(cohort.patients)} patients", flush=True)
    patients = pd.DataFrame(rows)
    report = summarize_warning(patients, seed, draws)
    report.update(**cohort.provenance(), method=METHOD, label_definition_source=SOURCE,
                  scope="Retrospective ICU development only. Onset is label-derived, not independently observed. "
                        "The timing window is an analysis convention, not a clinical requirement.",
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))})
    output.mkdir(parents=True, exist_ok=False)
    patients.to_csv(output / "patient_warnings.csv", index=False)
    (output / "early_warning.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Early-warning evaluation", "", report["scope"], "", METHOD, "",
             f"Label definition: [PhysioNet 2019]({SOURCE}).", "",
             f"{report['timing_eligible_patients']:,} timing-eligible patients out of "
             f"{report['positive_patients']:,} patients with a positive label.", "",
             "| Measure | Estimate | 95% patient-bootstrap interval |", "|---|---:|---:|"]
    for key, value in report["measures"].items():
        ci = report["bootstrap"]["intervals"][key]
        estimate = f"{value:.4f}" if value is not None else "Undefined"
        bounds = f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci else "Undefined"
        lines.append(f"| {key} | {estimate} | {bounds} |")
    lines.extend(["", f"Timing status counts: {report['timing_status_counts']}", "",
                  "Missed means no alert in the specified window, not a clinical missed diagnosis. "
                  "Lead times among detected patients exclude missed cases and are conditional summaries.", ""])
    (output / "EARLY_WARNING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return report
