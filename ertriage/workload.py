"""Audit frozen review schedules on held-out histories; no fitting or policy selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import observation_ages, read_patient
from .replay import review_schedule


POLICIES = ("adaptive", "fixed_1h", "fixed_2h", "fixed_4h")


def patient_workload(scores, stale, threshold):
    schedule = review_schedule(scores, stale, threshold)
    reviews = schedule.review_now.to_numpy()
    n = len(reviews)
    return dict(hours=n, stale_hours=int(np.sum(stale)), adaptive=int(reviews.sum()),
                stale_reviews=int(np.sum(reviews & np.asarray(stale))),
                fixed_1h=n, fixed_2h=(n + 1) // 2, fixed_4h=(n + 3) // 4,
                all_hours_reviewed=bool(reviews.all()))


def summarize_workload(patients, seed=42, draws=1000):
    """Paired patient bootstrap of workload rates; stays keep all their hours."""
    if patients.empty or draws < 1:
        raise ValueError("Workload intervals need patients and positive bootstrap draws")
    counts = patients[list(POLICIES)].to_numpy(dtype=float)
    hours = patients.hours.to_numpy(dtype=float)
    rates = 100 * counts.sum(axis=0) / hours.sum()
    rng = np.random.default_rng(seed)
    sampled = np.empty((draws, len(POLICIES)))
    for i in range(draws):
        pick = rng.integers(0, len(patients), len(patients))
        sampled[i] = 100 * counts[pick].sum(axis=0) / hours[pick].sum()

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return dict(low=float(low), high=float(high))

    policies = {}
    for i, name in enumerate(POLICIES):
        policies[name] = dict(reviews=int(counts[:, i].sum()), reviews_per_100_hours=float(rates[i]),
                              interval=interval(sampled[:, i]))
        if i:
            policies[name]["adaptive_minus_fixed_per_100_hours"] = dict(
                observed=float(rates[0] - rates[i]), interval=interval(sampled[:, 0] - sampled[:, i]))
    return dict(patients=len(patients), hours=int(hours.sum()), policies=policies,
                stale_hours=int(patients.stale_hours.sum()),
                stale_hour_fraction=float(patients.stale_hours.sum() / hours.sum()),
                adaptive_reviews_for_missing_or_stale_observations=int(patients.stale_reviews.sum()),
                patients_reviewed_every_hour=int(patients.all_hours_reviewed.sum()),
                bootstrap=dict(unit="patient", paired=True, draws=draws, seed=seed, level=.95))


def workload(root, run, output, seed=42, draws=1000):
    root, run, output = Path(root).resolve(), Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous workload audits")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    source_names = ["metrics.json", "test_predictions.csv", "test_patients.csv",
                    "train_patients.csv", "validation_patients.csv"]
    hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in source_names}
    report = json.loads((run / "metrics.json").read_text())
    manifest = pd.read_csv(run / "test_patients.csv")
    required = {"patient", "hours", "sha256"}
    if not required.issubset(manifest.columns) or manifest.empty or manifest.patient.duplicated().any():
        raise ValueError("Invalid or duplicate held-out patient manifest")
    heldout = set(manifest.patient)
    for name in ("train", "validation"):
        if heldout & set(pd.read_csv(run / f"{name}_patients.csv").patient):
            raise ValueError("Held-out patients overlap development patients")
    predictions = pd.read_csv(run / "test_predictions.csv")
    if not {"patient", "label", "score"}.issubset(predictions.columns):
        raise ValueError("Predictions need patient, label and score columns")
    if predictions.patient.isna().any() or set(predictions.patient) != heldout:
        raise ValueError("Predictions must contain exactly the held-out patients")
    selected = report["selected_model"]
    threshold = report["models"][selected]["validation"]["threshold"]
    grouped = predictions.groupby("patient", sort=False)
    rows = []
    for i, record in enumerate(manifest.sort_values("patient").itertuples(index=False), start=1):
        path = (root / record.patient).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Patient path must stay inside the data directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"Patient file differs from the training manifest: {record.patient}")
        frame = read_patient(path)
        saved = grouped.get_group(record.patient)
        if len(saved) != record.hours or len(frame) != record.hours:
            raise ValueError(f"Prediction hour count differs from manifest: {record.patient}")
        if not np.array_equal(saved.label.to_numpy(), frame.SepsisLabel.to_numpy()):
            raise ValueError(f"Prediction label order differs from source: {record.patient}")
        if "hour" in saved and not np.array_equal(saved.hour.to_numpy(), frame.ICULOS.to_numpy()):
            raise ValueError(f"Prediction hour order differs from source: {record.patient}")
        stale = (observation_ages(frame) >= 4).any(axis=1).to_numpy()
        rows.append(dict(patient=record.patient, **patient_workload(saved.score.to_numpy(), stale, threshold)))
        if i % 1000 == 0:
            print(f"Audited {i}/{len(manifest)} held-out patients", flush=True)
    patients = pd.DataFrame(rows)
    result = summarize_workload(patients, seed=seed, draws=draws)
    result.update(selected_model=selected, threshold=threshold, source_run=str(run),
                  source_sha256=hashes,
                  audit_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path(__file__).parent.glob("*.py"))},
                  scope="Retrospective ICU review workload simulation only; not clinical or ER validation.",
                  method="Frozen selected-model scores; original hourly observations still arrive. "
                         "Fixed schedules start on the first recorded hour. Adaptive scheduling uses "
                         "the same causal function as replay. No model, threshold, or policy is fitted or selected.",
                  alignment="Patient hashes, hour counts and ordered labels verified. Older prediction "
                            "files lack explicit hour keys; their within-patient row order is assumed intact.",
                  limitations="Review events are not alert hours, staff time, measurement acquisition, or "
                              "patient outcomes. Bootstrap intervals describe this cohort only. Policy "
                              "intervals and the alert budget remain illustrative.")
    # Validate everything before creating output. Existing model artifacts are read only.
    output.mkdir(parents=True, exist_ok=False)
    patients.to_csv(output / "patient_workload.csv", index=False)
    (output / "workload.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Held-out review workload", "", result["scope"], "", result["method"], "",
             f"Patients: {result['patients']:,}; recorded hours: {result['hours']:,}.", "",
             "| Schedule | Reviews | Reviews / 100 hours (95% patient bootstrap) |",
             "|---|---:|---:|"]
    for name, values in result["policies"].items():
        ci = values["interval"]
        lines.append(f"| {name} | {values['reviews']:,} | {values['reviews_per_100_hours']:.2f} "
                     f"[{ci['low']:.2f}, {ci['high']:.2f}] |")
    lines.extend(["", f"Missing or stale vitals occur in {result['stale_hour_fraction']:.1%} of hours "
                  "and force review on each of those hours.", "", result["limitations"], ""])
    (output / "WORKLOAD.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {output}")
    return result
