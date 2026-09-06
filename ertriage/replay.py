import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import read_patient, features, VITALS
from .evaluate import recalibrate


def policy(score, previous, threshold, stale):
    """Illustrative review cadence; no treatment or measurement recommendations."""
    if stale:
        return 1, "missing/stale observations"
    if score >= threshold:
        return 1, "score crosses validation threshold"
    if previous is not None and score - previous >= .05:
        return 1, "score increased by >=0.05"
    if score >= threshold / 2:
        return 2, "intermediate score"
    return 4, "lower score"


def review_schedule(scores, stale, threshold):
    """Shared causal scheduler for one patient's replay and workload audit."""
    scores, stale = np.asarray(scores, dtype=float), np.asarray(stale, dtype=bool)
    if scores.ndim != 1 or stale.shape != scores.shape or not len(scores):
        raise ValueError("Schedule needs nonempty aligned scores and observation flags")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or not np.isfinite(threshold):
        raise ValueError("Schedule needs finite scores in [0, 1] and a finite threshold")
    rows, previous, due = [], None, 1
    for hour, (score, missing) in enumerate(zip(scores, stale), start=1):
        interval, reason = policy(score, previous, threshold, missing)
        review = hour >= due or interval == 1
        due = hour + interval if review else min(due, hour + interval)
        rows.append(dict(review_now=bool(review), next_review_in_hours=due-hour, reason=reason))
        previous = score
    return pd.DataFrame(rows)


def replay(root, run, patient=None):
    root, run = Path(root), Path(run)
    report = json.loads((run / "metrics.json").read_text())
    heldout = pd.read_csv(run / "test_patients.csv")
    patient = patient or heldout.patient.iloc[0]
    if patient not in set(heldout.patient):
        raise ValueError("Replay must use a patient in the held-out test manifest")
    # Only load model files generated locally by this project; pickle is executable.
    bundle = joblib.load(run / f"{report['selected_model']}.joblib")
    df = read_patient(root / patient)
    rows, stale_flags = [], []
    with threadpool_limits(limits=2):
        for i in range(len(df)):
            prefix = df.iloc[:i + 1]
            x = features(prefix).iloc[[-1]][bundle["feature_columns"]]
            score = float(bundle["model"].predict_proba(x)[0, 1])
            stale = bool((x[[v + "_age" for v in VITALS]] >= 4).any(axis=None))
            stale_flags.append(stale)
            row = dict(hour=int(df.ICULOS.iloc[i]), score=score)
            # The frozen map is monotone, so the cadence below is unchanged by recalibration.
            if bundle.get("recalibration"):
                row["calibrated_score"] = float(recalibrate(bundle["recalibration"], score))
            row.update(retrospective_label=int(df.SepsisLabel.iloc[i]))
            rows.append(row)
    result = pd.DataFrame(rows)
    schedule = review_schedule(result.score, stale_flags, bundle["threshold"])
    result = pd.concat([result.drop(columns="retrospective_label"), schedule,
                        result[["retrospective_label"]]], axis=1)
    dest = run / f"replay_{Path(patient).stem}.csv"
    result.to_csv(dest, index=False)
    print("RESEARCH REPLAY | ICU history, not ER validation or clinical guidance")
    print(f"Patient: {patient}; model: {report['selected_model']}")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved {dest}")
    return result
