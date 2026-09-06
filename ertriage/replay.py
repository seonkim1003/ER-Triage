import json
from pathlib import Path

import joblib
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
    rows, previous, due = [], None, 1
    with threadpool_limits(limits=2):
        for i in range(len(df)):
            prefix = df.iloc[:i + 1]
            x = features(prefix).iloc[[-1]][bundle["feature_columns"]]
            score = float(bundle["model"].predict_proba(x)[0, 1])
            stale = bool((x[[v + "_age" for v in VITALS]] >= 4).any(axis=None))
            interval, reason = policy(score, previous, bundle["threshold"], stale)
            hour = i + 1
            review = hour >= due or interval == 1
            if review:
                due = hour + interval
            else:
                due = min(due, hour + interval)
            row = dict(hour=int(df.ICULOS.iloc[i]), score=score)
            # The frozen map is monotone, so the cadence below is unchanged by recalibration.
            if bundle.get("recalibration"):
                row["calibrated_score"] = float(recalibrate(bundle["recalibration"], score))
            row.update(review_now=review, next_review_in_hours=due-hour, reason=reason,
                       retrospective_label=int(df.SepsisLabel.iloc[i]))
            rows.append(row)
            previous = score
    result = pd.DataFrame(rows)
    dest = run / f"replay_{Path(patient).stem}.csv"
    result.to_csv(dest, index=False)
    print("RESEARCH REPLAY | ICU history, not ER validation or clinical guidance")
    print(f"Patient: {patient}; model: {report['selected_model']}")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved {dest}")
    return result
