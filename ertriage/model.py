import hashlib
import json
import platform
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import read_patient, features

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


def split_patients(manifest, seed):
    train, rest = train_test_split(manifest, test_size=.4, random_state=seed, stratify=manifest.ever_sepsis)
    val, test = train_test_split(rest, test_size=.5, random_state=seed, stratify=rest.ever_sepsis)
    return {"train": train, "validation": val, "test": test}


def threshold_for(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def metrics(y, p, threshold, ids):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    patients = pd.DataFrame({"id": ids, "label": y, "alert": pred}).groupby("id").max()
    negative = patients[patients.label == 0]
    return dict(hours=len(y), patients=len(patients), positive_hour_fraction=float(np.mean(y)),
                auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if np.any(y) else None,
                brier=float(brier_score_loss(y, p)), threshold=threshold,
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
                alert_hours_per_100=float(100 * pred.mean()),
                nonsepsis_patients_with_any_alert=float(negative.alert.mean()) if len(negative) else None)


def train(root, output, limit=2000, seed=42):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    paths = sorted(root.glob("site_*/*.psv"))
    if not paths:
        raise ValueError("No data found. Run download first.")
    if limit and limit < len(paths):
        paths = sorted(np.random.default_rng(seed).choice(paths, size=limit, replace=False))
    records, frames, hashes = [], {}, set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate patient contents: {path}")
        hashes.add(digest)
        df = read_patient(path)
        pid = path.relative_to(root).as_posix()
        frames[pid] = df
        records.append(dict(patient=pid, ever_sepsis=int(df.SepsisLabel.max()), hours=len(df), sha256=digest))
    manifest = pd.DataFrame(records)
    if manifest.ever_sepsis.value_counts().min() < 10 or manifest.ever_sepsis.nunique() != 2:
        raise ValueError("Need at least 10 patients of each class; increase --limit")
    splits = split_patients(manifest, seed)
    output.mkdir(parents=True)
    arrays = {}
    for name, subset in splits.items():
        subset.sort_values("patient").to_csv(output / f"{name}_patients.csv", index=False)
        dfs = [frames[p] for p in subset.patient]
        arrays[name] = (pd.concat([features(df) for df in dfs], ignore_index=True),
                        np.concatenate([df.SepsisLabel.to_numpy(dtype=int) for df in dfs]),
                        np.repeat(subset.patient.to_numpy(), [len(df) for df in dfs]))
    x, y, _ = arrays["train"]
    models = {
        "prevalence": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=seed),
    }
    report = dict(seed=seed, selected_patients=len(paths), available_patients=len(list(root.glob("site_*/*.psv"))))
    report.update(python=platform.python_version(), sklearn=sklearn.__version__, models={}, scope="ICU retrospective development only; not ER validation")
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))}
    provenance = root / "provenance.json"
    if provenance.exists():
        report["data_provenance_sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    with threadpool_limits(limits=2):
        for name, model in models.items():
            print(f"Fitting {name}: {len(y)} training hours", flush=True)
            model.fit(x, y)
            vx, vy, vi = arrays["validation"]
            vp = model.predict_proba(vx)[:, 1]
            threshold = threshold_for(vy, vp)
            report["models"][name] = {"validation": metrics(vy, vp, threshold, vi)}
            joblib.dump(dict(model=model, threshold=threshold, feature_columns=list(x.columns)), output / f"{name}.joblib")
        selected = max(models, key=lambda n: report["models"][n]["validation"]["average_precision"])
        report["selected_model"] = selected
        # Selection and thresholds are frozen before examining test outcomes.
        tx, ty, ti = arrays["test"]
        for name, model in models.items():
            p = model.predict_proba(tx)[:, 1]
            threshold = report["models"][name]["validation"]["threshold"]
            report["models"][name]["test"] = metrics(ty, p, threshold, ti)
            if name == selected:
                pd.DataFrame(dict(patient=ti, label=ty, score=p)).to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
