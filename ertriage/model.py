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

from .data import read_patient, features, patient_attributes
from .evaluate import (bootstrap, calibration, normalized_utility, paired_difference, patient_groups,
                       recalibrate, recalibrator, subgroup_metrics, utility_by_patient, utility_hours,
                       utility_of)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


def split_patients(manifest, seed, scheme="random"):
    """Patient-disjoint splits. "site" holds out an entire source site instead of random patients."""
    if scheme == "site":
        site = manifest.patient.str.split("/").str[0]
        if site.nunique() < 2:
            raise ValueError("Site-held-out split needs patients from at least two sites")
        holdout = sorted(site.unique())[-1]
        develop, test = manifest[site != holdout], manifest[site == holdout]
        train, val = train_test_split(develop, test_size=.25, random_state=seed, stratify=develop.ever_sepsis)
        return {"train": train, "validation": val, "test": test}
    if scheme != "random":
        raise ValueError(f"Unknown split scheme: {scheme}")
    train, rest = train_test_split(manifest, test_size=.4, random_state=seed, stratify=manifest.ever_sepsis)
    val, test = train_test_split(rest, test_size=.5, random_state=seed, stratify=rest.ever_sepsis)
    return {"train": train, "validation": val, "test": test}


SELECTION_ORDER = ("logistic", "boosting")  # simplest first; prevalence stays a reference baseline


def threshold_for(y, p, ids=None, rule="f1", budget=2., grid=201):
    """Validation-only threshold rules, all frozen before any test hour is scored.

    "f1" maximizes hourly F1 and weights long stays most; it chases a noisy
    precision-recall curve when barely one hour in seventy is positive. "utility"
    maximizes the official patient-weighted normalized utility, the score this
    project actually reports. "budget" takes the lowest threshold whose
    validation alert load stays within `budget` alert hours per 100, fixing the
    operating point by review capacity rather than by a curve maximum. None of
    these is a clinically justified operating point.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Threshold selection needs aligned nonempty labels and finite scores in [0, 1]")
    if rule == "f1":
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[np.argmax(f1)])
    candidates = np.unique(p)
    if rule == "budget":
        if not np.isfinite(budget) or not 0 < budget <= 100:
            raise ValueError("Alert budget must be within (0, 100]")
        rate = (len(p) - np.searchsorted(np.sort(p), candidates, side="left")) / len(p)
        allowed = np.flatnonzero(rate <= budget / 100)
        # No observed threshold can satisfy a budget smaller than the top tie.
        # A finite sentinel above the score domain represents the no-alert rule.
        return float(candidates[allowed[0]]) if len(allowed) else float(np.nextafter(1., np.inf))
    if rule != "utility":
        raise ValueError(f"Unknown threshold rule: {rule}")
    if ids is None:
        raise ValueError("The utility threshold rule needs patient identifiers")
    alert, silent, best = utility_hours(y, patient_groups(ids))
    candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, grid)))
    scored = [(utility_of(p >= t, alert, silent, best), float(t)) for t in candidates]
    scored = [pair for pair in scored if pair[0] is not None]
    if not scored:
        raise ValueError("No threshold earns defined utility on validation")
    # Equal utility breaks towards the higher threshold, the quieter operating point.
    return max(scored)[1]


def select_model(scores, y, ids, order=SELECTION_ORDER, seed=42, draws=1000):
    """Pick the average-precision leader, then step back to the simplest model it does not beat.

    The seed sweep in RESULTS.md showed leaders separated by validation margins
    as small as 0.0002, so the leader alone is close to arbitrary. A paired
    patient bootstrap on validation predictions decides whether the margin is
    distinguishable from resampling noise; when it is not, the earlier model in
    the prespecified order is reported. This uses validation only and is a
    tie-breaking convention, not evidence that the simpler model generalizes.
    """
    leader = max(order, key=lambda name: average_precision_score(y, scores[name]))
    margins = {}
    for name in order:
        if name == leader:
            break
        margins[name] = paired_difference(y, scores[leader], scores[name], ids, seed=seed, draws=draws)
        if margins[name] and margins[name]["low"] <= 0 <= margins[name]["high"]:
            return name, dict(rule="stable", leader=leader, chosen=name, stepped_back_to=name,
                              average_precision_margins=margins)
    # Declining to step back is a finding too, so the margins that justified it are always recorded.
    return leader, dict(rule="stable", leader=leader, chosen=leader, stepped_back_to=None,
                        average_precision_margins=margins)


def metrics(y, p, threshold, ids):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
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
                nonsepsis_patients_with_any_alert=float(negative.alert.mean()) if len(negative) else None,
                normalized_utility=normalized_utility(*utility_by_patient(y, pred, patient_groups(ids))),
                calibration=calibration(y, p))


def train(root, output, limit=2000, seed=42, scheme="random", draws=1000,
          rule="budget", budget=2., selection="stable"):
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
        records.append(dict(patient=pid, ever_sepsis=int(df.SepsisLabel.max()), hours=len(df), sha256=digest,
                            site=pid.split("/")[0], **patient_attributes(df)))
    manifest = pd.DataFrame(records)
    if manifest.ever_sepsis.value_counts().min() < 10 or manifest.ever_sepsis.nunique() != 2:
        raise ValueError("Need at least 10 patients of each class; increase --limit")
    splits = split_patients(manifest, seed, scheme)
    output.mkdir(parents=True)
    arrays = {}
    for name, subset in splits.items():
        if subset.ever_sepsis.nunique() != 2:
            raise ValueError(f"Split '{name}' lacks both classes; increase --limit or use --split random")
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
    descriptors = ["site", "age_band", "gender", "unit"]
    levels = {name: splits["test"].set_index("patient")[name].to_dict() for name in descriptors}
    report = dict(seed=seed, split=scheme, threshold_rule=rule, selection_rule=selection,
                  selected_patients=len(paths), available_patients=len(list(root.glob("site_*/*.psv"))))
    if rule == "budget":
        report["alert_budget_per_100"] = budget
    report.update(python=platform.python_version(), sklearn=sklearn.__version__, models={},
                  scope="ICU retrospective development only; not ER validation",
                  subgroups_note="Descriptive splits of one held-out cohort by recorded administrative"
                                 " fields; no intervals, no multiplicity control, not subgroup validation.",
                  recalibration_note="Platt map fitted on validation predictions only and applied unchanged"
                                     " to held-out hours; monotone, so alerts and ranking are unchanged.")
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
            threshold = threshold_for(vy, vp, vi, rule=rule, budget=budget)
            report["models"][name] = {"validation": metrics(vy, vp, threshold, vi)}
            # Recalibration is fitted on validation predictions only and frozen before test hours.
            fit = recalibrator(vy, vp)
            if fit:
                fit = dict(fit, mapped_threshold=float(recalibrate(fit, threshold)))
            report["models"][name]["recalibration"] = fit
            joblib.dump(dict(model=model, threshold=threshold, feature_columns=list(x.columns),
                             recalibration=fit), output / f"{name}.joblib")
        if selection == "stable":
            vx, vy, vi = arrays["validation"]
            scores = {name: model.predict_proba(vx)[:, 1] for name, model in models.items()}
            selected, decision = select_model(scores, vy, vi, seed=seed, draws=draws)
        else:
            selected = max(models, key=lambda n: report["models"][n]["validation"]["average_precision"])
            decision = dict(rule="ap", leader=selected, chosen=selected, stepped_back_to=None)
        report["selected_model"] = selected
        report["selection"] = decision
        # Selection and thresholds are frozen before examining test outcomes.
        tx, ty, ti = arrays["test"]
        for name, model in models.items():
            p = model.predict_proba(tx)[:, 1]
            threshold = report["models"][name]["validation"]["threshold"]
            report["models"][name]["test"] = metrics(ty, p, threshold, ti)
            report["models"][name]["test"]["bootstrap"] = bootstrap(ty, p, threshold, ti, seed=seed, draws=draws)
            report["models"][name]["test"]["subgroups"] = {
                name_: subgroup_metrics(ty, p, threshold, ti, level) for name_, level in levels.items()}
            fit = report["models"][name]["recalibration"]
            columns = dict(patient=ti, hour=np.concatenate([frames[pid].ICULOS.to_numpy()
                                                          for pid in splits["test"].patient]),
                           label=ty, score=p)
            if fit:
                calibrated = recalibrate(fit, p)
                columns["calibrated_score"] = calibrated
                report["models"][name]["test_recalibrated"] = dict(
                    threshold=fit["mapped_threshold"], brier=float(brier_score_loss(ty, calibrated)),
                    alerts_identical=bool(np.array_equal(calibrated >= fit["mapped_threshold"], p >= threshold)),
                    calibration=calibration(ty, calibrated))
            if name == selected:
                pd.DataFrame(columns).to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
