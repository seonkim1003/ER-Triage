"""Held-out evaluation add-ons: official utility scoring, calibration and patient bootstrap.

Nothing here fits a model or selects a threshold; these functions only describe
frozen predictions. Development metrics only, not clinical validation.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# PhysioNet/CinC 2019 utility parameters (Reyna et al.), in hours relative to sepsis onset.
DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP = 1.0, -2.0, -0.05


def patient_groups(ids):
    """Slices for each run of equal identifiers; hours arrive grouped by patient and in time order."""
    ids = np.asarray(ids)
    edges = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def utility_terms(labels):
    """Per-hour utility of alerting and of staying silent, plus the best attainable alert vector."""
    n = len(labels)
    alert, silent = np.zeros(n), np.zeros(n)
    best = np.zeros(n, dtype=bool)
    if not labels.any():
        alert[:] = U_FP
        return alert, silent, best
    # SepsisLabel is already shifted DT_OPTIMAL hours before onset, so undo that shift.
    delta = np.arange(n, dtype=float) - (float(np.argmax(labels)) - DT_OPTIMAL)
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    early = (delta > DT_EARLY) & (delta <= DT_OPTIMAL)
    late = (delta > DT_OPTIMAL) & (delta <= DT_LATE)
    alert[early] = m1 * (delta[early] - DT_EARLY)
    alert[late] = m2 * (delta[late] - DT_LATE)
    silent[late] = m3 * (delta[late] - DT_OPTIMAL)
    return alert, silent, (delta >= DT_EARLY) & (delta <= DT_LATE)


def utility_by_patient(y, alerts, groups):
    """Observed, best attainable and inaction utility per patient, so bootstraps resample patients."""
    observed, best, inaction = np.zeros(len(groups)), np.zeros(len(groups)), np.zeros(len(groups))
    for i, part in enumerate(groups):
        alert, silent, optimal = utility_terms(y[part])
        observed[i] = np.where(alerts[part], alert, silent).sum()
        best[i] = np.where(optimal, alert, silent).sum()
        inaction[i] = silent.sum()
    return observed, best, inaction


def normalized_utility(observed, best, inaction):
    """Official normalization; undefined when no selected patient can earn utility."""
    span = best.sum() - inaction.sum()
    return float((observed.sum() - inaction.sum()) / span) if span > 0 else None


def calibration(y, p, bins=10):
    """Equal-count reliability table, expected calibration error and a logit recalibration fit."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    parts = [g for g in np.array_split(np.argsort(p, kind="stable"), min(bins, len(p))) if len(g)]
    table = [dict(count=int(len(g)), mean_score=float(p[g].mean()), observed_rate=float(y[g].mean()))
             for g in parts]
    error = float(sum(len(g) / len(p) * abs(p[g].mean() - y[g].mean()) for g in parts))
    intercept = slope = None
    if len(np.unique(y)) == 2 and len(np.unique(p)) > 1:
        clipped = np.clip(p, 1e-9, 1 - 1e-9)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        fit = LogisticRegression(penalty=None, max_iter=1000).fit(logit, y)
        intercept, slope = float(fit.intercept_[0]), float(fit.coef_[0, 0])
    # A perfectly calibrated score has intercept 0 and slope 1; slope < 1 means overextended scores.
    return dict(expected_calibration_error=error, intercept=intercept, slope=slope, bins=table)


def bootstrap(y, p, threshold, ids, seed=42, draws=1000, level=.95):
    """Percentile intervals from resampling whole patients with replacement.

    Intervals describe sampling variation in this one cohort only. They are not
    evidence of generalization to another hospital, to an ER, or to future care.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    groups = patient_groups(ids)
    alerts = p >= threshold
    observed, best, inaction = utility_by_patient(y, alerts, groups)
    rows = [np.arange(part.start, part.stop) for part in groups]
    rng = np.random.default_rng(seed)
    collected = {key: [] for key in
                 ("auroc", "average_precision", "precision", "recall", "alert_hours_per_100", "normalized_utility")}
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([rows[i] for i in pick])
        by, bp, ba = y[index], p[index], alerts[index]
        if len(np.unique(by)) == 2:
            collected["auroc"].append(roc_auc_score(by, bp))
            collected["average_precision"].append(average_precision_score(by, bp))
        tp = float(np.sum(ba & (by == 1)))
        if ba.any():
            collected["precision"].append(tp / float(ba.sum()))
        if np.any(by == 1):
            collected["recall"].append(tp / float(np.sum(by == 1)))
        collected["alert_hours_per_100"].append(100 * float(ba.mean()))
        value = normalized_utility(observed[pick], best[pick], inaction[pick])
        if value is not None:
            collected["normalized_utility"].append(value)
    half = (1 - level) / 2
    return dict(draws=draws, level=level, seed=seed, intervals={
        key: (None if len(values) < draws // 2 else
              dict(low=float(np.percentile(values, 100 * half)),
                   high=float(np.percentile(values, 100 * (1 - half))), draws=len(values)))
        for key, values in collected.items()})
