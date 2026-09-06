"""Held-out evaluation add-ons: utility scoring, calibration, recalibration, subgroups, bootstrap.

Nothing here selects a model or a threshold. The one thing that is fitted is the
recalibration map, and it is fitted on validation predictions only and then
applied unchanged to held-out hours. Development metrics only, not clinical
validation.
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


def utility_hours(y, groups):
    """Per-hour alert and silence utility plus the best attainable alert mask, across the cohort.

    Utility is additive over hours, so laying these out once makes the score of
    any alert mask a single vectorized sum. Used to search thresholds on
    validation without refitting or rescoring anything.
    """
    alert, silent = np.zeros(len(y)), np.zeros(len(y))
    best = np.zeros(len(y), dtype=bool)
    for part in groups:
        alert[part], silent[part], best[part] = utility_terms(y[part])
    return alert, silent, best


def utility_of(mask, alert, silent, best):
    """Normalized utility of one alert mask against precomputed per-hour terms."""
    span = np.where(best, alert, silent).sum() - silent.sum()
    return float((np.where(mask, alert, silent).sum() - silent.sum()) / span) if span > 0 else None


def normalized_utility(observed, best, inaction):
    """Official normalization; undefined when no selected patient can earn utility."""
    span = best.sum() - inaction.sum()
    return float((observed.sum() - inaction.sum()) / span) if span > 0 else None


def logit(p):
    """Log-odds of a score, clipped away from the open interval's ends."""
    clipped = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(clipped / (1 - clipped))


def logit_fit(y, p):
    """Intercept and slope of an unpenalized logistic fit on logit scores, or (None, None)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(np.unique(y)) != 2 or len(np.unique(p)) < 2:
        return None, None
    fit = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def recalibrator(y, p):
    """Platt map fitted on validation predictions only; None when degenerate or not increasing.

    The map is strictly monotone, so it changes reported probabilities, Brier
    score and reliability, but never the ranking and never which hours alert
    once the threshold is passed through the same map. Recalibration cannot
    improve discrimination and is not clinical calibration for an ER.
    """
    intercept, slope = logit_fit(y, p)
    if intercept is None or not np.isfinite([intercept, slope]).all() or slope <= 0:
        return None
    return dict(method="platt_logit", fitted_on="validation", intercept=intercept, slope=slope)


def recalibrate(fit, p):
    """Apply a frozen recalibration map to scores or to a threshold."""
    return 1 / (1 + np.exp(-(fit["intercept"] + fit["slope"] * logit(p))))


def subgroup_metrics(y, p, threshold, ids, levels):
    """Descriptive held-out metrics for patient subgroups assigned before scoring.

    `levels` maps a patient identifier to one recorded administrative label.
    Nothing is fitted or selected here and no interval is reported: small
    subgroups give unstable estimates, so these are exploratory descriptions of
    one cohort, not subgroup validation or evidence about fairness in care.
    """
    y, p, ids = np.asarray(y), np.asarray(p, dtype=float), np.asarray(ids)
    alerts = p >= threshold
    groups = patient_groups(ids)
    observed, best, inaction = utility_by_patient(y, alerts, groups)
    assigned = np.array([str(levels.get(ids[part.start], "unknown")) for part in groups])
    report = {}
    for level in sorted(set(assigned)):
        take = assigned == level
        rows = np.concatenate([np.arange(part.start, part.stop)
                               for part, keep in zip(groups, take) if keep])
        sy, sp, sa = y[rows], p[rows], alerts[rows]
        hits = float(np.sum(sa & (sy == 1)))
        report[level] = dict(
            patients=int(take.sum()), hours=int(len(rows)),
            positive_hour_fraction=float(sy.mean()),
            auroc=float(roc_auc_score(sy, sp)) if len(np.unique(sy)) == 2 else None,
            average_precision=float(average_precision_score(sy, sp)) if sy.any() else None,
            precision=float(hits / sa.sum()) if sa.any() else None,
            recall=float(hits / sy.sum()) if sy.any() else None,
            alert_hours_per_100=float(100 * sa.mean()),
            normalized_utility=normalized_utility(observed[take], best[take], inaction[take]))
    return report


def calibration(y, p, bins=10):
    """Equal-count reliability table, expected calibration error and a logit recalibration fit."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    parts = [g for g in np.array_split(np.argsort(p, kind="stable"), min(bins, len(p))) if len(g)]
    table = [dict(count=int(len(g)), mean_score=float(p[g].mean()), observed_rate=float(y[g].mean()))
             for g in parts]
    error = float(sum(len(g) / len(p) * abs(p[g].mean() - y[g].mean()) for g in parts))
    intercept, slope = logit_fit(y, p)
    # A perfectly calibrated score has intercept 0 and slope 1; slope < 1 means overextended scores.
    return dict(expected_calibration_error=error, intercept=intercept, slope=slope, bins=table)


def paired_difference(y, first, second, ids, seed=42, draws=1000, level=.95):
    """Interval for the average-precision difference between two frozen score vectors.

    The same resampled patients are scored under both, so the comparison is
    paired. Run on validation predictions to ask whether a selection margin is
    distinguishable from resampling noise; it is not a hypothesis test and it
    says nothing about which model would generalize.
    """
    y = np.asarray(y)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    rows = [np.arange(part.start, part.stop) for part in patient_groups(ids)]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        index = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if len(np.unique(y[index])) == 2:
            differences.append(average_precision_score(y[index], first[index])
                               - average_precision_score(y[index], second[index]))
    if len(differences) < draws // 2:
        return None
    half = (1 - level) / 2
    return dict(draws=len(differences), level=level, seed=seed,
                observed=float(average_precision_score(y, first) - average_precision_score(y, second)),
                low=float(np.percentile(differences, 100 * half)),
                high=float(np.percentile(differences, 100 * (1 - half))))


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
