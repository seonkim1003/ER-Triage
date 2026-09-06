"""Early-window coverage against alert burden, with every threshold chosen on validation.

A single operating point cannot compare two models that spend their alerts
differently. The nominal alert budget does not fix this: the same 2% validation
budget produces different test alert loads under different targets, so a
comparison at one budget silently compares two burdens as well as two models.

This sweeps the validation alert-hour budget over a prespecified grid. Each
budget fixes a threshold using validation hours alone, and the held-out cohort is
then scored at that frozen threshold, so no test hour informs any threshold. The
result is a curve of coverage against burden for each model, and burden has two
different denominators that do not agree: alert hours, and how many patients are
disturbed. `matched_burden` interpolates two curves to common values of each.

Reading a run's own validation scores means loading the model it saved. As in
replay, only load `.joblib` files this project generated locally: the
serialization format can execute code.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import read_patient
from .early_warning import summarize_warning, warning_patient
from .model import fitted, target_view, threshold_for
from .target import HORIZON

BUDGETS = (0.5, 1.0, 2.0, 3.0, 5.0)
FAMILIES = ("logistic", "boosting")
MATCH_POINTS = (1.0, 2.0, 3.0)
BURDENS = {"alert_hours_per_100": "test_alert_hours_per_100",
           "nonsepsis_patients_alerted": "fp_patients"}

NOTE = ("Every threshold is chosen on validation hours alone and frozen before the held-out cohort "
        "is scored. Coverage is the early-warning fraction defined in early_warning.py, over the "
        "same timing-eligible patients. Curves describe one cohort at one seed; they are not a "
        "clinical operating characteristic and none of these budgets is a stated review capacity.")


def _target_of(run):
    """The target a run was trained against, defaulting to the persistent label."""
    report = json.loads((Path(run) / "metrics.json").read_text())
    return report.get("target", "persistent"), int(report.get("horizon_hours", HORIZON)), report


def run_curve(root, run, budgets=BUDGETS, families=FAMILIES, seed=42, draws=400):
    """Coverage and burden at each validation budget, for each model family in a run."""
    root, run = Path(root), Path(run)
    target, horizon, report = _target_of(run)
    names = {s: pd.read_csv(run / f"{s}_patients.csv").patient.tolist() for s in ("validation", "test")}
    frames = {p: read_patient(root / p) for p in set(names["validation"]) | set(names["test"])}
    views = {s: target_view([frames[p] for p in names[s]], np.array(names[s]), target, horizon)
             for s in ("validation", "test")}
    vx, vy, vi = fitted(views["validation"])
    tx = views["test"][0]
    test = names["test"]
    bounds = np.cumsum([0] + [len(frames[p]) for p in test])
    rows = []
    with threadpool_limits(limits=2):
        for family in families:
            bundle = joblib.load(run / f"{family}.joblib")
            columns = bundle["feature_columns"]
            vp = bundle["model"].predict_proba(vx[columns])[:, 1]
            tp = bundle["model"].predict_proba(tx[columns])[:, 1]
            for budget in budgets:
                threshold = threshold_for(vy, vp, vi, rule="budget", budget=budget)
                per = pd.DataFrame([dict(patient=pid, **warning_patient(
                    frames[pid].ICULOS, frames[pid].SepsisLabel, tp[bounds[i]:bounds[i + 1]], threshold))
                    for i, pid in enumerate(test)])
                summary = summarize_warning(per, seed=seed, draws=draws)
                measures = summary["measures"]
                interval = summary["bootstrap"]["intervals"]["early_warning_fraction"]
                rows.append(dict(
                    run=run.name, target=target, family=family, selected=report["selected_model"] == family,
                    budget=budget, threshold=threshold,
                    test_alert_hours_per_100=float(100 * (tp >= threshold).mean()),
                    eligible=summary["timing_eligible_patients"],
                    early=100 * measures["early_warning_fraction"],
                    early_low=100 * interval["low"], early_high=100 * interval["high"],
                    before_onset=100 * measures["before_onset_detection_fraction"],
                    fp_patients=100 * measures["nonsepsis_patients_with_any_alert"],
                    episodes_per_100_hours=measures["alert_episodes_per_100_hours"]))
    return pd.DataFrame(rows)


def matched_burden(first, second, points=MATCH_POINTS, burdens=BURDENS):
    """Interpolate two curves to common burdens, one row per family and burden point.

    Linear interpolation on the swept grid, and only inside the range both curves
    actually cover, so no row extrapolates past a measured budget.
    """
    rows = []
    for family in sorted(set(first.family) & set(second.family)):
        a = first[first.family == family].sort_values("budget")
        b = second[second.family == family].sort_values("budget")
        for burden, column in burdens.items():
            low, high = max(a[column].min(), b[column].min()), min(a[column].max(), b[column].max())
            for point in points:
                if not low <= point <= high:
                    continue
                first_at = float(np.interp(point, a[column], a.early))
                second_at = float(np.interp(point, b[column], b.early))
                rows.append(dict(family=family, matched_on=burden, level=point,
                                 first_run=a.run.iloc[0], second_run=b.run.iloc[0],
                                 first_coverage=first_at, second_coverage=second_at,
                                 difference=second_at - first_at))
    return pd.DataFrame(rows)


def _tally(matched):
    """How often the second run wins at each burden denominator."""
    out = {}
    for burden, group in matched.groupby("matched_on"):
        out[burden] = dict(comparisons=len(group), second_better=int((group.difference > 0).sum()),
                           mean_difference=float(group.difference.mean()),
                           low=float(group.difference.min()), high=float(group.difference.max()))
    return out


def _markdown(report, curves, matched):
    lines = ["# Coverage against alert burden", "",
             f"Budgets {', '.join(str(b) for b in report['budgets'])} alert hours per 100, chosen on "
             f"validation only. Bootstrap {report['draws']} draws, seed {report['seed']}.", "",
             "| Run | Target | Family | Budget | Test alerts/100h | Early window | Nonsepsis alerted | Episodes/100h |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for _, r in curves.iterrows():
        lines.append(f"| {r['run']} | {r['target']} | {r['family']}{'*' if r['selected'] else ''} | "
                     f"{r['budget']:.1f} | {r['test_alert_hours_per_100']:.2f} | "
                     f"{r['early']:.1f}% [{r['early_low']:.1f}, {r['early_high']:.1f}] | "
                     f"{r['fp_patients']:.2f}% | {r['episodes_per_100_hours']:.3f} |")
    lines += ["", "An asterisk marks the model that run selected on validation.", ""]
    if matched is not None and len(matched):
        tally = _tally(matched)
        lines += ["## Matched burden", "",
                  f"Coverage of `{matched['second_run'].iloc[0]}` minus "
                  f"`{matched['first_run'].iloc[0]}`, both "
                  "interpolated to the same burden. Positive favours the second.", "",
                  "| Matched on | Comparisons | Second better | Mean difference | Range |",
                  "|---|---:|---:|---:|---:|"]
        for burden, t in sorted(tally.items()):
            lines.append(f"| {burden} | {t['comparisons']} | {t['second_better']} | "
                         f"{t['mean_difference']:+.1f}pp | {t['low']:+.1f} to {t['high']:+.1f} |")
        lines += ["", "| Family | Matched on | Level | First | Second | Difference |",
                  "|---|---|---:|---:|---:|---:|"]
        for _, r in matched.iterrows():
            lines.append(f"| {r['family']} | {r['matched_on']} | {r['level']:.1f} | "
                         f"{r['first_coverage']:.1f}% | {r['second_coverage']:.1f}% | {r['difference']:+.1f}pp |")
        lines += ["", "The two denominators need not agree, and when they disagree neither run is "
                      "preferable without a stated review capacity.", ""]
    return "\n".join(lines + [NOTE, ""])


def sweep(root, run, out, against=None, budgets=BUDGETS, seed=42, draws=400):
    out = Path(out)
    if out.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    curves = run_curve(root, run, budgets, seed=seed, draws=draws)
    matched = None
    if against:
        other = run_curve(root, against, budgets, seed=seed, draws=draws)
        matched = matched_burden(curves, other, )
        curves = pd.concat([curves, other], ignore_index=True)
    report = dict(run=str(Path(run).resolve()), against=str(Path(against).resolve()) if against else None,
                  budgets=list(budgets), seed=seed, draws=draws, note=NOTE,
                  tally=_tally(matched) if matched is not None and len(matched) else None,
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))})
    out.mkdir(parents=True)
    curves.to_csv(out / "sweep.csv", index=False)
    if matched is not None:
        matched.to_csv(out / "matched_burden.csv", index=False)
    (out / "sweep.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "SWEEP.md").write_text(_markdown(report, curves, matched), encoding="utf-8")
    print(_markdown(report, curves, matched))
    return report
