"""How much target-site data a recalibration map needs before it transfers.

The recalibration already in this project is fitted on validation predictions
from the source sites and frozen. It closes most of the within-source calibration
gap and does not transfer: on an unseen site the scores stay systematically too
high. "Cross-site recalibration that actually transfers" has sat on the remaining
work list since, stated as a goal with no measurement attached.

This measures the price. A Platt map is fitted on a random sample of `k` patients
from the held-out site and evaluated on the held-out patients that sample did not
touch, for a grid of `k` and several draws each. The curve says how many labelled
target-site patients would be needed, if any number would do.

Two things this is not. It is not prospective: it presumes labelled outcomes from
the target site already exist, which in deployment means having waited for them.
And a map fitted on target patients and scored on other target patients from the
same frozen cohort is an upper bound on what a real transfer would achieve, since
both sides share the cohort's idiosyncrasies. Nothing here is refitted or
re-thresholded, and no model file is loaded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .evaluate import calibration, logit_fit, recalibrate, recalibrator
from .history import HeldoutRun

SIZES = (25, 50, 100, 250, 500, 1000)
REPEATS = 20

NOTE = ("Descriptive re-summary of frozen held-out scores. Patients spent on fitting are excluded "
        "from the evaluation they inform, so no patient is scored by a map fitted on itself. "
        "Calibration slope 1 and intercept 0 mean scores that neither overstate nor understate on "
        "average; slope alone does not make a score clinically meaningful, and recalibration is "
        "monotone so it changes no ranking and no alert ordering.")


def _quality(y, p):
    """Calibration slope and intercept plus Brier, or None where undefined."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(np.unique(y)) != 2:
        return None
    intercept, slope = logit_fit(y, p)
    return dict(slope=slope, intercept=intercept, brier=float(brier_score_loss(y, p)),
                calibration=calibration(y, p))


def transfer_curve(y, p, ids, frozen=None, sizes=SIZES, repeats=REPEATS, seed=42):
    """Fit a Platt map on k held-out patients, score the rest, across a grid of k."""
    y, p, ids = np.asarray(y), np.asarray(p, dtype=float), np.asarray(ids)
    patients = pd.unique(ids)
    rows = np.arange(len(y))
    index = {pid: rows[ids == pid] for pid in patients}
    rng = np.random.default_rng(seed)
    records = []
    for size in sizes:
        if size >= len(patients):
            continue
        for repeat in range(repeats):
            chosen = rng.choice(patients, size=size, replace=False)
            fit_rows = np.concatenate([index[pid] for pid in chosen])
            mask = np.ones(len(y), dtype=bool)
            mask[fit_rows] = False  # patients spent on fitting never score themselves
            fit = recalibrator(y[fit_rows], p[fit_rows])
            before = _quality(y[mask], p[mask])
            if before is None:
                continue
            record = dict(patients_used=size, repeat=repeat, fitted=fit is not None,
                          evaluation_patients=len(patients) - size,
                          slope_before=before["slope"], intercept_before=before["intercept"],
                          brier_before=before["brier"])
            if fit is not None:
                after = _quality(y[mask], recalibrate(fit, p[mask]))
                record.update(slope_after=after["slope"], intercept_after=after["intercept"],
                              brier_after=after["brier"])
            if frozen is not None:
                source = _quality(y[mask], recalibrate(frozen, p[mask]))
                record.update(slope_source_map=source["slope"], intercept_source_map=source["intercept"],
                              brier_source_map=source["brier"])
            records.append(record)
    if not records:
        raise ValueError("No usable transfer draw; the cohort has too few patients")
    return pd.DataFrame(records)


def summarize(curve):
    """Median and interquartile range of each measure at each sample size."""
    out = []
    for size, group in curve.groupby("patients_used"):
        row = dict(patients_used=int(size), draws=len(group),
                   fitted=int(group.fitted.sum()), evaluation_patients=int(group.evaluation_patients.iloc[0]))
        for column in group.columns:
            if column.startswith(("slope_", "intercept_", "brier_")) and group[column].notna().any():
                values = group[column].dropna()
                row[column] = float(values.median())
                row[column + "_q25"] = float(values.quantile(.25))
                row[column + "_q75"] = float(values.quantile(.75))
        out.append(row)
    return pd.DataFrame(out)


def _markdown(report, table):
    has_source = "slope_source_map" in table
    lines = ["# Cross-site recalibration transfer", "",
             f"Frozen run `{report['run']}`, model `{report['selected_model']}`. A Platt map is "
             f"fitted on k held-out-site patients and scored on the remaining "
             f"{report['patients']:,} minus k, {report['repeats']} draws per k, seed "
             f"{report['seed']}. Nothing is refitted and no model file is loaded.", "",
             "Calibration slope, median [IQR]. A slope below 1 means scores that are too extreme.", "",
             "| Patients used to fit | Evaluated on | Uncorrected | Fitted on target site |"
             + (" Source-fitted map |" if has_source else ""),
             "|---:|---:|---:|---:|" + ("---:|" if has_source else "")]
    for _, r in table.iterrows():
        line = (f"| {int(r['patients_used']):,} | {int(r['evaluation_patients']):,} | "
                f"{r['slope_before']:.3f} | ")
        line += (f"{r['slope_after']:.3f} [{r['slope_after_q25']:.3f}, {r['slope_after_q75']:.3f}] |"
                 if "slope_after" in r and pd.notna(r.get("slope_after")) else "undefined |")
        if has_source:
            line += f" {r['slope_source_map']:.3f} |"
        lines.append(line)
    lines += ["", "Brier score, median. Lower is better.", "",
              "| Patients used to fit | Uncorrected | Fitted on target site |"
              + (" Source-fitted map |" if has_source else ""),
              "|---:|---:|---:|" + ("---:|" if has_source else "")]
    for _, r in table.iterrows():
        line = f"| {int(r['patients_used']):,} | {r['brier_before']:.5f} | "
        line += (f"{r['brier_after']:.5f} |" if pd.notna(r.get("brier_after")) else "undefined |")
        if has_source:
            line += f" {r['brier_source_map']:.5f} |"
        lines.append(line)
    return "\n".join(lines + ["", report["note"], ""])


def transfer(root, run, out, sizes=SIZES, repeats=REPEATS, seed=42):
    out = Path(out)
    if out.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    cohort = HeldoutRun(root, run)
    predictions = cohort.predictions
    report_json = json.loads((Path(run) / "metrics.json").read_text())
    frozen = report_json["models"][cohort.selected].get("recalibration")
    curve = transfer_curve(predictions.label.to_numpy(), predictions.score.to_numpy(),
                           predictions.patient.to_numpy(), frozen, sizes, repeats, seed)
    table = summarize(curve)
    report = dict(run=str(cohort.run), selected_model=cohort.selected,
                  patients=len(cohort.patients), sizes=list(sizes), repeats=repeats, seed=seed,
                  source_map=frozen, note=NOTE, summary=json.loads(table.to_json(orient="records")),
                  provenance=cohort.provenance(),
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))})
    out.mkdir(parents=True)
    curve.to_csv(out / "transfer_draws.csv", index=False)
    table.to_csv(out / "transfer_summary.csv", index=False)
    (out / "transfer.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "TRANSFER.md").write_text(_markdown(report, table), encoding="utf-8")
    print(_markdown(report, table))
    return report
