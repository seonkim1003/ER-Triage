"""Prespecified subgroup contrasts on a frozen run, with multiplicity control.

Earlier releases described each subgroup level on its own and said plainly that
they carried no intervals and no multiplicity control. That combination invites
exactly the error the previous full-cohort release had to retract: scan enough
levels and some of them look alarming, with nothing to say whether the gap
survives resampling or is the largest of many draws from noise.

This states the comparisons instead. The descriptor family is fixed in code
before any held-out hour is read, every level-versus-rest difference gets a
whole-patient bootstrap interval, and Holm-Bonferroni is applied across the
family at once. It is a post-hoc description of one frozen cohort, like the
timing and workload audits, so it lives beside them as its own command and
refits nothing.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from .evaluate import CONTRAST_FAMILY, subgroup_contrasts
from .history import HeldoutRun

NOTE = ("One held-out cohort, described after the fact. A contrast that survives correction is a "
        "difference in this cohort's recorded administrative strata, not evidence about care, "
        "deployment, or any other population. Levels are recorded fields, not clinical categories, "
        "and 'unrecorded' is itself a level rather than a missing value to be imputed.")


def levels_for(run, families=CONTRAST_FAMILY):
    """Descriptor levels for the held-out patients, read from the run's own manifest."""
    manifest = pd.read_csv(Path(run) / "test_patients.csv")
    if "patient" not in manifest:
        raise ValueError("Held-out manifest needs a patient column")
    present = [f for f in families if f in manifest.columns]
    if not present:
        raise ValueError("Held-out manifest carries none of the prespecified descriptors")
    indexed = manifest.set_index("patient")
    return {f: indexed[f].astype(str).to_dict() for f in present}


def _markdown(report):
    result = report["contrasts"]
    lines = [
        "# Prespecified subgroup contrasts", "",
        f"Frozen run `{report['run']}`, model `{report['selected_model']}`. Nothing was refitted. "
        f"Family fixed in code before scoring: {', '.join(report['families'])}.", "",
        f"{result['family_size']} contrasts, {result['draws']} whole-patient draws, seed "
        f"{result['seed']}, {result['correction'].replace('_', '-')} correction.", "",
        "| Descriptor | Level | Patients | AUROC vs rest | 95% interval | p | p after Holm | Survives |",
        "|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for c in sorted(result["contrasts"], key=lambda z: z["p_value_holm"]):
        lines.append(
            f"| {c['family']} | {c['level']} | {c['patients']:,} | {c['auroc_difference']:+.4f} | "
            f"[{c['low']:+.3f}, {c['high']:+.3f}] | {c['p_value']:.4f} | {c['p_value_holm']:.4f} | "
            f"{'yes' if c['significant_at_05_after_holm'] else 'no'} |")
    survivors = [c for c in result["contrasts"] if c["significant_at_05_after_holm"]]
    lines += ["", (f"{len(survivors)} of {result['family_size']} contrasts survive correction at 0.05."
                   if survivors else
                   f"No contrast among the {result['family_size']} survives correction at 0.05."), ""]
    if result["undefined_contrasts"]:
        lines += ["Undefined contrasts (a side lacked both label classes): "
                  + ", ".join(f"{c['family']}/{c['level']}" for c in result["undefined_contrasts"]), ""]
    lines += [report["note"], "", result["note"], ""]
    return "\n".join(lines)


def contrasts(root, run, out, seed=42, draws=1000):
    out = Path(out)
    if out.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    cohort = HeldoutRun(root, run)
    predictions = cohort.predictions
    result = subgroup_contrasts(predictions.label.to_numpy(), predictions.score.to_numpy(),
                                predictions.patient.to_numpy(), levels_for(run), seed=seed, draws=draws)
    if not result:
        raise ValueError("No usable contrast in the prespecified family")
    report = dict(run=str(cohort.run), selected_model=cohort.selected, threshold=cohort.threshold,
                  families=list(levels_for(run)), contrasts=result, note=NOTE,
                  provenance=cohort.provenance())
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path(__file__).parent.glob("*.py"))}
    out.mkdir(parents=True)
    (out / "contrasts.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "CONTRASTS.md").write_text(_markdown(report), encoding="utf-8")
    pd.DataFrame(result["contrasts"]).to_csv(out / "subgroup_contrasts.csv", index=False)
    print(_markdown(report))
    return report
