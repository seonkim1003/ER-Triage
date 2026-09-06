import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ertriage.basis import basis, basis_report
from ertriage.contrasts import contrasts, levels_for
from ertriage.data import COLUMNS
from ertriage.history import HeldoutRun
from ertriage.target import ONSET_SHIFT


PATIENTS, HOURS, ONSET = 60, 40, 14
#: Positivity is deliberately coprime with the descriptor strides below, so no
#: descriptor level coincides with "never positive" and every contrast is defined.
POSITIVE = [i for i in range(PATIENTS) if i % 5]


@pytest.fixture
def cohort_run(tmp_path):
    """A frozen run: 60 patients, scores that carry real signal, recorded descriptors."""
    root, run = tmp_path / "data", tmp_path / "run"
    (root / "site_A").mkdir(parents=True)
    run.mkdir()
    rng = np.random.default_rng(0)
    manifest, predictions = [], []
    for i in range(PATIENTS):
        onset = ONSET if i in POSITIVE else None
        n = HOURS
        labels = np.zeros(n, dtype=int)
        if onset is not None:
            labels[onset:] = 1
        pid = f"site_A/p{i:03d}.psv"
        frame = pd.DataFrame(np.nan, index=range(n), columns=COLUMNS)
        frame["ICULOS"], frame["SepsisLabel"] = np.arange(1, n + 1), labels
        frame["HR"] = np.arange(n) + 70
        frame.to_csv(root / pid, sep="|", index=False)
        scores = np.clip(.15 + .5 * labels + rng.normal(0, .18, n), .001, .999)
        manifest.append(dict(patient=pid, hours=n,
                             sha256=hashlib.sha256((root / pid).read_bytes()).hexdigest(),
                             site="site_A", age_band=f"band{i % 3}",
                             gender=f"gender_{i % 2}", unit=f"unit{i % 2}"))
        predictions.append(pd.DataFrame(dict(patient=pid, hour=frame.ICULOS, label=labels, score=scores)))
    pd.DataFrame(manifest).to_csv(run / "test_patients.csv", index=False)
    pd.concat(predictions).to_csv(run / "test_predictions.csv", index=False)
    for split in ("train", "validation"):
        pd.DataFrame(dict(patient=[f"site_A/{split}.psv"])).to_csv(run / f"{split}_patients.csv", index=False)
    (run / "metrics.json").write_text(json.dumps(
        {"selected_model": "boosting", "models": {"boosting": {"validation": {"threshold": .5}}}}))
    return root, run


def test_basis_isolates_post_onset_hours_and_verifies_the_horizon_six_identity(cohort_run):
    report, per_patient = basis_report(HeldoutRun(*cohort_run), horizon=6, seed=42, draws=100)
    assert report["horizon6_label_identity"] is True
    onset, positives = ONSET + ONSET_SHIFT, len(POSITIVE)
    # Each positive patient contributes HOURS - onset hours at or after the onset proxy.
    assert report["hours_dropped_at_or_after_onset"] == positives * (HOURS - onset)
    assert report["hours_all"] == report["hours_preonset"] + report["hours_dropped_at_or_after_onset"]
    full, pre = report["all_hours_persistent_label"], report["preonset_hours_event_label"]
    assert full["positive_hours"] > pre["positive_hours"] == positives * 6
    assert report["auroc_difference"]["low"] <= report["auroc_difference"]["observed"] <= report["auroc_difference"]["high"]
    assert set(per_patient.columns) >= {"patient", "status", "usable", "preonset_hours", "dropped_hours"}
    assert per_patient.usable.all()


def test_a_longer_horizon_changes_the_question_and_drops_the_identity(cohort_run):
    six, _ = basis_report(HeldoutRun(*cohort_run), horizon=6, seed=42, draws=60)
    twelve, _ = basis_report(HeldoutRun(*cohort_run), horizon=12, seed=42, draws=60)
    assert twelve["horizon6_label_identity"] is None  # only checked at horizon 6
    # Masking is horizon-independent; only which pre-onset hours count as positive changes.
    assert twelve["hours_preonset"] == six["hours_preonset"]
    assert twelve["preonset_hours_event_label"]["positive_hours"] > six["preonset_hours_event_label"]["positive_hours"]


def test_basis_writes_its_artifacts_and_refuses_to_overwrite(cohort_run, tmp_path):
    root, run = cohort_run
    report = basis(root, run, tmp_path / "basis", horizon=6, draws=50)
    for name in ("BASIS.md", "basis.json", "patient_basis.csv"):
        assert (tmp_path / "basis" / name).exists()
    text = (tmp_path / "basis" / "BASIS.md").read_text(encoding="utf-8")
    assert "identical to the" in text and "recoverable onset proxy" in text
    json.dumps(report, allow_nan=False)  # every reported value stays JSON-safe
    with pytest.raises(ValueError, match="new output"):
        basis(root, run, tmp_path / "basis", horizon=6, draws=50)


def test_contrasts_read_descriptors_from_the_run_manifest(cohort_run):
    _, run = cohort_run
    levels = levels_for(run)
    assert set(levels) == {"site", "age_band", "gender", "unit"}
    assert set(levels["age_band"].values()) == {"band0", "band1", "band2"}


def test_contrasts_command_dedupes_binary_descriptors_end_to_end(cohort_run, tmp_path):
    root, run = cohort_run
    report = contrasts(root, run, tmp_path / "contrasts", draws=100)
    result = report["contrasts"]
    # site is single-level and drops out; gender and unit are binary and give one each;
    # age_band has three levels and gives three.
    assert result["family_size"] == 5
    assert {c["family"] for c in result["contrasts"]} == {"age_band", "gender", "unit"}
    assert all(c["p_value"] <= c["p_value_holm"] for c in result["contrasts"])
    assert (tmp_path / "contrasts" / "CONTRASTS.md").exists()
    assert len(pd.read_csv(tmp_path / "contrasts" / "subgroup_contrasts.csv")) == 5
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="new output"):
        contrasts(root, run, tmp_path / "contrasts", draws=10)


def test_contrasts_refuse_a_manifest_without_any_prespecified_descriptor(cohort_run, tmp_path):
    root, run = cohort_run
    manifest = pd.read_csv(run / "test_patients.csv").drop(columns=["site", "age_band", "gender", "unit"])
    manifest.to_csv(run / "test_patients.csv", index=False)
    with pytest.raises(ValueError, match="none of the prespecified descriptors"):
        contrasts(root, run, tmp_path / "none", draws=10)
