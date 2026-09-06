import json

import numpy as np
import pandas as pd
import pytest

from ertriage.evaluate import recalibrator
from ertriage.sweep import matched_burden
from ertriage.transfer import summarize, transfer, transfer_curve
from tests.test_basis import cohort_run  # noqa: F401  (frozen-run fixture)


def curve(run, early_by_burden, family="boosting", column="fp_patients"):
    """A synthetic sweep curve: burden on `column`, coverage rising with it."""
    rows = []
    for burden, early in early_by_burden:
        rows.append(dict(run=run, target="t", family=family, selected=True, budget=burden,
                         threshold=.5, test_alert_hours_per_100=burden, fp_patients=burden,
                         eligible=100, early=early, early_low=early - 2, early_high=early + 2,
                         before_onset=early + 5, episodes_per_100_hours=burden / 10))
    frame = pd.DataFrame(rows)
    if column != "fp_patients":
        frame["fp_patients"] = frame.fp_patients * 2
    return frame


def test_matched_burden_interpolates_and_signs_the_difference():
    a = curve("first", [(1., 10.), (2., 20.), (3., 30.)])
    b = curve("second", [(1., 15.), (2., 25.), (3., 35.)])
    matched = matched_burden(a, b)
    assert set(matched.matched_on) == {"alert_hours_per_100", "nonsepsis_patients_alerted"}
    # Second is uniformly five points better, at every matched level.
    assert np.allclose(matched.difference, 5.)
    # Column names must not shadow DataFrame methods, or attribute access returns one.
    assert (matched["second_run"] == "second").all() and (matched["first_run"] == "first").all()
    assert matched["first_run"].tolist() and not callable(matched["first_run"])
    half = matched_burden(a, b, points=(1.5,))
    assert np.allclose(half.first_coverage, 15.) and np.allclose(half.second_coverage, 20.)


def test_matched_burden_never_extrapolates_past_a_measured_budget():
    a = curve("first", [(1., 10.), (2., 20.)])
    b = curve("second", [(2., 25.), (3., 35.)])
    # Only level 2.0 lies inside both curves; 1.0 and 3.0 would extrapolate one of them.
    assert set(matched_burden(a, b).level) == {2.}


def test_matched_burden_uses_each_denominator_separately():
    """The two burdens can rank the same pair of runs differently."""
    a = curve("first", [(1., 10.), (2., 20.), (3., 30.)])
    b = curve("second", [(1., 12.), (2., 22.), (3., 32.)])
    b = b.assign(test_alert_hours_per_100=b.test_alert_hours_per_100 * 2)  # same coverage, double the hours
    matched = matched_burden(a, b)
    by_hours = matched[matched.matched_on == "alert_hours_per_100"]
    by_patients = matched[matched.matched_on == "nonsepsis_patients_alerted"]
    assert (by_patients.difference > 0).all()  # ahead per patient disturbed
    assert (by_hours.difference < 0).all()     # behind per alert hour
    assert matched_burden(a, b.iloc[0:0]).empty


def cohort(patients=400, hours=20, seed=0, shift=1.8, signal=1.5):
    """Held-out scores that are over-extreme, as scores carried to an unseen site are.

    A latent log-odds drives the labels honestly; the reported score inflates the
    same log-odds by `shift`. The calibration slope is then 1/shift by
    construction, so the fixture states the answer the test is checking for.
    """
    rng = np.random.default_rng(seed)
    y, p, ids = [], [], []
    for i in range(patients):
        true_logit = signal * rng.normal(size=hours)
        labels = (rng.random(hours) < 1 / (1 + np.exp(-true_logit))).astype(int)
        y.append(labels)
        p.append(1 / (1 + np.exp(-shift * true_logit)))
        ids.append(np.repeat(f"p{i:04d}", hours))
    return np.concatenate(y), np.concatenate(p), np.concatenate(ids)


def test_transfer_never_scores_a_patient_with_a_map_fitted_on_itself():
    y, p, ids = cohort()
    frame = transfer_curve(y, p, ids, sizes=(50, 100), repeats=3, seed=1)
    assert set(frame.patients_used) == {50, 100}
    for size, group in frame.groupby("patients_used"):
        assert (group.evaluation_patients == 400 - size).all()
    assert frame.fitted.all()


def test_transfer_recovers_calibration_and_improves_with_more_target_patients():
    y, p, ids = cohort(seed=3, shift=2.0)
    table = summarize(transfer_curve(y, p, ids, sizes=(25, 300), repeats=8, seed=2))
    small, large = table.iloc[0], table.iloc[1]
    # Over-extreme scores, so the slope sits below one; here 1/shift by construction.
    assert small.slope_before < .85 and abs(small.slope_before - 0.5) < .15
    # Fitting on the target site moves the slope to about one, at both sample sizes.
    for row in (small, large):
        assert abs(row.slope_after - 1) < .15 < abs(row.slope_before - 1)
        assert row.slope_after_q25 <= row.slope_after <= row.slope_after_q75
    # A two-parameter map is already well determined by 25 patients, so more target
    # data buys stability across draws rather than a better median.
    assert (large.slope_after_q75 - large.slope_after_q25) < (small.slope_after_q75 - small.slope_after_q25)
    assert large.brier_after < large.brier_before


def test_transfer_compares_against_a_frozen_source_map_when_one_exists():
    y, p, ids = cohort(seed=5)
    source = recalibrator(y[:2000], p[:2000])
    frame = transfer_curve(y, p, ids, frozen=source, sizes=(100,), repeats=3, seed=4)
    assert {"slope_source_map", "brier_source_map"} <= set(frame.columns)
    assert frame.slope_source_map.notna().all()


def test_transfer_skips_sizes_that_would_leave_nothing_to_evaluate():
    y, p, ids = cohort(patients=60)
    frame = transfer_curve(y, p, ids, sizes=(10, 60, 500), repeats=2, seed=6)
    assert set(frame.patients_used) == {10}  # 60 and 500 leave no evaluation patients
    with pytest.raises(ValueError, match="too few patients"):
        transfer_curve(y, p, ids, sizes=(500,), repeats=2, seed=6)


def test_transfer_command_writes_artifacts_and_refuses_to_overwrite(cohort_run, tmp_path):
    root, run = cohort_run
    report = transfer(root, run, tmp_path / "transfer", sizes=(10, 20), repeats=3, seed=1)
    for name in ("TRANSFER.md", "transfer.json", "transfer_draws.csv", "transfer_summary.csv"):
        assert (tmp_path / "transfer" / name).exists()
    assert [r["patients_used"] for r in report["summary"]] == [10, 20]
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="new output"):
        transfer(root, run, tmp_path / "transfer", sizes=(10,), repeats=1, seed=1)


def test_transfer_summary_is_json_safe():
    y, p, ids = cohort(seed=7)
    table = summarize(transfer_curve(y, p, ids, sizes=(50,), repeats=3, seed=8))
    json.dumps(json.loads(table.to_json(orient="records")), allow_nan=False)
