import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ertriage.data import COLUMNS, features, patient_attributes, read_patient
from ertriage.model import select_model, split_patients, threshold_for, metrics
from ertriage.replay import policy


def patient(n=12):
    df = pd.DataFrame(np.nan, index=range(n), columns=COLUMNS)
    df["ICULOS"] = np.arange(1, n + 1)
    df["HR"] = [np.nan, 80, np.nan, 90] * (n // 4)
    df["SepsisLabel"] = 0
    return df


def test_features_never_use_future_or_labels():
    df = patient()
    expected = features(df.iloc[:5])
    df.loc[5:, "HR"] = 10000
    df["SepsisLabel"] = 1
    assert_frame_equal(expected, features(df).iloc[:5])
    assert np.isnan(expected.HR.iloc[0])
    assert expected.HR.iloc[2] == 80
    assert expected.HR_age.iloc[0] == 999
    assert expected.HR_age.iloc[2] == 1


def test_patient_split_is_disjoint_reproducible_stratified():
    manifest = pd.DataFrame({"patient": [f"p{i}" for i in range(100)], "ever_sepsis": [0, 1] * 50})
    a, b = split_patients(manifest, 42), split_patients(manifest, 42)
    sets = [set(s.patient) for s in a.values()]
    assert len(set.union(*sets)) == 100
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    for name in a:
        assert_frame_equal(a[name], b[name])
        assert a[name].ever_sepsis.nunique() == 2


def test_schema_rejects_invalid_labels_and_time(tmp_path):
    df = patient()
    path = tmp_path / "p.psv"
    df.loc[1, "SepsisLabel"] = 2
    df.to_csv(path, sep="|", index=False)
    with pytest.raises(ValueError, match="labels"):
        read_patient(path)
    df["SepsisLabel"] = 0
    df.loc[1, "ICULOS"] = 999
    df.to_csv(path, sep="|", index=False)
    with pytest.raises(ValueError, match="hourly"):
        read_patient(path)


def test_threshold_and_single_class_metrics():
    assert threshold_for([0, 0, 1, 1], [.1, .2, .8, .9]) == .8
    result = metrics(np.array([0, 0]), np.array([.1, .7]), .5, ["a", "a"])
    assert result["auroc"] is None
    assert result["nonsepsis_patients_with_any_alert"] == 1


def test_policy_prioritizes_missingness_and_rising_scores():
    assert policy(.01, .01, .5, True)[0] == 1
    assert policy(.5, .5, .5, False)[0] == 1
    assert policy(.2, .1, .5, False)[0] == 1
    assert policy(.3, .3, .5, False)[0] == 2
    assert policy(.01, .01, .5, False)[0] == 4


def test_end_to_end_replay_matches_batch_and_rejects_training_patient(tmp_path):
    import json
    import joblib
    from threadpoolctl import threadpool_limits
    from ertriage.model import train
    from ertriage.replay import replay

    root = tmp_path / "data"
    site = root / "site_A"
    site.mkdir(parents=True)
    rng = np.random.default_rng(5)
    for i in range(60):
        df = patient()
        df["Age"] = 30 + i
        df["HR"] = rng.normal(80, 10, len(df))
        if i % 2:
            df.loc[6:, "SepsisLabel"] = 1
            df.loc[6:, "HR"] += 25
        df.to_csv(site / f"p{i:06}.psv", sep="|", index=False)
    run = tmp_path / "run"
    train(root, run, limit=0)
    report = json.loads((run / "metrics.json").read_text())
    pid = pd.read_csv(run / "test_patients.csv").patient.iloc[0]
    result = replay(root, run, pid)
    bundle = joblib.load(run / f"{report['selected_model']}.joblib")
    with threadpool_limits(limits=2):
        expected = bundle["model"].predict_proba(features(read_patient(root / pid)))[:, 1]
    np.testing.assert_allclose(result.score, expected)
    logistic = report["models"]["logistic"]
    assert logistic["recalibration"]["fitted_on"] == "validation"
    assert logistic["test_recalibrated"]["alerts_identical"]
    assert set(logistic["test"]["subgroups"]) == {"site", "age_band", "gender", "unit"}
    counted = logistic["test"]["subgroups"]["age_band"]
    assert sum(level["patients"] for level in counted.values()) == logistic["test"]["patients"]
    assert "calibrated_score" in pd.read_csv(run / "test_predictions.csv").columns
    train_pid = pd.read_csv(run / "train_patients.csv").patient.iloc[0]
    with pytest.raises(ValueError, match="held-out"):
        replay(root, run, train_pid)
    with pytest.raises(ValueError, match="new output"):
        train(root, run)


def septic(n=24, onset=15):
    """Label is positive from six hours before onset, as PhysioNet 2019 provides it."""
    labels = np.zeros(n, dtype=int)
    labels[onset - 6:] = 1
    return labels


def test_utility_follows_official_ramps_and_normalization():
    from ertriage.evaluate import normalized_utility, utility_by_patient, utility_terms

    labels = septic()
    alert, silent, best = utility_terms(labels)
    assert alert[9] == pytest.approx(1.0)      # peak reward six hours before onset
    assert alert[3] == pytest.approx(0.0)      # twelve hours before onset earns nothing
    assert silent[18] == pytest.approx(-2.0)   # worst penalty for silence three hours after
    assert silent[8] == 0 and alert[19] == 0 and silent[19] == 0
    assert list(np.flatnonzero(best)) == list(range(3, 19))

    groups = [slice(0, len(labels))]
    assert normalized_utility(*utility_by_patient(labels, best, groups)) == pytest.approx(1.0)
    silence = np.zeros(len(labels), dtype=bool)
    assert normalized_utility(*utility_by_patient(labels, silence, groups)) == pytest.approx(0.0)

    healthy = np.zeros(12, dtype=int)
    always = np.ones(12, dtype=bool)
    assert normalized_utility(*utility_by_patient(healthy, always, [slice(0, 12)])) is None
    y = np.concatenate([labels, healthy])
    ids = ["a"] * len(labels) + ["b"] * 12
    from ertriage.evaluate import patient_groups
    scored = normalized_utility(*utility_by_patient(y, np.concatenate([best, always]), patient_groups(ids)))
    assert scored < 1  # false alerts on the healthy patient cost utility


def test_calibration_reports_error_and_recalibration_slope():
    from ertriage.evaluate import calibration

    rng = np.random.default_rng(0)
    truth = rng.random(4000)
    y = (rng.random(4000) < truth).astype(int)
    honest = calibration(y, truth)
    assert honest["expected_calibration_error"] < .05
    assert honest["slope"] == pytest.approx(1, abs=.15)
    assert honest["intercept"] == pytest.approx(0, abs=.15)
    assert sum(b["count"] for b in honest["bins"]) == 4000
    inflated = calibration(y, np.clip(truth * 3, 0, .999))
    assert inflated["expected_calibration_error"] > honest["expected_calibration_error"]
    assert calibration(np.zeros(10, dtype=int), np.full(10, .3))["slope"] is None


def test_site_split_holds_out_a_whole_site():
    manifest = pd.DataFrame({"patient": [f"site_A/p{i}.psv" for i in range(60)]
                                        + [f"site_B/p{i}.psv" for i in range(40)],
                             "ever_sepsis": [0, 1] * 50})
    splits = split_patients(manifest, 42, "site")
    assert set(splits["test"].patient.str.split("/").str[0]) == {"site_B"}
    assert not set(splits["test"].patient) & set(splits["train"].patient) | set(splits["test"].patient) & set(splits["validation"].patient)
    assert len(splits["train"]) + len(splits["validation"]) == 60
    assert_frame_equal(splits["train"], split_patients(manifest, 42, "site")["train"])
    with pytest.raises(ValueError, match="two sites"):
        split_patients(manifest[manifest.patient.str.startswith("site_A")], 42, "site")
    with pytest.raises(ValueError, match="split scheme"):
        split_patients(manifest, 42, "kfold")


def test_bootstrap_is_reproducible_and_resamples_whole_patients():
    from ertriage.evaluate import bootstrap

    rng = np.random.default_rng(1)
    y = np.concatenate([septic() if i % 2 else np.zeros(24, dtype=int) for i in range(40)])
    ids = np.repeat([f"p{i}" for i in range(40)], 24)
    p = np.clip(.02 + .5 * y + rng.normal(0, .05, len(y)), 0, 1)
    first = bootstrap(y, p, .3, ids, seed=42, draws=200)
    assert first == bootstrap(y, p, .3, ids, seed=42, draws=200)
    for name, interval in first["intervals"].items():
        assert interval is not None and interval["low"] <= interval["high"], name
    assert first["intervals"]["auroc"]["low"] > .9
    assert 0 < first["intervals"]["normalized_utility"]["low"] <= 1


def test_recalibration_is_fitted_on_validation_monotone_and_alert_preserving():
    from ertriage.evaluate import calibration, recalibrate, recalibrator

    rng = np.random.default_rng(3)
    truth = rng.random(6000) * .4
    y = (rng.random(6000) < truth).astype(int)
    overextended = np.clip(truth * 3, 1e-6, .999)
    validation, held_out = slice(0, 3000), slice(3000, 6000)
    fit = recalibrator(y[validation], overextended[validation])
    assert fit["fitted_on"] == "validation" and fit["slope"] > 0
    before = calibration(y[held_out], overextended[held_out])
    after = calibration(y[held_out], recalibrate(fit, overextended[held_out]))
    assert after["expected_calibration_error"] < before["expected_calibration_error"]
    assert abs(after["slope"] - 1) < abs(before["slope"] - 1)
    mapped = recalibrate(fit, .3)
    assert np.array_equal(recalibrate(fit, overextended) >= mapped, overextended >= .3)
    assert recalibrator(np.zeros(10, dtype=int), np.linspace(.1, .9, 10)) is None


def test_subgroups_partition_the_cohort_and_match_direct_metrics():
    from sklearn.metrics import roc_auc_score

    from ertriage.evaluate import subgroup_metrics

    rng = np.random.default_rng(2)
    ids = np.repeat([f"p{i}" for i in range(20)], 24)
    y = np.concatenate([septic() if i % 2 else np.zeros(24, dtype=int) for i in range(20)])
    p = np.clip(.05 + .4 * y + rng.normal(0, .1, len(y)), 0, 1)
    levels = {f"p{i}": ("even" if i % 2 == 0 else "odd") for i in range(19)}
    result = subgroup_metrics(y, p, .3, ids, levels)
    assert set(result) == {"even", "odd", "unknown"}
    assert sum(v["patients"] for v in result.values()) == 20
    assert sum(v["hours"] for v in result.values()) == len(y)
    assert result["unknown"]["patients"] == 1  # the unmapped patient is described, not dropped
    odd = np.isin(ids, [f"p{i}" for i in range(1, 19, 2)])  # p19 is unmapped, so it is not in "odd"
    assert result["odd"]["auroc"] == pytest.approx(roc_auc_score(y[odd], p[odd]))
    assert result["odd"]["alert_hours_per_100"] == pytest.approx(100 * (p[odd] >= .3).mean())
    assert result["even"]["auroc"] is None and result["even"]["average_precision"] is None


def test_patient_attributes_bracket_age_and_mark_unrecorded_fields():
    df = patient()
    df["Age"], df["Gender"], df["Unit1"], df["Unit2"] = 80.4, 0, 0, 1
    assert patient_attributes(df) == dict(age_band="age_80_plus", gender="gender_0", unit="unit2")
    df["Age"], df["Gender"], df["Unit1"], df["Unit2"] = 64.9, 1, np.nan, np.nan
    assert patient_attributes(df) == dict(age_band="age_50_64", gender="gender_1", unit="unit_unrecorded")
    df["Age"], df["Unit1"], df["Unit2"] = 49.9, 0, 0
    assert patient_attributes(df)["age_band"] == "age_lt_50"
    assert patient_attributes(df)["unit"] == "unit_other"


def test_threshold_rules_optimize_their_own_objective():
    from ertriage.evaluate import patient_groups, utility_hours, utility_of

    y = np.concatenate([septic() if i % 2 else np.zeros(24, dtype=int) for i in range(20)])
    ids = np.repeat([f"p{i}" for i in range(20)], 24)
    rng = np.random.default_rng(4)
    p = np.clip(.05 + .35 * y + rng.normal(0, .08, len(y)), 1e-6, 1 - 1e-6)

    budgeted = threshold_for(y, p, ids, rule="budget", budget=2.)
    assert (p >= budgeted).mean() <= .02
    below = np.unique(p)[np.searchsorted(np.unique(p), budgeted) - 1]
    assert (p >= below).mean() > .02  # no lower observed score stays inside the budget
    assert threshold_for(y, p, ids, rule="budget", budget=100.) == p.min()

    alert, silent, best = utility_hours(y, patient_groups(ids))
    chosen = threshold_for(y, p, ids, rule="utility")
    achieved = utility_of(p >= chosen, alert, silent, best)
    grid = np.unique(np.quantile(np.unique(p), np.linspace(0, 1, 201)))
    assert achieved >= max(utility_of(p >= t, alert, silent, best) for t in grid) - 1e-12
    assert achieved > utility_of(p >= threshold_for(y, p, ids, rule="budget", budget=.5),
                                 alert, silent, best)
    with pytest.raises(ValueError, match="threshold rule"):
        threshold_for(y, p, ids, rule="youden")
    with pytest.raises(ValueError, match="identifiers"):
        threshold_for(y, p, rule="utility")


def test_stable_selection_steps_back_only_on_indistinguishable_margins():
    y = np.concatenate([septic() if i % 2 else np.zeros(24, dtype=int) for i in range(40)])
    ids = np.repeat([f"p{i}" for i in range(40)], 24)
    rng = np.random.default_rng(6)
    logistic = np.clip(.05 + .06 * y + rng.normal(0, .05, len(y)), 0, 1)
    # Boosting leads only because of one patient, so resampled cohorts disagree about the winner.
    boosting = logistic.copy()
    one = (ids == "p1") & (y == 1)
    boosting[one] = np.clip(boosting[one] + .05, 0, 1)
    chosen, decision = select_model({"logistic": logistic, "boosting": boosting}, y, ids, seed=42, draws=200)
    assert decision["leader"] == "boosting" and chosen == "logistic"
    assert decision["stepped_back_to"] == "logistic"
    margin = decision["average_precision_margins"]["logistic"]
    assert margin["observed"] > 0 and margin["low"] <= 0 <= margin["high"]

    separated = {"logistic": np.clip(.05 + rng.normal(0, .05, len(y)), 0, 1),
                 "boosting": np.clip(.05 + .3 * y + rng.normal(0, .05, len(y)), 0, 1)}
    clear, decision = select_model(separated, y, ids, seed=42, draws=200)
    assert clear == "boosting" and decision["stepped_back_to"] is None
    # Refusing to step back records the margin that justified it.
    assert decision["average_precision_margins"]["logistic"]["low"] > 0
