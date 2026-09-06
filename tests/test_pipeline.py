import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ertriage.data import COLUMNS, features, read_patient
from ertriage.model import split_patients, threshold_for, metrics
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
    train_pid = pd.read_csv(run / "train_patients.csv").patient.iloc[0]
    with pytest.raises(ValueError, match="held-out"):
        replay(root, run, train_pid)
    with pytest.raises(ValueError, match="new output"):
        train(root, run)
