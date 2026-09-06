import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ertriage.evaluate import recalibrate
from ertriage.model import threshold_for
from ertriage.replay import review_schedule
from ertriage.workload import patient_workload, summarize_workload, workload


def test_budget_respects_top_ties_including_constant_and_endpoint_scores():
    for scores in ([.2] * 100, [1.] * 100, [0.] * 100, [.1] * 95 + [.9] * 5):
        y = [0, 1] * 50
        threshold = threshold_for(y, scores, rule="budget", budget=2)
        assert np.isfinite(threshold)
        assert not (np.asarray(scores) >= threshold).any()
        fit = dict(intercept=0., slope=1.)
        assert not (recalibrate(fit, scores) >= recalibrate(fit, threshold)).any()
        assert json.loads(json.dumps(threshold)) == threshold
    for budget in (0, -1, 101, np.nan, np.inf):
        with pytest.raises(ValueError, match="budget"):
            threshold_for([0, 1], [.1, .2], rule="budget", budget=budget)


def test_scheduler_is_prefix_causal_and_interrupts_for_new_information():
    scores = [.01, .01, .01, .3, .3, .8, .01, .01]
    stale = [False, False, True, False, False, False, False, False]
    result = review_schedule(scores, stale, .5)
    assert list(np.flatnonzero(result.review_now)) == [0, 2, 3, 4, 5, 6]
    # Row 4's score rise causes immediate review; row 5 retains its already-due review.
    for stop in range(1, len(scores) + 1):
        assert_frame_equal(review_schedule(scores[:stop], stale[:stop], .5), result.iloc[:stop])
    fresh = patient_workload([.01] * 9, [False] * 9, .5)
    assert [fresh[k] for k in ("adaptive", "fixed_1h", "fixed_2h", "fixed_4h")] == [3, 9, 5, 3]
    missing = patient_workload([.01] * 9, [True] * 9, .5)
    assert missing["adaptive"] == missing["stale_reviews"] == 9
    assert missing["all_hours_reviewed"]


def test_workload_bootstrap_weights_hours_and_pairs_comparisons():
    patients = pd.DataFrame([patient_workload([.01], [False], .5),
                             patient_workload([.01] * 9, [False] * 9, .5)])
    summary = summarize_workload(patients, draws=100)
    assert summary == summarize_workload(patients, draws=100)
    assert summary["policies"]["adaptive"]["reviews_per_100_hours"] == 40
    assert summary["policies"]["fixed_1h"]["interval"] == dict(low=100, high=100)
    assert summary["policies"]["fixed_4h"]["adaptive_minus_fixed_per_100_hours"] == dict(
        observed=0, interval=dict(low=0, high=0))
    with pytest.raises(ValueError, match="positive"):
        summarize_workload(patients, draws=0)


def test_workload_verifies_artifact_alignment_and_preserves_outputs(tmp_path):
    import hashlib
    from ertriage.data import COLUMNS

    root, run = tmp_path / "data", tmp_path / "run"
    (root / "site_A").mkdir(parents=True)
    run.mkdir()
    pid = "site_A/p1.psv"
    frame = pd.DataFrame(np.nan, index=range(4), columns=COLUMNS)
    frame["ICULOS"] = [10, 11, 12, 13]
    frame["SepsisLabel"] = [0, 0, 1, 1]
    frame.to_csv(root / pid, sep="|", index=False)
    digest = hashlib.sha256((root / pid).read_bytes()).hexdigest()
    pd.DataFrame([dict(patient=pid, hours=4, sha256=digest)]).to_csv(run / "test_patients.csv", index=False)
    for name in ("train", "validation"):
        pd.DataFrame(dict(patient=[f"site_A/{name}.psv"])).to_csv(run / f"{name}_patients.csv", index=False)
    predictions = pd.DataFrame(dict(patient=[pid] * 4, hour=[10, 11, 12, 13],
                                    label=[0, 0, 1, 1], score=[.01] * 4))
    predictions.to_csv(run / "test_predictions.csv", index=False)
    (run / "metrics.json").write_text(json.dumps(dict(selected_model="boosting",
        models=dict(boosting=dict(validation=dict(threshold=.5))))))
    report = workload(root, run, tmp_path / "audit", draws=10)
    assert report["hours"] == report["policies"]["adaptive"]["reviews"] == 4
    assert (tmp_path / "audit" / "WORKLOAD.md").exists()
    with pytest.raises(ValueError, match="new output"):
        workload(root, run, tmp_path / "audit", draws=10)
    predictions.loc[0, "hour"] = 11
    predictions.to_csv(run / "test_predictions.csv", index=False)
    with pytest.raises(ValueError, match="hour order"):
        workload(root, run, tmp_path / "bad", draws=10)
    assert not (tmp_path / "bad").exists()
    predictions.drop(columns="hour").to_csv(run / "test_predictions.csv", index=False)
    assert workload(root, run, tmp_path / "legacy", draws=10)["hours"] == 4
    (root / pid).write_text((root / pid).read_text() + "\n")
    with pytest.raises(ValueError, match="differs from the training manifest"):
        workload(root, run, tmp_path / "changed", draws=10)
