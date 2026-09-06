import numpy as np
import pandas as pd
import pytest

from ertriage.data import COLUMNS, features
from ertriage.early_warning import warning_patient
from ertriage.model import fitted, target_view
from ertriage.target import ONSET_SHIFT, event_target, onset_row


def labels_with(first_positive, n=40):
    y = np.zeros(n, dtype=int)
    if first_positive is not None:
        y[first_positive:] = 1
    return y


def frame(first_positive, n=40, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in COLUMNS}
    data["ICULOS"] = np.arange(1, n + 1)
    df = pd.DataFrame(data)
    df.loc[df.index % 3 == 0, "HR"] = np.nan  # gaps, so ffill and age features do real work
    df["SepsisLabel"] = labels_with(first_positive, n)
    return df


def test_horizon_six_event_label_equals_the_persistent_label_before_onset():
    """The identity the basis comparison relies on: at horizon 6 only masking differs."""
    for first in (1, 5, 14, 25):
        persistent = labels_with(first)
        y, keep, status = event_target(persistent, horizon=6)
        assert status == "eligible"
        np.testing.assert_array_equal(y[keep], persistent[keep])
        # ... and only at horizon 6.
        assert not np.array_equal(event_target(persistent, horizon=12)[0][keep], persistent[keep])


def test_event_target_covers_the_window_the_early_warning_metric_scores():
    """At horizon 12 every hour in [onset-12, onset-6] is positive; the persistent label calls
    all but the last of them negative, which is the mismatch this target exists to remove."""
    persistent = labels_with(14)
    onset = 14 + ONSET_SHIFT
    window = np.arange(onset - 12, onset - 6 + 1)
    assert warning_patient(np.arange(1, 41), persistent, np.zeros(40), .5)["onset_proxy_hour"] == onset + 1
    assert event_target(persistent, horizon=12)[0][window].all()
    assert persistent[window].sum() == 1 and window.size == 7  # only the boundary hour


def test_post_onset_hours_are_dropped_and_unrecoverable_records_excluded():
    y, keep, status = event_target(labels_with(14), horizon=12)
    onset = 14 + ONSET_SHIFT
    assert status == "eligible"
    assert keep.sum() == onset and not keep[onset:].any()
    assert y[keep].sum() == 12 and not y[onset:].any()
    nonpersistent = np.r_[np.zeros(5, int), np.ones(3, int), np.zeros(4, int)]
    for seq, expected in ((labels_with(0), "positive_at_record_start"), (nonpersistent, "nonpersistent_label")):
        y, keep, got = event_target(seq, horizon=12)
        assert (y, keep, got) == (None, None, expected)
    assert onset_row(labels_with(None))[1] == "no_positive_label"
    assert onset_row(labels_with(36, n=40))[1] == "onset_beyond_record"  # kept: all hours are pre-onset


def test_onset_beyond_record_keeps_every_hour_as_pre_onset():
    y, keep, status = event_target(labels_with(36, n=40), horizon=12)
    assert status == "onset_beyond_record" and keep.all()
    assert y.sum() == 10 and y[30:].all()  # onset row 42; hours 30..39 are within 12


@pytest.mark.parametrize("horizon", [0, -3, 2.5])
def test_horizon_must_be_a_positive_whole_number(horizon):
    with pytest.raises(ValueError, match="Horizon"):
        event_target(labels_with(14), horizon=horizon)


def test_masking_after_the_fact_equals_truncating_the_record_first():
    """Features are causal, so dropping post-onset rows late cannot change earlier rows.

    The whole design rests on this: if it failed, post-onset physiology could
    leak backwards into hours the model is allowed to fit.
    """
    df = frame(14)
    _, _, _, _, use, _, _ = target_view([df], ["p"], "event", 12)
    late = features(df).iloc[use].reset_index(drop=True)
    early = features(df.iloc[:len(use)]).reset_index(drop=True)
    pd.testing.assert_frame_equal(late, early)


def test_target_view_keeps_every_hour_for_scoring_but_fits_only_pre_onset():
    dfs = [frame(14), frame(None, seed=1), frame(0, seed=2)]
    x, y, ids, hours, use, target, counts = target_view(dfs, ["a", "b", "c"], "event", 12)
    assert len(x) == len(y) == len(ids) == len(hours) == 120  # every recorded hour is still scored
    assert counts == {"eligible": 1, "no_positive_label": 1, "positive_at_record_start": 1}
    # 'a' contributes its 20 pre-onset hours, 'b' all 40, and 'c' none.
    assert sorted(set(ids[use])) == ["a", "b"] and len(use) == 20 + 40
    fx, fy, fi = fitted((x, y, ids, hours, use, target, counts))
    assert len(fx) == len(fy) == len(fi) == 60 and fy.sum() == 12


def test_persistent_target_view_is_unchanged():
    df = frame(14)
    x, y, ids, hours, use, target, counts = target_view([df], ["p"], "persistent", 12)
    np.testing.assert_array_equal(use, np.arange(len(df)))
    np.testing.assert_array_equal(target, df.SepsisLabel.to_numpy())
    np.testing.assert_array_equal(y, df.SepsisLabel.to_numpy())
    assert counts == {}
    with pytest.raises(ValueError, match="Unknown target"):
        target_view([df], ["p"], "nonsense", 12)
