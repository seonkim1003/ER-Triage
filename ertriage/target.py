"""Event-anchored prediction targets.

The PhysioNet label is *persistent*: once it turns 1 it stays 1 for the rest of
the stay. Training and scoring on it silently mixes two different questions,
"will this patient deteriorate?" and "is this patient already deteriorating?",
and the second one is much easier. Because most positive hours in a septic stay
lie at or after onset, a model can score well on the persistent label largely by
recognizing physiology that has already declared itself.

This module defines the harder question on its own terms. The onset proxy is the
same quantity `early_warning` uses, imported from one place so the two cannot
drift: the first observed 0-to-1 transition plus six hours. An hour is positive
when the onset proxy falls inside the next `horizon` hours, and every hour at or
after the onset proxy is removed from the cohort entirely, so no post-onset hour
can be learned from or scored.

This is a reframing of an existing label, not a new clinical outcome. It does not
observe sepsis, and nothing here makes the label itself more valid.
"""
import numpy as np

HORIZON = 6
ONSET_SHIFT = 6  # PhysioNet sets the label six hours before its estimated onset.

#: Statuses whose onset proxy cannot be recovered, so the patient is unusable here.
UNUSABLE = ("nonpersistent_label", "positive_at_record_start")

METHOD = ("Event-anchored target. Onset proxy = first observed 0-to-1 label transition + 6, the same "
          "proxy used for warning timing. An hour is positive when the onset proxy falls within the "
          "next `horizon` hours, that is when 0 < onset - t <= horizon. Every hour at or after the "
          "onset proxy is dropped from training, validation and test, so no post-onset hour is ever "
          "fitted or scored. Patients whose label is nonpersistent, or already positive at the first "
          "recorded hour, are excluded because their transition cannot be recovered. Patients whose "
          "onset proxy falls beyond the recorded stay are kept: every one of their recorded hours is "
          "genuinely pre-onset. Never-positive patients keep every hour as a negative. The official "
          "PhysioNet utility is defined against the persistent label and its timing, so it is not "
          "reported for this target.")


def onset_row(labels):
    """Row index of the onset proxy and a status, or (None, status) when unrecoverable.

    Mirrors `early_warning.warning_patient` exactly. Recorded hours are
    consecutive, so a row offset and an hour offset are the same thing.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Event target needs a nonempty binary label sequence")
    positive = np.flatnonzero(labels)
    if not len(positive):
        return None, "no_positive_label"
    if (np.diff(labels.astype(int)) < 0).any():
        return None, "nonpersistent_label"
    first = int(positive[0])
    if first == 0:
        return None, "positive_at_record_start"
    row = first + ONSET_SHIFT
    return row, ("eligible" if row < len(labels) else "onset_beyond_record")


def event_target(labels, horizon=HORIZON):
    """Event-anchored labels and a pre-onset mask for one patient.

    Returns `(y, keep, status)`, or `(None, None, status)` when the patient
    cannot be used. `keep` selects the hours strictly before the onset proxy;
    `y` is aligned to the full record and is positive only inside `keep`.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Horizon must be a positive whole number of hours")
    row, status = onset_row(labels)
    if status in UNUSABLE:
        return None, None, status
    n = len(np.asarray(labels))
    if row is None:
        return np.zeros(n, dtype=int), np.ones(n, dtype=bool), status
    lead = row - np.arange(n)
    # Positive strictly before onset and within the horizon; the mask drops the rest.
    return ((lead > 0) & (lead <= horizon)).astype(int), np.arange(n) < row, status


def event_frame(df, horizon=HORIZON):
    """Event-anchored labels and mask for a patient frame read by `read_patient`."""
    return event_target(df.SepsisLabel.to_numpy(dtype=int), horizon)
