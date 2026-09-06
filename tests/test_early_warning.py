import hashlib
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from ertriage.data import COLUMNS
from ertriage.dashboard import make_server, patient_payload
from ertriage.early_warning import early_warning, summarize_warning, warning_patient
from ertriage.history import HeldoutRun


def record(alert_indices=(), first_positive=14, n=30):
    labels, scores = np.zeros(n, dtype=int), np.zeros(n)
    if first_positive is not None:
        labels[first_positive:] = 1
    scores[list(alert_indices)] = .8
    return np.arange(1, n+1), labels, scores


def test_timing_boundaries_and_contiguous_alert_episodes():
    # Transition at hour 15 -> onset proxy hour 21 -> early window [9, 15].
    result = warning_patient(*record([0, 1, 8, 9, 14, 20]), .5)
    assert result['timing_eligible'] and result['onset_proxy_hour'] == 21
    assert result['early_warning'] and result['early_warning_lead_hours'] == 12
    assert result['first_alert_lead_hours'] == 20  # recorded but not the primary warning-window lead
    assert result['alert_episodes'] == 4 and result['repeated_episodes'] == 3
    assert warning_patient(*record([14]), .5)['early_warning']  # -6 inclusive
    assert not warning_patient(*record([15]), .5)['early_warning']
    assert warning_patient(*record([19]), .5)['detected_before_onset']
    assert not warning_patient(*record([20]), .5)['detected_before_onset']  # onset itself excluded
    assert not warning_patient(*record([0]), .5)['detected_before_onset']  # far too early
    assert warning_patient(*record(range(7, 10)), .5)['early_warning']  # ongoing episode overlaps window


@pytest.mark.parametrize('first,n,status', [(0,30,'positive_at_record_start'),
    (27,30,'onset_beyond_record'), (3,30,'incomplete_warning_window'), (None,30,'no_positive_label')])
def test_ambiguous_and_incomplete_timing_is_excluded(first, n, status):
    result = warning_patient(*record([1], first, n), .5)
    assert result['timing_status'] == status and not result['timing_eligible']
    if first == 0:
        assert result['onset_proxy_hour'] is None


def test_nonpersistent_labels_and_invalid_inputs():
    hours, labels, scores = record()
    labels[-1] = 0
    assert warning_patient(hours, labels, scores, .5)['timing_status'] == 'nonpersistent_label'
    hours[-1] += 1
    with pytest.raises(ValueError, match='consecutive'):
        warning_patient(hours, labels, scores, .5)


def test_summary_denominators_exclude_unrecoverable_onsets_and_keep_misses():
    patients = pd.DataFrame([warning_patient(*record([8]), .5), warning_patient(*record(), .5),
        warning_patient(*record([0], 0), .5), warning_patient(*record([2, 4], None), .5)])
    report = summarize_warning(patients, draws=100)
    assert report == summarize_warning(patients, draws=100)
    assert report['positive_patients'] == 3 and report['timing_eligible_patients'] == 2
    assert report['measures']['early_warning_fraction'] == .5
    assert report['missed_before_onset_patients'] == report['missed_early_window_patients'] == 1
    assert report['nonsepsis_patients_alerted'] == 1
    assert report['early_warning_lead_hours_among_detected']['count'] == 1
    empty_timing = summarize_warning(patients.iloc[[-1]], draws=1)
    assert empty_timing['measures']['early_warning_fraction'] is None
    assert empty_timing['bootstrap']['intervals']['early_warning_fraction'] is None
    json.dumps(empty_timing, allow_nan=False)


@pytest.fixture
def saved_run(tmp_path):
    root, run = tmp_path/'data', tmp_path/'run'
    (root/'site_A').mkdir(parents=True); run.mkdir()
    manifests, predictions = [], []
    for i, onset in enumerate([14, None]):
        pid = f'site_A/p{i}.psv'
        hours, labels, scores = record([8, 9, 17], onset)
        frame = pd.DataFrame(np.nan, index=range(len(hours)), columns=COLUMNS)
        frame['ICULOS'], frame['SepsisLabel'] = hours, labels
        frame['HR'] = np.arange(len(hours)) + 70
        frame.to_csv(root/pid, sep='|', index=False)
        manifests.append(dict(patient=pid, hours=len(hours), sha256=hashlib.sha256((root/pid).read_bytes()).hexdigest()))
        predictions.append(pd.DataFrame(dict(patient=pid, hour=hours, label=labels, score=scores)))
    pd.DataFrame(manifests).to_csv(run/'test_patients.csv', index=False)
    pd.concat(predictions).to_csv(run/'test_predictions.csv', index=False)
    for split in ('train','validation'):
        pd.DataFrame(dict(patient=[f'site_A/{split}.psv'])).to_csv(run/f'{split}_patients.csv', index=False)
    (run/'metrics.json').write_text(json.dumps({'selected_model':'boosting','models':{'boosting':{'validation':{'threshold':.5}}}}))
    return root, run


def test_end_to_end_warning_report_and_viewer_integrity(saved_run, tmp_path):
    root, run = saved_run
    report = early_warning(root, run, tmp_path/'evaluation', draws=10)
    assert report['early_warned_patients'] == report['timing_eligible_patients'] == 1
    assert (tmp_path/'evaluation'/'EARLY_WARNING.md').exists()
    with pytest.raises(ValueError, match='new output'):
        early_warning(root, run, tmp_path/'evaluation', draws=10)
    cohort = HeldoutRun(root, run)
    payload = patient_payload(cohort, cohort.patients[0])
    assert payload['rows'][0]['Temp'] is None
    assert payload['rows'][8]['alert'] and payload['rows'][8]['review_now']
    assert payload['timing']['onset_proxy_hour'] == 21
    with pytest.raises(ValueError, match='held-out'):
        patient_payload(cohort, '../outside.psv')
    prediction = pd.read_csv(run/'test_predictions.csv')
    prediction.loc[0,'hour'] = 2
    prediction.to_csv(run/'test_predictions.csv', index=False)
    with pytest.raises(ValueError, match='hour order'):
        HeldoutRun(root, run).patient(cohort.patients[0])
    with pytest.raises(ValueError, match='does not match'):
        make_server(HeldoutRun(root, run), 0, tmp_path/'evaluation'/'early_warning.json')


def test_dashboard_http_routes_local_origin_and_errors(saved_run):
    server = make_server(HeldoutRun(*saved_run), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/meta') as response:
            assert json.load(response)['patients'] == 2
        with urlopen(base+'/') as response:
            assert b'History prefix' in response.read()
            assert "default-src 'self'" in response.headers['Content-Security-Policy']
        with pytest.raises(HTTPError) as caught:
            urlopen(base+'/api/patient?id=missing')
        assert caught.value.code == 400
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base+'/api/meta', headers={'Origin':'https://example.com'}))
        assert caught.value.code == 403
        with pytest.raises(HTTPError) as caught:
            urlopen(base+'/../README.md')
        assert caught.value.code == 404
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
