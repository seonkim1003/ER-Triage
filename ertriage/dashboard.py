"""Loopback-only, read-only research viewer. No third-party web dependencies."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .data import VITALS, observation_ages
from .early_warning import warning_patient
from .history import HeldoutRun
from .replay import review_schedule


def patient_payload(cohort, pid):
    frame, saved = cohort.patient(pid)
    ages = observation_ages(frame)
    schedule = review_schedule(saved.score, (ages >= 4).any(axis=1), cohort.threshold)
    rows = pd.concat([frame[["ICULOS"] + VITALS].rename(columns={"ICULOS": "hour"}),
                      ages, saved[["score"]], schedule], axis=1)
    rows["alert"] = saved.score >= cohort.threshold
    rows["retrospective_label"] = frame.SepsisLabel
    index = cohort.patients.index(pid)
    return dict(patient=pid, threshold=cohort.threshold, rows=json.loads(rows.to_json(orient="records", double_precision=15)),
                timing=warning_patient(frame.ICULOS, frame.SepsisLabel, saved.score, cohort.threshold),
                previous=cohort.patients[index - 1] if index else None,
                next=cohort.patients[index + 1] if index + 1 < len(cohort.patients) else None)


def make_server(cohort, port=8765, evaluation=None):
    summary = None
    if evaluation:
        summary = json.loads(Path(evaluation).read_text(encoding="utf-8"))
        if summary.get("source_sha256") != cohort.hashes:
            raise ValueError("Evaluation report does not match this run's source artifacts")
    assets = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Do not let a foreign web origin read the local research API.
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self.send_error(403)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                self.send_error(403)
                return
            request = urlparse(self.path)
            query = parse_qs(request.query)
            try:
                if request.path == "/api/meta":
                    payload = dict(run=cohort.run.name, model=cohort.selected, patients=len(cohort.patients),
                                   first_patient=cohort.patients[0], evaluation=summary)
                elif request.path == "/api/patients":
                    search = query.get("q", [""])[0].lower()
                    matches = [p for p in cohort.patients if search in p.lower()]
                    payload = dict(patients=matches[:50], matches=len(matches))
                elif request.path == "/api/patient":
                    payload = patient_payload(cohort, query.get("id", [""])[0])
                elif request.path in ("/", "/app.js", "/style.css"):
                    name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[request.path]
                    mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                    self.respond((assets / name).read_bytes(), mime)
                    return
                else:
                    self.send_error(404)
                    return
                self.respond(json.dumps(payload, allow_nan=False).encode(), "application/json")
            except (ValueError, FileNotFoundError) as exc:
                self.respond(json.dumps(dict(error=str(exc))).encode(), "application/json", 400)

        def respond(self, data, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def dashboard(root, run, port=8765, evaluation=None):
    cohort = HeldoutRun(root, run)
    server = make_server(cohort, port, evaluation)
    print(f"Research replay: http://127.0.0.1:{server.server_port}", flush=True)
    print("Local historical ICU viewer. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
