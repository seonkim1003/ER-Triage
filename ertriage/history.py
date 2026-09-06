"""Read and verify frozen held-out histories without loading executable model files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_patient


class HeldoutRun:
    def __init__(self, root, run):
        self.root, self.run = Path(root).resolve(), Path(run).resolve()
        names = ("metrics.json", "test_predictions.csv", "test_patients.csv",
                 "train_patients.csv", "validation_patients.csv")
        self.hashes = {n: hashlib.sha256((self.run / n).read_bytes()).hexdigest() for n in names}
        self.report = json.loads((self.run / "metrics.json").read_text())
        self.manifest = pd.read_csv(self.run / "test_patients.csv")
        m = self.manifest
        if not {"patient", "hours", "sha256"}.issubset(m) or m.empty:
            raise ValueError("Invalid held-out manifest")
        if m.patient.isna().any() or m.patient.duplicated().any():
            raise ValueError("Invalid or duplicate held-out patient identifiers")
        self.patients = sorted(m.patient.tolist())
        self.records = m.set_index("patient")
        for split in ("train", "validation"):
            if set(self.patients) & set(pd.read_csv(self.run / f"{split}_patients.csv").patient):
                raise ValueError("Held-out patients overlap development patients")
        self.predictions = pd.read_csv(self.run / "test_predictions.csv")
        p = self.predictions
        if not {"patient", "label", "score"}.issubset(p):
            raise ValueError("Predictions need patient, label and score columns")
        if p.patient.isna().any() or set(p.patient) != set(self.patients):
            raise ValueError("Predictions must contain exactly the held-out patients")
        if not np.isfinite(p.score).all() or not p.score.between(0, 1).all():
            raise ValueError("Prediction scores must be finite and within [0, 1]")
        self.groups = p.groupby("patient", sort=False)
        self.selected = self.report["selected_model"]
        self.threshold = float(self.report["models"][self.selected]["validation"]["threshold"])
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Invalid frozen threshold")

    def patient(self, pid):
        if pid not in self.records.index:
            raise ValueError("Patient must be in the held-out test manifest")
        path = (self.root / pid).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Patient path must stay inside the data directory")
        record = self.records.loc[pid]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"Patient file differs from the training manifest: {pid}")
        frame, saved = read_patient(path), self.groups.get_group(pid).reset_index(drop=True)
        if len(frame) != record.hours or len(saved) != record.hours:
            raise ValueError(f"Prediction hour count differs from manifest: {pid}")
        if not np.array_equal(saved.label, frame.SepsisLabel):
            raise ValueError(f"Prediction label order differs from source: {pid}")
        if "hour" in saved and not np.array_equal(saved.hour, frame.ICULOS):
            raise ValueError(f"Prediction hour order differs from source: {pid}")
        return frame, saved

    def provenance(self):
        return dict(source_run=str(self.run), source_sha256=self.hashes,
                    selected_model=self.selected, threshold=self.threshold,
                    alignment="Patient hashes, split separation, row counts and ordered labels verified. "
                              "Explicit hour keys checked when present; legacy within-patient row order "
                              "is assumed intact.")
