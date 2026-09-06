import hashlib
import json
import urllib.request
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://physionet.org/files/challenge-2019/1.0.0"
COLUMNS = "HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets Age Gender Unit1 Unit2 HospAdmTime ICULOS".split()
VITALS = "HR O2Sat Temp SBP MAP DBP Resp".split()


def download(root, limit=2000, seed=42):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    candidates = []
    for site in "AB":
        url = f"{BASE}/training/training_set{site}/"
        with urllib.request.urlopen(url, timeout=60) as response:
            listing = response.read().decode("utf-8")
        names = sorted(set(re.findall(r'href="(p[0-9]+\.psv)"', listing)))
        if not names:
            raise ValueError(f"No patient files in {url}")
        target = root / f"site_{site}"
        target.mkdir(exist_ok=True)
        candidates.extend((url + name, target / name) for name in names)
    total = len(candidates)
    if limit and limit < total:
        indexes = sorted(np.random.default_rng(seed).choice(total, limit, replace=False))
        candidates = [candidates[i] for i in indexes]

    def fetch(item):
        url, dest = item
        if not dest.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        content = response.read()
                    temp = dest.with_suffix(".part")
                    temp.write_bytes(content)
                    read_patient(temp)
                    temp.replace(dest)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        read_patient(dest)
        return dict(patient=dest.relative_to(root).as_posix(), url=url,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())

    provenance = []
    print(f"Downloading/verifying {len(candidates)} of {total} patients", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for item in pool.map(fetch, candidates):
            provenance.append(item)
            if len(provenance) % 200 == 0:
                print(f"Verified {len(provenance)} patients", flush=True)
    record = dict(dataset_version="1.0.0", available_patients=total, seed=seed,
                  requested_limit=limit, files=provenance)
    (root / "provenance.json").write_text(json.dumps(record, indent=2))
    print(f"Saved provenance for {len(provenance)} patients", flush=True)


def read_patient(path):
    df = pd.read_csv(path, sep="|")
    if list(df.columns) != COLUMNS + ["SepsisLabel"] or df.empty:
        raise ValueError(f"Invalid schema or empty patient: {path}")
    df = df.apply(pd.to_numeric, errors="raise")
    if not df.SepsisLabel.isin([0, 1]).all():
        raise ValueError(f"Invalid labels: {path}")
    if not np.isfinite(df.ICULOS).all() or not (df.ICULOS.diff().dropna() == 1).all():
        raise ValueError(f"Expected consecutive hourly ICULOS: {path}")
    if np.isinf(df.to_numpy()).any():
        raise ValueError(f"Infinite value: {path}")
    return df


def patient_attributes(df):
    """Recorded administrative descriptors from the first hour, for subgroup description only.

    Age is bracketed; ages above 89 are already masked by the dataset. Gender is
    the dataset's 0/1 code, documented as female/male but kept unlabelled here
    because it is an administrative record, not a verified identity. Unit1 and
    Unit2 are the dataset's two ICU indicators and are absent for a whole source
    set, so "unknown" is a data-availability level, not a clinical one.
    """
    row = df.iloc[0]
    age = row.Age
    band = "unknown" if pd.isna(age) else next(
        name for edge, name in ((50, "age_lt_50"), (65, "age_50_64"), (80, "age_65_79"),
                                (np.inf, "age_80_plus")) if age < edge)
    gender = "unknown" if pd.isna(row.Gender) else f"gender_{int(row.Gender)}"
    if row.Unit1 == 1:
        unit = "unit1"
    elif row.Unit2 == 1:
        unit = "unit2"
    else:
        unit = "unit_unrecorded" if pd.isna(row.Unit1) and pd.isna(row.Unit2) else "unit_other"
    return dict(age_band=band, gender=gender, unit=unit)


def observation_ages(df):
    """Elapsed recorded hours since each vital was observed, using history only."""
    observed = df[VITALS].notna().reset_index(drop=True)
    t = np.arange(len(df))
    ages = {}
    for col in VITALS:
        last = np.maximum.accumulate(np.where(observed[col], t, -1))
        ages[col + "_age"] = np.where(last >= 0, t - last, 999)
    return pd.DataFrame(ages)


def features(df):
    """Prefix invariant: no labels, backwards fill, or future aggregates."""
    raw = df[COLUMNS].reset_index(drop=True)
    filled = raw.ffill()
    return pd.concat([filled, raw.isna().astype(float).add_suffix("_missing"),
                      filled[VITALS].diff(3).add_suffix("_change3"),
                      observation_ages(df)], axis=1).astype(np.float32)
