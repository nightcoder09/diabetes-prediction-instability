"""Rebuild the 21-predictor diabetes dataset from raw CDC BRFSS annual files (SAS transport, .XPT).

The derivation follows the recoding used to create the published 2015 extract (UCI id 891 /
Kaggle release): select the source items, drop respondents with any missing / "don't know" /
refused answer, and recode to the binary / ordinal codings documented in `data.DESCRIPTIONS`.
It is validated by reproducing the published 2015 extract exactly from the raw 2015 file
(scripts/14_build_external.py); the same rules are then applied to later survey years.

Year-specific item names (CDC renames items when questions change) are resolved from candidate
lists, so the exact source variable used for each year is recorded in the build report.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET

CDC_XPT_URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"

# target feature -> ordered candidate source names (first present in the file is used)
CANDIDATES = {
    "diabetes": ["DIABETE3", "DIABETE4"],
    "HighBP": ["_RFHYPE5", "_RFHYPE6"],
    "HighChol": ["TOLDHI2", "TOLDHI3"],
    "CholCheck": ["_CHOLCHK", "_CHOLCH2", "_CHOLCH3"],
    "BMI": ["_BMI5"],
    "Smoker": ["SMOKE100"],
    "Stroke": ["CVDSTRK3"],
    "HeartDiseaseorAttack": ["_MICHD"],
    "PhysActivity": ["_TOTINDA"],
    "Fruits": ["_FRTLT1", "_FRTLT1A"],
    "Veggies": ["_VEGLT1", "_VEGLT1A"],
    "HvyAlcoholConsump": ["_RFDRHV5", "_RFDRHV6", "_RFDRHV7", "_RFDRHV8", "_RFDRHV9"],
    "AnyHealthcare": ["HLTHPLN1", "_HLTHPLN", "_HLTHPL1", "_HLTHPL2"],
    "NoDocbcCost": ["MEDCOST", "MEDCOST1"],
    "GenHlth": ["GENHLTH"],
    "MentHlth": ["MENTHLTH"],
    "PhysHlth": ["PHYSHLTH"],
    "DiffWalk": ["DIFFWALK"],
    "Sex": ["SEX", "_SEX", "SEXVAR"],
    "Age": ["_AGEG5YR"],
    "Education": ["EDUCA"],
    "Income": ["INCOME2", "INCOME3"],
}

# Items that are not asked in every survey year (e.g. fruit/vegetable intake is absent from the 2023
# questionnaire). When absent they are left missing and imputed downstream with development medians.
OPTIONAL = {"Fruits", "Veggies"}

# Value codes shared by BRFSS yes/no items: 1 = yes, 2 = no, 7 = don't know, 9 = refused.
YES_NO = {1: 1, 2: 0}
# BRFSS calculated risk-factor flags: 1 = no (risk absent), 2 = yes (risk present), 9 = missing.
RISK_FLAG = {1: 0, 2: 1}


def _map(s: pd.Series, mapping: dict) -> pd.Series:
    return s.map(mapping)  # anything not in the mapping becomes NaN (= dropped later)


def _income(s: pd.Series, source: str) -> pd.Series:
    if source == "INCOME3":
        # 2021+: 1-7 identical to INCOME2; 8 = $75-100k, 9 = $100-150k, 10 = $150-200k, 11 = $200k+
        # -> collapse 8-11 into the 2015 top category (>= $75k, nominal dollars)
        return s.where(s.between(1, 11)).clip(upper=8)
    return s.where(s.between(1, 8))


def recode(raw: pd.DataFrame, sources: dict[str, str | None]) -> pd.DataFrame:
    """Apply the published recoding to one survey year. ``sources`` maps target -> source column
    (None for an optional item the year does not contain)."""
    g = {k: (raw[v] if v is not None else pd.Series(np.nan, index=raw.index)) for k, v in sources.items()}
    out = pd.DataFrame(index=raw.index)
    d = g["diabetes"]
    # 1 = diabetes, 2 = only during pregnancy (-> no), 3 = no, 4 = prediabetes/borderline; 7/9 dropped
    out["Diabetes_012"] = d.map({1: 2, 2: 0, 3: 0, 4: 1})
    out[TARGET] = (out["Diabetes_012"] == 2).astype(float).where(out["Diabetes_012"].notna())
    out["HighBP"] = _map(g["HighBP"], RISK_FLAG)
    out["HighChol"] = _map(g["HighChol"], YES_NO)
    out["CholCheck"] = g["CholCheck"].map({1: 1, 2: 0, 3: 0})
    out["BMI"] = (g["BMI"] / 100).round(0)
    out["Smoker"] = _map(g["Smoker"], YES_NO)
    out["Stroke"] = _map(g["Stroke"], YES_NO)
    out["HeartDiseaseorAttack"] = _map(g["HeartDiseaseorAttack"], YES_NO)
    out["PhysActivity"] = _map(g["PhysActivity"], YES_NO)
    out["Fruits"] = _map(g["Fruits"], YES_NO)
    out["Veggies"] = _map(g["Veggies"], YES_NO)
    out["HvyAlcoholConsump"] = _map(g["HvyAlcoholConsump"], RISK_FLAG)
    out["AnyHealthcare"] = _map(g["AnyHealthcare"], YES_NO)
    out["NoDocbcCost"] = _map(g["NoDocbcCost"], YES_NO)
    out["GenHlth"] = g["GenHlth"].where(g["GenHlth"].between(1, 5))
    for f in ("MentHlth", "PhysHlth"):
        s = g[f].replace(88, 0)
        out[f] = s.where(s.between(0, 30))
    out["DiffWalk"] = _map(g["DiffWalk"], YES_NO)
    out["Sex"] = g["Sex"].map({1: 1, 2: 0})          # 1 = male, 0 = female
    out["Age"] = g["Age"].where(g["Age"].between(1, 13))
    out["Education"] = g["Education"].where(g["Education"].between(1, 6))
    out["Income"] = _income(g["Income"], sources["Income"])
    return out[[TARGET, "Diabetes_012"] + FEATURES]


def resolve_sources(columns) -> dict[str, str | None]:
    cols = set(columns)
    out, missing = {}, []
    for k, cands in CANDIDATES.items():
        hit = next((c for c in cands if c in cols), None)
        if hit is None and k in OPTIONAL:
            out[k] = None
        elif hit is None:
            missing.append(f"{k} (tried {cands})")
        else:
            out[k] = hit
    if missing:
        raise KeyError("Source items not found: " + "; ".join(missing))
    return out


def unzip_xpt(path: Path, workdir: Path | None = None) -> Path:
    """Extract the .XPT member of a CDC zip (member names can carry trailing spaces)."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path
    workdir = Path(workdir) if workdir else path.parent
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.strip().upper().endswith(".XPT"))
        target = workdir / Path(member.strip()).name
        if not target.exists():
            with z.open(member) as src, open(target, "wb") as dst:
                while True:
                    buf = src.read(1 << 24)
                    if not buf:
                        break
                    dst.write(buf)
    return target


def read_xpt(path: Path, columns: list[str] | None = None, chunksize: int = 50_000) -> pd.DataFrame:
    """Read a SAS transport file in chunks, keeping only ``columns`` to limit memory."""
    path = Path(path)
    parts = []
    reader = pd.read_sas(path, format="xport", chunksize=chunksize, encoding="latin-1")
    for chunk in reader:
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        parts.append(chunk if columns is None else chunk[[c for c in columns if c in chunk.columns]])
    return pd.concat(parts, ignore_index=True)


def xpt_columns(path: Path) -> list[str]:
    reader = pd.read_sas(path, format="xport", chunksize=10, encoding="latin-1")
    first = next(iter(reader))
    return [c.strip().upper() for c in first.columns]


def build_year(xpt_path: Path, year: int, workdir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Full derivation for one year. Returns (clean integer frame, build report)."""
    xpt_path = unzip_xpt(xpt_path, workdir)
    cols = xpt_columns(xpt_path)
    sources = resolve_sources(cols)
    extra = [c for c in ("_STATE", "IMONTH", "IYEAR", "_LLCPWT") if c in cols]
    raw = read_xpt(xpt_path, [v for v in sources.values() if v] + extra)
    rec = recode(raw, sources)
    unavailable = [k for k, v in sources.items() if v is None]
    report = {"year": year, "raw_rows": int(len(raw)), "sources": sources, "unavailable_items": unavailable,
              "missing_by_item": {k: int(v) for k, v in rec.isna().sum().items()}}
    keep = rec.drop(columns=unavailable).notna().all(axis=1)
    clean = rec[keep].reset_index(drop=True)
    avail = [c for c in clean.columns if c not in unavailable]
    clean[avail] = clean[avail].astype(np.int64)
    if extra:
        clean = pd.concat([clean, raw.loc[keep, extra].reset_index(drop=True)], axis=1)
    report.update({"clean_rows": int(len(clean)), "prevalence": float(clean[TARGET].mean()),
                   "prediabetes": int((clean["Diabetes_012"] == 1).sum())})
    return clean, report
