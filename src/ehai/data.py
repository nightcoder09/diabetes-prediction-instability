"""Dataset acquisition, schema validation, canonicalisation, and reproducible splits.

Dataset: CDC Diabetes Health Indicators (BRFSS 2015), UCI Machine Learning Repository id 891,
DOI 10.24432/C53919. One row per survey respondent; 21 self-reported predictors.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import derive_seed, resolve
from .utils import get_logger, load_json, save_json

TARGET = "Diabetes_binary"

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL = ["GenHlth", "Age", "Education", "Income"]
CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]

# Valid integer domains (inclusive) used for validation, perturbation clipping and inference checks.
DOMAINS = {f: (0, 1) for f in BINARY}
DOMAINS.update({"BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30), "PhysHlth": (0, 30),
                "Age": (1, 13), "Education": (1, 6), "Income": (1, 8)})

DESCRIPTIONS = {
    "HighBP": "Ever told blood pressure is high (0 = no, 1 = yes)",
    "HighChol": "Ever told cholesterol is high (0 = no, 1 = yes)",
    "CholCheck": "Cholesterol check within past 5 years (0 = no, 1 = yes)",
    "BMI": "Body-mass index, kg/m^2 (integer, derived from self-reported height and weight)",
    "Smoker": "Smoked at least 100 cigarettes in lifetime (0 = no, 1 = yes)",
    "Stroke": "Ever told had a stroke (0 = no, 1 = yes)",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction (0 = no, 1 = yes)",
    "PhysActivity": "Physical activity in past 30 days, excluding job (0 = no, 1 = yes)",
    "Fruits": "Consumes fruit >= 1 time per day (0 = no, 1 = yes)",
    "Veggies": "Consumes vegetables >= 1 time per day (0 = no, 1 = yes)",
    "HvyAlcoholConsump": "Heavy drinking: men > 14, women > 7 drinks/week (0 = no, 1 = yes)",
    "AnyHealthcare": "Has any health-care coverage (0 = no, 1 = yes)",
    "NoDocbcCost": "Could not see a doctor in past year because of cost (0 = no, 1 = yes)",
    "GenHlth": "Self-rated general health (1 = excellent ... 5 = poor)",
    "MentHlth": "Days of poor mental health in past 30 days (0-30)",
    "PhysHlth": "Days of poor physical health in past 30 days (0-30)",
    "DiffWalk": "Serious difficulty walking or climbing stairs (0 = no, 1 = yes)",
    "Sex": "Sex (0 = female, 1 = male)",
    "Age": "Age group, 13 levels (1 = 18-24, 2 = 25-29, ..., 12 = 75-79, 13 = 80+)",
    "Education": "Education level (1 = never attended/kindergarten ... 6 = college graduate)",
    "Income": "Annual household income (1 = < $10k ... 8 = >= $75k)",
}

UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"


class DataValidationError(ValueError):
    """Raised when the dataset does not match the documented schema."""


def _abs(cfg: dict, p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else Path(cfg["_root"]) / p


# --------------------------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------------------------
def _from_uci(uci_id: int, logger) -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo

        logger.info(f"Fetching UCI dataset id={uci_id} with ucimlrepo ...")
        ds = fetch_ucirepo(id=uci_id)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        return df
    except Exception as exc:  # network / client failure -> try the static CSV
        logger.warning(f"ucimlrepo failed ({exc!r}); trying direct CSV {UCI_CSV_URL}")
        import urllib.request

        with urllib.request.urlopen(UCI_CSV_URL, timeout=120) as resp:  # noqa: S310 (fixed https URL)
            return pd.read_csv(io.BytesIO(resp.read()))


def acquire(cfg: dict) -> tuple[pd.DataFrame, Path]:
    """Download (or read) the raw binary-label release and store an untouched copy in data/raw."""
    logger = get_logger("data", resolve(cfg, "experiments_dir") / "logs")
    raw_dir = resolve(cfg, "raw_dir")
    dcfg = cfg["data"]
    if dcfg.get("source", "uci") == "local":
        src = _abs(cfg, dcfg["local_binary_csv"])
        if not src.exists():
            raise FileNotFoundError(f"local_binary_csv not found: {src}")
        df = pd.read_csv(src)
        origin = f"local file {src}"
    else:
        df = _from_uci(int(dcfg.get("uci_id", 891)), logger)
        origin = f"UCI Machine Learning Repository id {dcfg.get('uci_id', 891)}"
    out = raw_dir / "diabetes_binary_raw.csv"
    df.to_csv(out, index=False)
    logger.info(f"Raw data saved to {out} ({len(df):,} rows) from {origin}")
    save_json({"origin": origin, "rows": len(df), "columns": list(df.columns),
               "sha256_file": hashlib.sha256(out.read_bytes()).hexdigest()}, raw_dir / "raw_provenance.json")
    return df, out


# --------------------------------------------------------------------------------------------
# Validation and canonicalisation
# --------------------------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, expected_rows: int | None = None) -> dict:
    """Check columns, missingness, integer-valuedness and value domains. Raises on violations."""
    report: dict = {"problems": [], "warnings": []}
    cols = set(df.columns)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in cols]
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")
    extra = sorted(cols - set(FEATURES) - {TARGET})
    if extra:
        report["warnings"].append(f"Extra columns ignored: {extra}")
    sub = df[FEATURES + [TARGET]]
    n_na = int(sub.isna().sum().sum())
    report["n_missing_cells"] = n_na
    if n_na:
        report["problems"].append(f"{n_na} missing cells")
    non_int = [c for c in sub.columns if not np.all(np.mod(sub[c].dropna().to_numpy(dtype=float), 1) == 0)]
    if non_int:
        report["problems"].append(f"Non-integer values in {non_int}")
    for f, (lo, hi) in {**DOMAINS, TARGET: (0, 1)}.items():
        v = sub[f].dropna()
        if (v < lo).any() or (v > hi).any():
            report["problems"].append(f"{f} outside [{lo}, {hi}]: min={v.min()}, max={v.max()}")
    report["n_rows"] = int(len(df))
    if expected_rows is not None and len(df) != expected_rows:
        report["warnings"].append(f"Row count {len(df):,} differs from documented {expected_rows:,}")
    if report["problems"]:
        raise DataValidationError("; ".join(report["problems"]))
    return report


def canonicalise(df: pd.DataFrame, sort: bool = True, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Integer-typed frame with columns [features..., target] (+extras), rows sorted canonically.

    Sorting by (features, target) with a stable sort makes the split independent of the row
    order of the downloaded file, so the UCI and Kaggle copies yield identical splits.
    """
    extra_cols = extra_cols or []
    out = df[FEATURES + [TARGET] + extra_cols].astype(np.int64).copy()
    if sort:
        out = out.sort_values(FEATURES + [TARGET], kind="mergesort").reset_index(drop=True)
    return out


def content_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the canonical (sorted, integer) features+target table."""
    canon = canonicalise(df, sort=True)
    return hashlib.sha256(canon.to_csv(index=False).encode()).hexdigest()


def label_mapping_check(binary_df: pd.DataFrame, three_class_df: pd.DataFrame) -> dict:
    """Verify how the binary label relates to the 3-class release (0 none, 1 prediabetes, 2 diabetes)."""
    if len(binary_df) != len(three_class_df):
        raise DataValidationError("Binary and 3-class files differ in length")
    same = bool((binary_df[FEATURES].to_numpy() == three_class_df[FEATURES].to_numpy()).all())
    if not same:
        raise DataValidationError("Binary and 3-class files are not row-aligned (features differ)")
    ct = pd.crosstab(three_class_df["Diabetes_012"].astype(int), binary_df[TARGET].astype(int))
    return {
        "rows_aligned": same,
        "crosstab_012_vs_binary": {f"012={i}": {f"bin={j}": int(ct.loc[i, j]) if j in ct.columns else 0
                                                 for j in [0, 1]} for i in ct.index},
        "prediabetes_in_binary_class": int(binary_df.loc[three_class_df["Diabetes_012"] == 1, TARGET].mode()[0]),
    }


def duplicate_report(df: pd.DataFrame) -> dict:
    full_dup = int(df.duplicated(FEATURES + [TARGET]).sum())
    feat_dup = int(df.duplicated(FEATURES).sum())
    g = df.groupby(FEATURES, sort=False)[TARGET].nunique()
    return {"duplicate_rows": full_dup, "duplicate_feature_vectors": feat_dup,
            "feature_vectors_with_conflicting_labels": int((g > 1).sum()),
            "distinct_feature_vectors": int(len(g))}


def prepare(cfg: dict) -> dict:
    """Acquire -> validate -> canonicalise -> save processed table + provenance metadata."""
    logger = get_logger("data", resolve(cfg, "experiments_dir") / "logs")
    raw, _ = acquire(cfg)
    if "ID" in raw.columns:
        raw = raw.drop(columns=["ID"])
    report = validate_schema(raw, cfg["data"].get("expected_rows"))
    extra_cols: list[str] = []
    meta: dict = {"validation": report}
    p012 = cfg["data"].get("local_012_csv")
    if p012:
        three = pd.read_csv(_abs(cfg, p012))
        try:
            meta["label_mapping"] = label_mapping_check(raw, three)
            raw = raw.copy()
            raw["prediabetes"] = (three["Diabetes_012"].astype(int) == 1).astype(int).to_numpy()
            extra_cols = ["prediabetes"]
            logger.info(f"Label mapping check: {meta['label_mapping']}")
        except DataValidationError as exc:
            logger.warning(f"3-class file not usable ({exc}); prediabetes sensitivity analysis disabled. "
                           "Use the binary and 3-class CSVs from the same (Kaggle) release.")
    canon = canonicalise(raw, sort=cfg["split"].get("canonical_sort", True), extra_cols=extra_cols)
    meta["content_sha256"] = content_hash(canon)
    meta["duplicates"] = duplicate_report(canon)
    meta["n_rows"] = len(canon)
    meta["prevalence"] = float(canon[TARGET].mean())
    out_dir = resolve(cfg, "processed_dir")
    canon.to_csv(out_dir / "brfss2015_diabetes.csv", index=False)
    save_json(meta, out_dir / "metadata.json")
    logger.info(f"Processed data: {len(canon):,} rows, prevalence {meta['prevalence']:.4f}, "
                f"hash {meta['content_sha256'][:16]}...")
    return meta


# --------------------------------------------------------------------------------------------
# Loading and splitting
# --------------------------------------------------------------------------------------------
def load_processed(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg, "processed_dir") / "brfss2015_diabetes.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/01_get_data.py")
    return pd.read_csv(path)


def make_splits(cfg: dict) -> dict:
    """Stratified held-out test split + stratified K-fold assignment of the development set."""
    df = load_processed(cfg)
    seed = int(cfg["project"]["seed"])
    y = df[TARGET].to_numpy()
    idx = np.arange(len(df))
    n_sub = cfg["data"].get("subsample_rows")
    if n_sub:  # smoke runs only
        idx, _ = train_test_split(idx, train_size=int(n_sub), stratify=y, random_state=derive_seed(seed, "subsample"))
        idx = np.sort(idx)
    dev_idx, test_idx = train_test_split(idx, test_size=cfg["split"]["test_size"], stratify=y[idx],
                                         random_state=derive_seed(seed, "test_split"))
    dev_idx, test_idx = np.sort(dev_idx), np.sort(test_idx)
    folds = np.full(len(dev_idx), -1, dtype=int)
    skf = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=derive_seed(seed, "cv_folds"))
    for k, (_, va) in enumerate(skf.split(dev_idx, y[dev_idx])):
        folds[va] = k
    assert np.intersect1d(dev_idx, test_idx).size == 0, "dev/test overlap"
    assert (folds >= 0).all()
    out = resolve(cfg, "experiments_dir") / "splits"
    out.mkdir(parents=True, exist_ok=True)
    meta = load_json(resolve(cfg, "processed_dir") / "metadata.json")
    np.savez(out / "splits.npz", dev_idx=dev_idx, test_idx=test_idx, folds=folds)
    info = {"n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
            "prev_dev": float(y[dev_idx].mean()), "prev_test": float(y[test_idx].mean()),
            "fold_sizes": np.bincount(folds).tolist(),
            "fold_prevalence": [float(y[dev_idx][folds == k].mean()) for k in range(folds.max() + 1)],
            "content_sha256": meta["content_sha256"]}
    save_json(info, out / "split_info.json")
    return info


def load_split(cfg: dict, part: str):
    """Return (X, y, extra) for part in {'dev', 'test'}; test access is logged."""
    logger = get_logger("data", resolve(cfg, "experiments_dir") / "logs")
    df = load_processed(cfg)
    sp = np.load(resolve(cfg, "experiments_dir") / "splits" / "splits.npz")
    if part not in ("dev", "test"):
        raise ValueError(part)
    if part == "test":
        logger.info("TEST SET ACCESSED")
    idx = sp[f"{part}_idx"]
    sub = df.iloc[idx].reset_index(drop=True)
    X = sub[FEATURES].astype(float)
    y = sub[TARGET].to_numpy().astype(int)
    extra = sub.drop(columns=FEATURES + [TARGET])
    if part == "dev":
        extra = extra.assign(fold=sp["folds"])
    return X, y, extra
