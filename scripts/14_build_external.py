#!/usr/bin/env python
"""Step 14 - rebuild the 21-predictor dataset from raw CDC BRFSS files for several survey years.

1. 2015 is rebuilt from the raw CDC file and compared with the published extract (validation of
   the derivation: same rows, same values, same canonical content hash).
2. Later years (default 2021 and 2023) are built with the identical rules and saved for temporal
   external validation. 2021 carries every item of the 2015 extract; 2023 did not ask the fruit and
   vegetable items, which are left missing here and imputed with development medians downstream.

Raw files: data/raw/brfss/LLCP{year}XPT.zip, from
https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip  (see README, section 2, for the download commands)

    python scripts/14_build_external.py --years 2015 2021 2023
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ehai.brfss_raw import build_year  # noqa: E402
from ehai.config import load_config, resolve  # noqa: E402
from ehai.data import FEATURES, TARGET, canonicalise, content_hash  # noqa: E402
from ehai.utils import get_logger, save_json  # noqa: E402


def compare_with_published(rebuilt: pd.DataFrame, published: pd.DataFrame) -> dict:
    a = canonicalise(rebuilt, sort=True)
    b = canonicalise(published, sort=True)
    out = {"rows_rebuilt": int(len(a)), "rows_published": int(len(b)),
           "hash_rebuilt": content_hash(a), "hash_published": content_hash(b)}
    out["identical"] = out["hash_rebuilt"] == out["hash_published"]
    out["column_mean_diff"] = {c: float(a[c].mean() - b[c].mean()) for c in FEATURES + [TARGET]}
    if len(a) == len(b):
        out["cells_differing"] = int((a.to_numpy() != b.to_numpy()).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2021, 2023])
    ap.add_argument("--raw-dir", default="data/raw/brfss")
    ap.add_argument("--out-dir", default="data/external")
    ap.add_argument("--workdir", default=None, help="scratch folder for the ~1 GB extracted .XPT files")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_root"])
    logger = get_logger("external_build", resolve(cfg, "experiments_dir") / "logs")
    raw_dir, out_dir = root / args.raw_dir, root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="brfss_"))
    work.mkdir(parents=True, exist_ok=True)
    reports = {}
    for year in args.years:
        z = raw_dir / f"LLCP{year}XPT.zip"
        if not z.exists():
            logger.warning(f"{z} not found - skipping {year} (see README, section 2, to download it)")
            continue
        local_zip = work / z.name
        if not local_zip.exists():
            local_zip.write_bytes(z.read_bytes())
        logger.info(f"Building {year} from {z.name} ...")
        clean, rep = build_year(local_zip, year, workdir=work)
        cols = [TARGET, "Diabetes_012"] + FEATURES
        clean[cols].to_csv(out_dir / f"brfss{year}_diabetes.csv", index=False)
        logger.info(f"{year}: {rep['raw_rows']:,} raw -> {rep['clean_rows']:,} complete rows, "
                    f"prevalence {rep['prevalence']:.4f}, sources {rep['sources']}")
        if year == 2015:
            pub_path = resolve(cfg, "processed_dir") / "brfss2015_diabetes.csv"
            if pub_path.exists():
                pub = pd.read_csv(pub_path)
                rep["validation_vs_published"] = compare_with_published(clean, pub)
                logger.info(f"2015 rebuilt vs published: {rep['validation_vs_published']['identical']} "
                            f"({rep['validation_vs_published']['rows_rebuilt']:,} vs "
                            f"{rep['validation_vs_published']['rows_published']:,} rows)")
        reports[str(year)] = rep
    save_json(reports, out_dir / "build_report.json")
    print(pd.DataFrame({y: {"raw": r["raw_rows"], "complete": r["clean_rows"], "prevalence": round(r["prevalence"], 4)}
                        for y, r in reports.items()}).T.to_string())


if __name__ == "__main__":
    main()
