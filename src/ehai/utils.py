"""Logging, timing, and small I/O helpers."""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info(f"START {label}")
    try:
        yield
    finally:
        logger.info(f"END   {label} ({time.time() - t0:.1f}s)")


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_NpEncoder)


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def environment_info() -> dict:
    import importlib

    pkgs = {}
    for name in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "shap", "matplotlib"]:
        try:
            pkgs[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover
            pkgs[name] = "not installed"
    import os

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
    }


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)
