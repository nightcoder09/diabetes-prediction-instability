"""Configuration loading with single-level inheritance and derived seeds."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may declare ``inherit: other.yaml`` (same folder)."""
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_update(load_config(path.parent / parent), cfg)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for a ``paths`` entry, created if missing."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def derive_seed(master: int, *tags: Any) -> int:
    """Deterministic sub-seed from the master seed and string tags (order-sensitive)."""
    h = hashlib.sha256(("|".join([str(master)] + [str(t) for t in tags])).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def n_jobs(cfg: dict) -> int:
    import os

    j = int(cfg.get("models", {}).get("n_jobs", -1))
    return (os.cpu_count() or 1) if j == -1 else j
