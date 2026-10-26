"""YAML loading for the simulation config, world and mission files."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
WORLDS_DIR = REPO_ROOT / "worlds"
MISSIONS_DIR = REPO_ROOT / "missions"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(path: str | Path, default_dir: Path, suffix: str = ".yaml") -> Path:
    """Accept either a full path or a bare name such as 'investor_demo'."""
    p = Path(path)
    if p.exists():
        return p
    candidate = default_dir / (p.name if p.suffix else p.name + suffix)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find {path} (also looked in {default_dir})")


def load_sim_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = load_yaml(CONFIG_DIR / "sim.yaml")
    if path:
        cfg = deep_merge(cfg, load_yaml(path))
    return deep_merge(cfg, overrides or {})
