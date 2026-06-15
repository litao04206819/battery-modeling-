"""Small helpers for loading YAML configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root: src/bmr/config.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    A bare filename (no directory component) is resolved against ``configs/``.
    """
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        p = CONFIG_DIR / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
