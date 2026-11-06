"""Utility helpers for configuration, paths, and platform detection."""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.
    
    Checks environment variable `PROJECT_ROOT` first (useful for Kaggle/Colab),
    otherwise resolves relative to this file.
    """
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    # src/utils/config.py -> parents[2] is root
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML experiment configuration file."""
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "experiment_config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

