"""Baseline model comparison and trade-off evaluation."""

from typing import Dict, Any, List
import pandas as pd


def generate_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate comparative summary table across baseline vs gated architectures."""
    return pd.DataFrame(results)

