from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_ROOT]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


# Transitional bridge:
# Feature construction and legacy data paths are still sourced from the mature
# research runner. The production engine should import this module instead of
# importing feature_experiment_runner directly.
import feature_experiment_runner as exp  # noqa: E402


def load_vnindex_data() -> pd.DataFrame:
    df_vnindex = exp.radar.load_data(exp.VNINDEX_FILE, is_vnindex=True)

    if df_vnindex is None:
        raise ValueError(f"VNINDEX data could not be loaded: {exp.VNINDEX_FILE}")

    return df_vnindex


def build_feature_dataframe(ticker: str, df_vnindex: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_vnindex is None:
        df_vnindex = load_vnindex_data()

    return exp.build_feature_dataframe(ticker, df_vnindex)


def get_feature_sets() -> dict[str, list[str]]:
    return exp.get_feature_sets()


def validate_feature_columns(
    df_features: pd.DataFrame,
    requested_columns: list[str],
) -> list[str]:
    return exp.validate_feature_columns(df_features, requested_columns)


def get_feature_columns(
    df_features: pd.DataFrame,
    feature_set: str,
) -> list[str]:
    feature_sets = get_feature_sets()

    if feature_set not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(
            f"Unknown feature_set={feature_set}. Available feature sets: {available}"
        )

    return validate_feature_columns(df_features, feature_sets[feature_set])


def calculate_targets(
    df_features: pd.DataFrame,
    lookahead: int,
    tp: float,
    sl: float,
) -> pd.DataFrame:
    return exp.radar.calculate_targets(
        df_features,
        lookahead=lookahead,
        tp=tp,
        sl=sl,
    )


__all__ = [
    "build_feature_dataframe",
    "calculate_targets",
    "get_feature_columns",
    "get_feature_sets",
    "load_vnindex_data",
    "validate_feature_columns",
]
