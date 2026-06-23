from __future__ import annotations

import numpy as np
import pandas as pd

from ml_signal.pipelines.feature_builder import (
    build_feature_dataframe,
    calculate_targets,
    get_feature_columns,
    load_vnindex_data,
)


def build_latest_training_frame(
    ticker: str,
    feature_set: str,
    lookahead: int,
    tp: float,
    sl: float,
    min_training_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Timestamp]:
    """
    Build feature, labeled training, and latest-signal frames.

    The latest complete feature row is selected as the signal row. Training rows
    are restricted to dates before the latest signal date to avoid training on
    the row being scored.
    """
    df_vnindex = load_vnindex_data()
    df_features = build_feature_dataframe(ticker, df_vnindex)
    feature_cols = get_feature_columns(df_features, feature_set)

    latest_candidates = (
        df_features
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + ["Close"])
    )

    if latest_candidates.empty:
        raise ValueError("No latest row has a complete feature vector.")

    latest_feature_date = latest_candidates.index.max()

    df_labeled = calculate_targets(
        df_features,
        lookahead=lookahead,
        tp=tp,
        sl=sl,
    )

    required_cols = feature_cols + [
        "Target",
        "Exit_Return",
        "Holding_Days",
        "Timeout",
    ]

    df_labeled = (
        df_labeled
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_cols)
        .sort_index()
    )

    df_train = df_labeled[df_labeled.index < latest_feature_date].copy()

    if len(df_train) < min_training_rows:
        raise ValueError(
            f"Insufficient labeled training rows before latest signal date: {len(df_train)}"
        )

    if df_train["Target"].nunique() < 2:
        raise ValueError("Training target has only one class.")

    return df_features, df_train, feature_cols, latest_feature_date


__all__ = [
    "build_latest_training_frame",
]
