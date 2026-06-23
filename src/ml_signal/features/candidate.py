from __future__ import annotations

import numpy as np
import pandas as pd

from ml_signal.features.momentum import add_momentum_features
from ml_signal.features.trend import add_trend_quality_features
from ml_signal.features.volume import add_volume_features
from ml_signal.features.recovery import add_recovery_features


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add experimental feature families on top of baseline features.

    All features use current and past information only.
    """
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df = add_momentum_features(df)
    df = add_trend_quality_features(df)
    df = add_volume_features(df)
    df = add_recovery_features(df)

    return df.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "add_candidate_features",
]
