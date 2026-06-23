from __future__ import annotations

import numpy as np
import pandas as pd


RECOVERY_COLUMNS = [
    "Drawdown_60",
    "Distance_52W_High",
    "Distance_52W_Low",
]


RECOVERY_QUALITY_COLUMNS = [
    "Recovery_20_From_60D_Low",
    "Recovery_Slope_10",
    "Days_Since_60D_Low",
    "Reclaim_EMA55",
    "Dist_EMA55",
    "Higher_Low_20",
    "Drawdown_Recovery_Ratio",
]


def add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    rolling_high_60 = df["Close"].rolling(60).max()
    rolling_high_252 = df["Close"].rolling(252).max()
    rolling_low_252 = df["Close"].rolling(252).min()

    df["Drawdown_60"] = (
        df["Close"] / (rolling_high_60 + 1e-9)
        - 1
    )

    df["Distance_52W_High"] = (
        df["Close"] / (rolling_high_252 + 1e-9)
        - 1
    )

    df["Distance_52W_Low"] = (
        df["Close"] / (rolling_low_252 + 1e-9)
        - 1
    )

    return df


def _days_since_rolling_low(values: np.ndarray) -> float:
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.nan

    low_position = int(np.nanargmin(values))
    return float(len(values) - 1 - low_position)


def add_recovery_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    rolling_low_60 = df["Close"].rolling(60).min()
    rolling_low_20 = df["Low"].rolling(20).min() if "Low" in df.columns else df["Close"].rolling(20).min()
    prior_rolling_low_20 = rolling_low_20.shift(20)

    df["Recovery_20_From_60D_Low"] = (
        df["Close"] / (rolling_low_60 + 1e-9)
        - 1
    )

    df["Recovery_Slope_10"] = (
        df["Close"].pct_change(10)
        / 10
    )

    df["Days_Since_60D_Low"] = df["Close"].rolling(60).apply(
        _days_since_rolling_low,
        raw=True,
    )

    ema55 = df["Close"].ewm(span=55, adjust=False).mean()
    df["Reclaim_EMA55"] = (df["Close"] > ema55).astype(float)
    df["Dist_EMA55"] = (df["Close"] - ema55) / (ema55 + 1e-9)

    df["Higher_Low_20"] = (
        rolling_low_20 > prior_rolling_low_20
    ).astype(float)

    drawdown_abs = df.get("Drawdown_60", pd.Series(index=df.index, dtype=float)).abs()
    df["Drawdown_Recovery_Ratio"] = (
        df["Recovery_20_From_60D_Low"]
        / (drawdown_abs + 1e-9)
    )

    return df
