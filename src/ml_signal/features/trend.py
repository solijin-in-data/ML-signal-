from __future__ import annotations

import pandas as pd


TREND_QUALITY_COLUMNS = [
    "ER_10",
    "EMA_13_Slope",
    "EMA_21_55_Gap",
    "BB_Position",
]


def add_trend_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    if "EMA_13" not in df.columns:
        df["EMA_13"] = df["Close"].ewm(span=13, adjust=False).mean()

    if "EMA_21" not in df.columns:
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    df["EMA_55"] = df["Close"].ewm(span=55, adjust=False).mean()

    df["EMA_13_Slope"] = df["EMA_13"].pct_change()

    df["EMA_21_55_Gap"] = (
        (df["EMA_21"] - df["EMA_55"])
        / (df["EMA_55"] + 1e-9)
    )

    df["ER_10"] = (
        (df["Close"] - df["Close"].shift(10)).abs()
        / (df["Close"].diff().abs().rolling(10).sum() + 1e-9)
    )

    if "BB_Mid" not in df.columns:
        df["BB_Mid"] = df["Close"].rolling(window=20).mean()

    if "BB_Std" not in df.columns:
        df["BB_Std"] = df["Close"].rolling(window=20).std(ddof=0)

    df["BB_Position"] = (
        (df["Close"] - df["BB_Mid"])
        / (2 * df["BB_Std"] + 1e-9)
    )

    return df
