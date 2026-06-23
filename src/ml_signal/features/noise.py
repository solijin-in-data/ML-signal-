from __future__ import annotations

import numpy as np
import pandas as pd


NOISE_FILTER_COLUMNS = [
    "Volatility_Z_60",
    "Downside_Volatility_20",
    "Trend_Efficiency_20",
    "ER_20",
    "ER_30",
]


def add_noise_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    if "Volatility" not in df.columns:
        df["Volatility"] = df["Log_Return"].rolling(20).std()

    vol_mean_60 = df["Volatility"].rolling(60).mean()
    vol_std_60 = df["Volatility"].rolling(60).std()

    df["Volatility_Z_60"] = (
        (df["Volatility"] - vol_mean_60)
        / (vol_std_60 + 1e-9)
    )

    downside_returns = df["Log_Return"].where(df["Log_Return"] < 0, 0.0)
    df["Downside_Volatility_20"] = downside_returns.rolling(20).std()

    abs_path_20 = df["Close"].diff().abs().rolling(20).sum()
    abs_path_30 = df["Close"].diff().abs().rolling(30).sum()

    df["Trend_Efficiency_20"] = (
        (df["Close"] - df["Close"].shift(20)).abs()
        / (abs_path_20 + 1e-9)
    )

    df["ER_20"] = df["Trend_Efficiency_20"]

    df["ER_30"] = (
        (df["Close"] - df["Close"].shift(30)).abs()
        / (abs_path_30 + 1e-9)
    )

    return df
